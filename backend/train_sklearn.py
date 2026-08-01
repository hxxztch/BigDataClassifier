import os, json, time, pickle, glob, shutil, sys
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

BACKEND = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BACKEND), "models")
DATA_DIR = os.path.join(BACKEND, "datasets")

from utils.config import get_spark_builder
from utils.preprocessing import clean_column_names, custom_preprocessing, build_feature_preprocessing_stages
from utils.logger import get_logger
from utils.version_manager import next_version as _next_ver, register as _reg_ver

logger = get_logger("sklearn_trainer")


def train_sklearn(file_path, scene_id=None, spark=None, progress_callback=None):
    task_name = scene_id or os.path.splitext(os.path.basename(file_path))[0]
    if task_name.endswith("_clean"):
        task_name = task_name.replace("_clean", "")

    # Resolve target column from scenes.yaml or DATASET_META
    target_col = None
    if scene_id:
        try:
            import yaml
            sy = os.path.join(BACKEND, "scenes.yaml")
            with open(sy, "r", encoding="utf-8") as f:
                sc = yaml.safe_load(f) or {}
            target_col = (sc.get("scenes", {}).get(scene_id, {}) or {}).get("target_col")
        except:
            pass
    if not target_col:
        from utils.config import DATASET_META
        target_col = DATASET_META.get(task_name)
    if not target_col:
        raise ValueError(f"No target column configured for scene: {task_name}")

    logger.info(f"\n{'='*60}\nProcessing: [{task_name}] (target={target_col})")

    _spark_owned = spark is None
    if _spark_owned:
        spark = get_spark_builder(app_name="SklearnTrain", driver_memory="4g").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path) if not file_path.endswith(".parquet") else spark.read.parquet(file_path)
        df = clean_column_names(df)
        df = df.fillna(0).fillna("Unknown")
        df = custom_preprocessing(df, task_name)
        if progress_callback: progress_callback(10, '数据预处理完成')

        target_col_clean = target_col.replace(".", "_").replace(" ", "_")
        if target_col_clean not in df.columns:
            raise ValueError(f"Target column '{target_col_clean}' not found in uploaded CSV")

        logger.info(f"Target column locked: {target_col_clean}")

        feature_cols = [c for c in df.columns if c != target_col_clean and c.lower() not in ("id", "user_id", "order_id", "unnamed: 0", "_c0")]

        # Build and save preprocessing pipeline
        stages = build_feature_preprocessing_stages(df, feature_cols, target_col_clean)
        from pyspark.ml import Pipeline
        prep_pipeline = Pipeline(stages=stages)
        prep_model = prep_pipeline.fit(df)

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        prep_save = os.path.join(MODELS_DIR, f"{task_name}_preprocessing.model")
        if os.path.exists(prep_save):
            shutil.rmtree(prep_save)
        prep_model.write().overwrite().save(prep_save)
        if progress_callback: progress_callback(20, '特征工程完成')
        logger.info(f"  Preprocessing saved: {prep_save}")

        # ==================================================================
        total_rows = df.count()  # computed before MLlib for metadata
        # Phase 1: Spark MLlib distributed training (RF / GBT / NB)
        # Trains on FULL dataset using Spark-native classifiers.
        # Each saves as complete PipelineModel for single-call prediction.
        # ==================================================================
        from pyspark.ml.classification import (
            RandomForestClassifier as SparkRF,
            GBTClassifier as SparkGBT,
            NaiveBayes as SparkNB,
        )
        from pyspark.ml import Pipeline as MLlibPipeline
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        _train_ver = _next_ver(MODELS_DIR, task_name)
        _train_base = os.path.join(MODELS_DIR, "v" + str(_train_ver)) if _train_ver > 1 else MODELS_DIR
        if _train_ver > 1 and not os.path.exists(_train_base):
            os.makedirs(_train_base)

        mllib_classifiers = {
            "random_forest": SparkRF(
                featuresCol="features", labelCol="label",
                numTrees=100, maxDepth=10, seed=42,
            ),
            "gbdt": SparkGBT(
                featuresCol="features", labelCol="label",
                maxIter=100, maxDepth=5, seed=42,
            ),
            "naive_bayes": SparkNB(
                featuresCol="features", labelCol="label",
            ),
        }

        mllib_results = {}
        for _idx, (algo_name, spark_clf) in enumerate(mllib_classifiers.items()):
            if progress_callback:
                progress_callback(22 + _idx * 2, f"Spark MLlib: {algo_name}...")

            mllib_save = os.path.join(
                _train_base, f"{task_name}_{algo_name}_mllib"
            )
            if os.path.exists(mllib_save):
                shutil.rmtree(mllib_save)

            logger.info(f"  [MLlib] Training {algo_name} on full dataset...")
            t0 = time.time()
            try:
                full_stages = list(stages) + [spark_clf]
                full_pipeline = MLlibPipeline(stages=full_stages)
                full_model = full_pipeline.fit(df)

                train_preds = full_model.transform(df)
                acc_eval = MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction",
                    metricName="accuracy"
                )
                f1_eval = MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction",
                    metricName="f1"
                )
                train_acc = acc_eval.evaluate(train_preds)
                train_f1 = f1_eval.evaluate(train_preds)

                full_model.write().overwrite().save(mllib_save)

                elapsed = time.time() - t0
                mllib_results[algo_name] = {
                    "accuracy": round(float(train_acc), 4),
                    "f1_score": round(float(train_f1), 4),
                }
                logger.info(
                    f"    [MLlib] {algo_name} -> F1={train_f1*100:.2f}% "
                    f"Acc={train_acc*100:.2f}% ({elapsed:.1f}s)"
                )

                # Save metadata
                mllib_meta = {
                    "model_type": algo_name,
                    "scene": task_name,
                    "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "f1_score": round(float(train_f1), 4),
                    "accuracy": round(float(train_acc), 4),
                    "train_size": total_rows,
                    "num_classes": len(df.select("label").distinct().collect()),
                    "backend": "Spark MLlib (distributed)",
                }
                with open(os.path.join(mllib_save, "metadata.json"), "w") as mf:
                    json.dump(mllib_meta, mf, indent=2)

            except Exception as ml_err:
                logger.error(f"    [MLlib] {algo_name} FAILED: {ml_err}")
                mllib_results[algo_name] = {"accuracy": 0.0, "f1_score": 0.0}

        if progress_callback:
            progress_callback(28, "MLlib finished")
        logger.info(f"  MLlib phase complete: {len(mllib_results)} models")

        # ==================================================================
        # Phase 2: sklearn single-machine training (XGBoost + calibration)
        # ==================================================================

        # Transform to pandas (sample to 30k rows max to avoid OOM)
        final_df_full = prep_model.transform(df).select("features", "label")
        total_rows = final_df_full.count()
        MAX_SAMPLE = 30000
        if total_rows > MAX_SAMPLE:
            final_df = final_df_full.sample(fraction=MAX_SAMPLE/total_rows, seed=42)
        else:
            final_df = final_df_full
        pdf = final_df.toPandas()
        # Convert features: handles Vector, string reprs, and numpy arrays
        def _safe_feat(v):
            if hasattr(v, 'toArray'):
                return v.toArray()
            if isinstance(v, dict):
                if 'values' in v:
                    if v.get('type') == 1:
                        arr = np.zeros(v.get('size', 0))
                        arr[v['indices']] = v['values']
                        return arr
                    return np.asarray(v['values'], dtype=np.float64)
                return np.zeros(0)
            if isinstance(v, str):
                import re
                m = re.match(r'\((\d+),\s*\[(.*?)\],\s*\[(.*?)\]\)', v)
                if m:
                    size = int(m.group(1))
                    idx = [int(i.strip()) for i in m.group(2).split(',') if i.strip()]
                    vals = [float(x.strip()) for x in m.group(3).split(',') if x.strip()]
                    arr = np.zeros(size)
                    if idx:
                        arr[idx] = vals
                    return arr
                try:
                    return np.array(eval(v))
                except:
                    pass
            return np.asarray(v, dtype=np.float64)
        X = np.asarray([_safe_feat(v) for v in pdf["features"]], dtype=np.float64)
        if progress_callback: progress_callback(30, '特征提取完成，开始训练模型')
        y = pdf["label"].values.astype(int)

        row_count = len(pdf)
        num_classes = len(np.unique(y))
        logger.info(f"  Data: {row_count} rows, {num_classes} classes")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train = X_train[:5000], y_train[:5000]

        # Stop Spark BEFORE sklearn training to avoid JVM/sklearn native memory conflict
        if _spark_owned:
            try:
                spark.stop()
            except Exception:
                pass

        classifiers = {
            "random_forest": RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=20, random_state=42, n_jobs=1),
            "gbdt": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "xgboost": XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, reg_lambda=1.0, reg_alpha=0.5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", verbosity=0),
            "naive_bayes": GaussianNB(),
        }

        results = {}
        _train_ver = _next_ver(MODELS_DIR, task_name)
        _train_base = os.path.join(MODELS_DIR, "v" + str(_train_ver)) if _train_ver > 1 else MODELS_DIR
        if _train_ver > 1 and not os.path.exists(_train_base):
            os.makedirs(_train_base)

        for algo_name, clf in classifiers.items():
            if progress_callback: progress_callback(40 + list(classifiers.keys()).index(algo_name) * 15, f'训练中: {algo_name}...')
            save_path = os.path.join(_train_base, f"{task_name}_{algo_name}.model")
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)

            logger.info(f"  Training: {algo_name} ...")
            start = time.time()

            try:
                clf.fit(X_train, y_train)
            except Exception as fit_err:
                logger.error(f"    {algo_name} FIT FAILED: {fit_err}")
                continue

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)

            calibrator = None
            best_threshold = 0.5
            if y_proba.shape[1] >= 2:
                try:
                    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                    calibrator.fit(y_proba[:, 1].reshape(-1, 1), y_test)
                    best_f1_th = 0.0
                    for th in [round(x, 2) for x in [i * 0.05 for i in range(1, 19)]]:
                        y_pred_th = (y_proba[:, 1] >= th).astype(int)
                        tp = ((y_pred_th == 1) & (y_test == 1)).sum()
                        fp = ((y_pred_th == 1) & (y_test == 0)).sum()
                        fn = ((y_pred_th == 0) & (y_test == 1)).sum()
                        f1_th = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                        if f1_th > best_f1_th:
                            best_f1_th = f1_th
                            best_threshold = th
                    logger.info(f"    Optimal threshold: {best_threshold:.2f} (calibrated)")
                except Exception as ce:
                    logger.warning(f"    Calibration skipped: {ce}")

            with open(os.path.join(save_path, "sklearn_model.pkl"), "wb") as f:
                pickle.dump(clf, f)
            if calibrator is not None:
                with open(os.path.join(save_path, "calibrator.pkl"), "wb") as f:
                    pickle.dump(calibrator, f)
            with open(os.path.join(save_path, "threshold.txt"), "w") as f:
                f.write(str(round(best_threshold, 2)))

            meta = {
                "model_type": algo_name, "scene": task_name,
                "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "f1_score": round(float(f1), 4), "accuracy": round(float(acc), 4),
                "train_size": row_count, "num_classes": num_classes,
                "optimal_threshold": round(float(best_threshold), 2),
                "calibrator_a": round(float(calibrator.coef_[0][0]), 4) if calibrator is not None else None,
                "calibrator_b": round(float(calibrator.intercept_[0]), 4) if calibrator is not None else None,
            }
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            results[algo_name] = {"accuracy": round(float(acc), 4), "f1_score": round(float(f1), 4), "train_size": row_count}
            elapsed = time.time() - start
            logger.info(f"    {algo_name} F1: {f1*100:.2f}% (Acc: {acc*100:.2f}%) - {elapsed:.1f}s")

        if progress_callback: progress_callback(90, '保存结果中...')
        _reg_ver(MODELS_DIR, task_name, _train_ver, results, dataset=task_name + ".csv", rows=row_count)
        logger.info(f"  Registered version {_train_ver} ({len(results)} models)")

        # Safely update required_cols in scenes.yaml from training features
        _scenes_yaml = os.path.join(BACKEND, "scenes.yaml")
        if os.path.isfile(_scenes_yaml) and feature_cols:
            try:
                import yaml as _yaml
                with open(_scenes_yaml, "r", encoding="utf-8") as _f:
                    _sc = _yaml.safe_load(_f) or {}
                _scene = (_sc.get("scenes", {}) or {}).get(task_name)
                if _scene:
                    _scene["required_cols"] = feature_cols
                    with open(_scenes_yaml, "w", encoding="utf-8") as _f:
                        _yaml.dump(_sc, _f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                    logger.info(f"  Updated required_cols ({len(feature_cols)} columns)")
            except Exception as _e:
                logger.warning(f"  Failed to update required_cols: {_e}")

    finally:
        pass  # Spark already stopped before training
