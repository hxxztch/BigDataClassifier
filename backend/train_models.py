import os
import glob
import time
import shutil
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import NaiveBayes, GBTClassifier, RandomForestClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark import StorageLevel
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.config import (
    _HAS_GPU,

    MODELS_DIR, DATA_DIR, DATASET_META,
    get_spark_builder, AUTO_CANDIDATES, SIMPLE_CLASSIFIERS,
)
from utils.preprocessing import (
    clean_column_names, custom_preprocessing,
    build_feature_preprocessing_stages,
    validate_and_filter_columns,
)
from utils.logger import get_logger
from utils.version_manager import next_version as _next_ver, register as _reg_ver, get_base as _get_base

logger = get_logger(__name__)


def train_one_file(spark, file_path):
    filename = os.path.basename(file_path)
    task_name = os.path.splitext(filename)[0]
    if task_name.endswith("_clean"):
        task_name = task_name.replace("_clean", "")
    if task_name not in DATASET_META:
        logger.info(f"Skip unconfigured scene: {filename}")
        return

    logger.info(f"\n{'='*60}\nProcessing scene: [{task_name}]")
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path) if not file_path.endswith(".parquet") else spark.read.parquet(file_path)
        row_count = df.count()
        spark.conf.set("spark.sql.shuffle.partitions", str(max(200, min(row_count // 50000, 2000))))
        df = clean_column_names(df)
        df = df.fillna(0).fillna("Unknown")
        df = custom_preprocessing(df, task_name)

        target_col = DATASET_META.get(task_name)
        if target_col:
            target_col = target_col.replace(".", "_").replace(" ", "_")
        if not target_col or target_col not in df.columns:
            logger.error(f"Target column not found: {target_col}")
            return

        logger.info(f"Target column locked: {target_col}")
        feature_cols = [c for c in df.columns if c != target_col and
                        c.lower() not in ["id", "user_id", "order_id", "unnamed: 0", "_c0"]]

        # Build and fit preprocessing pipeline
        stages = build_feature_preprocessing_stages(df, feature_cols, target_col)
        preprocessing_pipeline = Pipeline(stages=stages)
        preprocessing_model = preprocessing_pipeline.fit(df)
        final_df = preprocessing_model.transform(df).select("features", "label")
        final_df.persist(StorageLevel.MEMORY_AND_DISK)

        num_classes = final_df.select("label").distinct().count()
        train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

        # Smart oversampling for imbalanced data
        label_counts = train_data.groupBy("label").count().collect()
        counts = {row["label"]: row["count"] for row in label_counts}
        if len(counts) > 1:
            max_count = max(counts.values())
            min_count = min(counts.values())
            if max_count / min_count > 2:
                logger.info(f"Oversampling (ratio {max_count/min_count:.1f}:1)...")
                dfs = []
                for label_val, count_val in counts.items():
                    label_df = train_data.filter(col("label") == label_val)
                    if count_val < max_count:
                        limit = 50.0 if task_name == "ind_quality" else 10.0
                        ratio = min(max_count / count_val, limit)
                        label_df = label_df.sample(withReplacement=True, fraction=ratio, seed=42)
                    dfs.append(label_df)
                train_data = dfs[0]
                for d in dfs[1:]:
                    train_data = train_data.union(d)
                train_data = train_data.repartition(spark.sparkContext.defaultParallelism)

        # Define classifiers
        gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42, maxBins=128)
        nb = NaiveBayes(labelCol="label", featuresCol="features")
        import logging as _l; _l.getLogger('XGBoost-PySpark').setLevel(_l.WARNING)
        from utils.config import get_xgboost_device
        device_str = get_xgboost_device()
        xgb = SparkXGBClassifier(features_col="features", label_col="label", max_depth=6, n_estimators=100, learning_rate=0.1, reg_lambda=1.0, reg_alpha=0.5, subsample=0.8, colsample_bytree=0.8, num_workers=1, device=device_str)
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42, maxBins=128, maxDepth=8, numTrees=20)
        classifiers = {"xgboost": xgb, "gbdt": gbt, "naive_bayes": nb, "random_forest": rf}
        if num_classes > 2:
            del classifiers["gbdt"]

        evaluator = MulticlassClassificationEvaluator(metricName="f1")
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        # Version management
        _train_ver = _next_ver(MODELS_DIR, task_name)
        _train_base = os.path.join(MODELS_DIR, "v" + str(_train_ver)) if _train_ver > 1 else MODELS_DIR
        if _train_ver > 1 and not os.path.exists(_train_base):
            os.makedirs(_train_base)

        for algo_name, clf in classifiers.items():
            save_path = os.path.join(_train_base, f"{task_name}_{algo_name}.model")
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            logger.info(f"Training: {algo_name} ...")
            start = time.time()

            best_clf_model = clf.fit(train_data)
            preds = best_clf_model.transform(test_data)

            # Platt calibration: fit on validation predictions
            calibrator = None
            if "rawPrediction" in preds.columns:
                try:
                    local = preds.select("rawPrediction", "label").toPandas()
                    X = np.array([v[1] for v in local["rawPrediction"]])
                    y = local["label"].values
                    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                    calibrator.fit(X.reshape(-1, 1), y)
                    logger.info(f"  {algo_name} calibrated (A={calibrator.coef_[0][0]:.4f}, B={calibrator.intercept_[0]:.4f})")

                except Exception as ce:
                    logger.warning(f"  {algo_name} calibration skipped: {ce}")
            # Threshold tuning: find optimal classification threshold
            best_threshold = 0.5
            if "probability" in preds.columns:
                try:
                    local = preds.select("probability", "label").toPandas()
                    probs = np.array([v[1] for v in local["probability"]])
                    y_true = local["label"].values
                    best_f1_th = 0.0
                    for th in [round(x, 2) for x in [i * 0.05 for i in range(1, 19)]]:
                        y_pred = (probs >= th).astype(int)
                        tp = ((y_pred == 1) & (y_true == 1)).sum()
                        fp = ((y_pred == 1) & (y_true == 0)).sum()
                        fn = ((y_pred == 0) & (y_true == 1)).sum()
                        f1_th = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                        if f1_th > best_f1_th:
                            best_f1_th = f1_th
                            best_threshold = th
                    logger.info(f"  Optimal threshold: {best_threshold:.2f} (F1: {best_f1_th*100:.2f}%)")
                except Exception as te:
                    logger.warning(f"  Threshold tuning: {te}")
            f1 = evaluator.evaluate(preds)
            acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(preds)

            # XGBoost model is already a PipelineModel internally -- save separately to avoid nesting
            if algo_name == "xgboost":
                best_clf_model.write().overwrite().save(save_path)
                preprocessing_model.write().overwrite().save(save_path + ".preprocessing")
            else:
                full_stages = preprocessing_model.stages + [best_clf_model]
                full_pipeline_model = PipelineModel(stages=full_stages)
                full_pipeline_model.write().overwrite().save(save_path)

            # Save calibrator
            if calibrator is not None:
                cal_path = os.path.join(save_path, "calibrator.pkl")
                with open(cal_path, "wb") as f:
                    pickle.dump(calibrator, f)
                logger.info(f"  Calibrator saved: {cal_path}")

            # Save optimal threshold
            th_path = os.path.join(save_path, "threshold.txt")
            with open(th_path, "w") as f:
                f.write(str(round(best_threshold, 2)))
            logger.info(f"  Threshold: {best_threshold:.2f}")

            # Save model metadata
            import json as _json
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
                _json.dump(meta, f, indent=2)
            logger.info(f"  Metadata saved")

        # Register version
            _train_metrics = {}
            for _algo in classifiers:
                _mp = os.path.join(_train_base, task_name + "_" + _algo + ".model", "metadata.json")
                if os.path.exists(_mp):
                    with open(_mp) as _f:
                        _m = json.load(_f)
                    _train_metrics[_algo] = {"accuracy": _m.get("accuracy"), "f1_score": _m.get("f1_score"), "train_size": _m.get("train_size")}
            _total = df.count()
            _reg_ver(MODELS_DIR, task_name, _train_ver, _train_metrics, dataset=task_name + ".csv", rows=_total)
            logger.info("  Registered version " + str(_train_ver) + " (" + str(len(_train_metrics)) + " models)")

            elapsed = time.time() - start
            logger.info(f"  {algo_name} F1: {f1*100:.2f}% (Acc: {acc*100:.2f}%) - {elapsed:.1f}s")

        final_df.unpersist()
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        logger.exception(f"Training error: {e}")


if __name__ == "__main__":
    spark = get_spark_builder(app_name="SmartTrainer_FullPipeline",
                              driver_memory="8g").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for f in csv_files:
        train_one_file(spark, f)
    spark.stop()












