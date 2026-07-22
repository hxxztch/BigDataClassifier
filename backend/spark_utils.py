import os
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel, NaiveBayesModel
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, lit, when, exp

from database import update_task_progress
from utils.config import (
    MODELS_DIR, DATASET_META, SCENE_LABEL_MAP,
    get_spark_builder, AUTO_CANDIDATES,
)
from utils.preprocessing import (
    clean_column_names, custom_preprocessing,
    validate_and_filter_columns, get_feature_cols,
    prepare_features_fallback, calculate_confusion_matrix,
    extract_feature_importance,
)
import pickle
from utils.logger import get_logger

logger = get_logger(__name__)


class SparkClassifier:

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.spark = get_spark_builder(
            app_name="SparkPredictor_FullLabels"
        ).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        logger.info("SparkClassifier initialized")

    def _load_model(self, model_type, scene_type):
        model_name = f"{scene_type}_{model_type}.model"
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_name}"
        # XGBoost is saved as raw SparkXGBClassifierModel + separate preprocessing
        if model_type == "xgboost":
            try:
                from xgboost.spark import SparkXGBClassifierModel
                xgb_model = SparkXGBClassifierModel.load(model_path)
                prep_path = model_path + ".preprocessing"
                if os.path.exists(prep_path):
                    prep_model = PipelineModel.load(prep_path)
                    combined_stages = prep_model.stages + [xgb_model]
                    return PipelineModel(stages=combined_stages), None
                else:
                    model = PipelineModel.load(model_path)
                    return model, None
            except Exception as e:
                return None, f"XGBoost model load failed: {e}"
        try:
            model = PipelineModel.load(model_path)
            return model, None
        except Exception:
            pass
        loaders = {
            "random_forest": RandomForestClassificationModel,
            "gbdt": GBTClassificationModel,
            "xgboost": SparkXGBClassifierModel,
            "naive_bayes": NaiveBayesModel,
        }
        loader = loaders.get(model_type)
        if not loader:
            return None, f"Unsupported model: {model_type}"
        try:
            model = loader.load(model_path)
            return model, None
        except Exception as e:
            return None, f"Model load failed: {e}"

    def _run_model(self, df, model_type, scene_type, task_id, has_label, target_col, feature_cols):
        model, err = self._load_model(model_type, scene_type)
        if err:
            return 0.0, 0.0, None, [], err
        try:
            update_task_progress(
                task_id, current_model=model_type,
                model_progress_status=f"Evaluating {model_type}..."
            )
            logger.info(f"Running: {model_type} ...")
            is_pipeline = isinstance(model, PipelineModel)
            if not is_pipeline and "features" not in df.columns:
                df_prep = prepare_features_fallback(df, feature_cols)
                if df_prep is None:
                    return 0.0, 0.0, None, [], "Feature prep failed"
                df = df_prep
            predictions = model.transform(df)
            if "probability" in predictions.columns:
                # Take the max probability as confidence (the model's certainty in its own prediction)
                predictions = predictions.withColumn("confidence_0", vector_to_array(col("probability"))[0].cast("double"))
                predictions = predictions.withColumn("confidence_1", vector_to_array(col("probability"))[1].cast("double"))
                predictions = predictions.withColumn("confidence", when(col("prediction") == 0.0, col("confidence_0")).otherwise(col("confidence_1")))
            else:
                predictions = predictions.withColumn("confidence", lit(0.0))

            # --- Platt scaling calibration (overrides raw confidence if calibrator exists) ---
            _cal_path = os.path.join(self.models_dir, f"{scene_type}_{model_type}.model", "calibrator.pkl")
            if os.path.exists(_cal_path) and "rawPrediction" in predictions.columns:
                try:
                    with open(_cal_path, "rb") as _cf:
                        _calibrator = pickle.load(_cf)
                    _A = float(_calibrator.coef_[0][0])
                    _B = float(_calibrator.intercept_[0])
                    predictions = predictions.withColumn(
                        "calibrated_conf",
                        1.0 / (1.0 + exp(-(lit(_A) * vector_to_array(col("rawPrediction"))[1].cast("double") + lit(_B))))
                    )
                    predictions = predictions.withColumn("confidence", col("calibrated_conf"))
                    logger.info(f"  Platt calibration applied: A={_A:.4f}, B={_B:.4f}")
                except Exception as _ce:
                    logger.warning(f"Calibration apply failed: {_ce}")

            # Apply optimal threshold if available
            _th_path = os.path.join(self.models_dir, f"{scene_type}_{model_type}.model", "threshold.txt")
            if os.path.exists(_th_path) and "probability" in predictions.columns:
                try:
                    with open(_th_path) as _f:
                        _opt_th = float(_f.read().strip())
                    predictions = predictions.withColumn("prediction_orig", col("prediction"))
                    predictions = predictions.withColumn(
                        "prediction",
                        when(vector_to_array(col("probability"))[1] >= _opt_th, 1.0).otherwise(0.0)
                    )
                    predictions = predictions.withColumn("prediction", col("prediction").cast("double"))
                    logger.info(f"  Threshold applied: {_opt_th:.2f}")
                except Exception as _te:
                    logger.warning(f"Threshold apply: {_te}")
            acc, f1 = 0.0, 0.0
            if has_label:
                predictions = predictions.withColumn("prediction", col("prediction").cast("double"))
                if scene_type == "trans_accident":
                    predictions = predictions.withColumn(
                        "label",
                        when(col(target_col) == 2, 0.0)
                        .when(col(target_col) == 3, 1.0)
                        .otherwise(None)
                    ).filter(col("label").isNotNull())
                else:
                    predictions = predictions.withColumn("label", col(target_col).cast("double"))
                f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(predictions)
                acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(predictions)
                logger.info(f"  {model_type} -> F1: {f1*100:.2f}%, Acc: {acc*100:.2f}%")
            feature_imp = extract_feature_importance(model, feature_cols)
            return f1, acc, predictions, feature_imp, None
        except Exception as e:
            logger.error(f"{model_type} failed: {e}")
            return 0.0, 0.0, None, [], str(e)

    def predict(self, file_path, model_type, scene_type, task_id):
        logger.info(f"Predict start {scene_type} | mode: {model_type}")
        try:
            update_task_progress(
                task_id, current_model="None",
                model_progress_status="Reading data / preprocessing..."
            )
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path) if not file_path.endswith(".parquet") else self.spark.read.parquet(file_path)
            df = clean_column_names(df)
            df = validate_and_filter_columns(df, scene_type)
            df = df.fillna(0).fillna("Unknown")
            df = custom_preprocessing(df, scene_type)
            target_col = DATASET_META.get(scene_type, "")
            if target_col:
                target_col = target_col.replace(".", "_").replace(" ", "_")
            has_label = target_col in df.columns
            feature_cols = get_feature_cols(df, target_col)
            best_acc = -1.0
            best_predictions = None
            final_model_name = model_type
            final_feature_imp = []
            if model_type == "auto":
                results = []
                for m in AUTO_CANDIDATES:
                    f1_val, acc_val, preds, imp, err = self._run_model(
                        df, m, scene_type, task_id, has_label, target_col, feature_cols
                    )
                    if preds is not None:
                        results.append((f1_val, acc_val, m, preds, imp))
                    elif err:
                        logger.error(f"Model {m} failed: {err}")
                if not results:
                    return {"error": "All models failed"}
                if has_label:
                    results.sort(key=lambda x: x[0], reverse=True)
                    best_f1, best_acc, final_model_name, best_predictions, final_feature_imp = results[0]
                    logger.info(f"Auto best: {final_model_name} (F1: {best_f1*100:.2f}%)")
                else:
                    default = next((r for r in results if r[2] == "random_forest"), results[0])
                    _, best_acc, final_model_name, best_predictions, final_feature_imp = default
            else:
                best_f1, best_acc, best_predictions, final_feature_imp, err = self._run_model(
                    df, model_type, scene_type, task_id, has_label, target_col, feature_cols
                )
                if best_predictions is None:
                    return {"error": err}
                final_model_name = model_type
            confusion_matrix = []
            matrix_categories = []
            if has_label:
                try:
                    matrix_categories, confusion_matrix = calculate_confusion_matrix(
                        best_predictions, target_col, scene_type
                    )
                except Exception as e:
                    logger.warning(f"Confusion matrix failed: {e}")
            update_task_progress(
                task_id, current_model=final_model_name,
                model_progress_status="Generating report..."
            )
            display_cols = ["prediction", "confidence"]
            if has_label:
                display_cols.append(target_col)
            results_rows = best_predictions.select(*display_cols).limit(100).collect()
            dist_rows = best_predictions.select("prediction").groupBy("prediction").count().collect()
            label_map = SCENE_LABEL_MAP.get(scene_type, {})
            distribution = []
            for row in dist_rows:
                pred_val = int(row["prediction"])
                name = label_map.get(pred_val, str(pred_val))
                distribution.append({"name": name, "value": int(row["count"])})
            chart_data = []
            for i, row in enumerate(results_rows):
                pred_val = int(row["prediction"])
                pred_str = label_map.get(pred_val, str(pred_val))
                act_str = "-"
                if has_label and row[1] is not None:
                    act_val = int(row[1])
                    act_str = label_map.get(act_val, str(act_val))
                    confidence_val = float(row["confidence"]) if "confidence" in row else 0.0
                    chart_data.append({"id": i + 1, "actual": act_str, "predicted": pred_str, "confidence": round(confidence_val, 4)})
            update_task_progress(
                task_id, current_model=final_model_name,
                model_progress_status="Complete"
            )
            logger.info(f"Predict done: model={final_model_name}, acc={best_acc:.4f}")
            return {
                "accuracy": best_acc,
                "chart_data": chart_data,
                "distribution": distribution,
                "final_model": final_model_name,
                "confusion_matrix": confusion_matrix,
                "all_labels_indices": matrix_categories,
                "feature_importance": final_feature_imp,
                "classification_report": [],
            }
        except Exception as e:
            logger.exception(f"Predict error: {e}")
            return {"error": str(e)}

    def stop(self):
        self.spark.stop()
        logger.info("SparkSession stopped")

