# -*- coding: utf-8 -*-
"""
MLflow experiment tracking: replace version_registry.json with proper ML tracking.
"""

import os, mlflow
from .logger import get_logger

logger = get_logger("mlflow")

_TRACKING_DIR = None


def init_mlflow(project_root: str):
    """Initialize MLflow tracking. Called once at app startup."""
    global _TRACKING_DIR
    _TRACKING_DIR = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{_TRACKING_DIR.replace(chr(92), '/')}")
    mlflow.set_experiment("BigDataClassifier")
    logger.info(f"MLflow initialized: {_TRACKING_DIR}")


def log_mllib_model(algo_name: str, scene_id: str, params: dict, metrics: dict):
    """Log a Spark MLlib model run to MLflow."""
    with mlflow.start_run(run_name=f"{scene_id}_{algo_name}_MLlib", nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("backend", "Spark MLlib")
        mlflow.set_tag("scene", scene_id)
        mlflow.set_tag("algorithm", algo_name)
        logger.info(
            f"  [MLflow] {algo_name} | F1={metrics.get('f1', 0):.4f} "
            f"Acc={metrics.get('accuracy', 0):.4f}"
        )


def log_sklearn_model(algo_name: str, scene_id: str, params: dict, metrics: dict):
    """Log a sklearn model run to MLflow."""
    with mlflow.start_run(run_name=f"{scene_id}_{algo_name}_sklearn", nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("backend", "sklearn")
        mlflow.set_tag("scene", scene_id)
        mlflow.set_tag("algorithm", algo_name)
        logger.info(
            f"  [MLflow] {algo_name}(sklearn) | F1={metrics.get('f1', 0):.4f} "
            f"Acc={metrics.get('accuracy', 0):.4f}"
        )


def get_ui_url() -> str:
    return "http://127.0.0.1:5001"
