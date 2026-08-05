# -*- coding: utf-8 -*-
"""
MLflow experiment tracking: replace version_registry.json with proper ML tracking.
"""

import os, mlflow, glob
from .logger import get_logger

logger = get_logger("mlflow")

_TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")


def init_mlflow():
    """Initialize MLflow tracking. Called once at app startup."""
    global _TRACKING_DIR
    global _TRACKING_DIR
    # Use default file-based tracking (works with MLflow 3.x)
    mlflow.set_experiment("BigDataClassifier")



def list_runs():
    import yaml
    global _TRACKING_DIR
    _dir = _TRACKING_DIR or os.path.join(os.path.dirname(__file__), "..", "mlruns")
    runs = []
    for exp_dir in glob.glob(os.path.join(_dir, "*")):
        if not os.path.isdir(exp_dir) or os.path.basename(exp_dir).startswith("."):
            continue
        for run_dir in glob.glob(os.path.join(exp_dir, "*")):
            meta_file = os.path.join(run_dir, "meta.yaml")
            if not os.path.isfile(meta_file):
                continue
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            metrics = {}
            metrics_dir = os.path.join(run_dir, "metrics")
            if os.path.isdir(metrics_dir):
                for mf in glob.glob(os.path.join(metrics_dir, "*")):
                    with open(mf) as m:
                        val = m.read().strip().split()
                        metrics[os.path.basename(mf)] = float(val[1]) if len(val) > 1 else (float(val[0]) if val else 0)
            tags = {}
            tags_dir = os.path.join(run_dir, "tags")
            if os.path.isdir(tags_dir):
                for tf in glob.glob(os.path.join(tags_dir, "*")):
                    with open(tf) as t:
                        tags[os.path.basename(tf)] = t.read().strip()
            runs.append({
                "run_id": os.path.basename(run_dir),
                "name": meta.get("run_name", ""),
                "start_time": str(meta.get("start_time", ""))[:19],
                "metrics": metrics,
                "tags": tags,
            })
    return sorted(runs, key=lambda r: r["start_time"], reverse=True)
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
