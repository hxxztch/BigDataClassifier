import json, os
from datetime import datetime
from pyspark.sql.functions import col, count, when, isnan, isnull, lit, min as min_, max as max_, mean, std as std_
from .config import MODELS_DIR, DATASET_META, SCENE_REQUIRED_COLS
from .logger import get_logger
logger = get_logger(__name__)

def analyze_dataframe(df, scene_type):
    total = df.count()
    if total == 0:
        return {"error": "Empty dataset", "total_rows": 0}
    columns = df.columns
    col_stats = {}
    for c in columns:
        col_type = [t for n, t in df.dtypes if n == c]
        col_type = col_type[0] if col_type else "unknown"
        if col_type in ("int", "bigint", "double", "float", "long", "decimal"):
            null_count = df.filter(col(c).isNull() | isnan(col(c))).count()
        else:
            null_count = df.filter(col(c).isNull()).count()
        stats = {"name": c, "type": col_type, "null_count": null_count, "null_pct": round(null_count / total * 100, 1)}
        if col_type in ("int", "bigint", "double", "float", "long", "decimal"):
            try:
                row = df.agg(min_(c).alias("min"), max_(c).alias("max"), mean(c).alias("mean"), std_(c).alias("std")).collect()[0]
                stats["min"] = round(float(row["min"]), 2) if row["min"] is not None else None
                stats["max"] = round(float(row["max"]), 2) if row["max"] is not None else None
                stats["mean"] = round(float(row["mean"]), 2) if row["mean"] is not None else None
                stats["std"] = round(float(row["std"]), 2) if row["std"] is not None else None
            except Exception:
                pass
        col_stats[c] = stats
    report = {"total_rows": total, "total_columns": len(columns), "columns": col_stats, "missing_threshold": 50, "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    warnings = []
    for c, s in col_stats.items():
        if s["null_pct"] > report["missing_threshold"]:
            warnings.append('Column "%s" has %.1f%% missing values (threshold: %.0f%%)' % (c, s["null_pct"], report["missing_threshold"]))
    report["warnings"] = warnings
    return report

def compare_with_schema(df, scene_type):
    required = SCENE_REQUIRED_COLS.get(scene_type, [])
    if not required:
        return {"match": True, "message": "No schema defined for this scene"}
    actual_cols = set(df.columns)
    required_set = set(required)
    missing = required_set - actual_cols
    extra = actual_cols - required_set
    result = {"match": len(missing) == 0, "required_cols": len(required), "actual_cols": len(actual_cols), "missing_cols": sorted(missing)[:10], "extra_cols": sorted(extra)[:10], "missing_count": len(missing), "extra_count": len(extra)}
    if missing:
        result["message"] = "Missing %d required feature columns" % len(missing)
    elif extra:
        result["message"] = "Found %d extra columns (will be ignored)" % len(extra)
    else:
        result["message"] = "All columns match perfectly"
    return result
