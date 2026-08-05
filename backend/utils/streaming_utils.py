# -*- coding: utf-8 -*-
"""
Spark Structured Streaming classification: simulated real-time data pipeline.
Demonstrates: rate source, micro-batch processing, MLlib inference, memory sink.
"""

import time
from threading import Thread, Lock
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, struct, to_json, rand, when, floor
from .logger import get_logger

logger = get_logger("streaming")


class StreamingClassifier:
    """Simulates real-time classification using Spark Structured Streaming.

    Flow:
      rate source (rows/sec) -> random features -> MLlib predict -> memory sink
    """

    def __init__(self, spark, scene_type: str, feature_cols: list):
        self.spark = spark
        self.scene_type = scene_type
        self.feature_cols = feature_cols
        self._query = None
        self._running = False
        self._lock = Lock()
        self._stats = {
            "total_processed": 0,
            "class_0_count": 0,
            "class_1_count": 0,
            "batches": 0,
            "start_time": None,
            "latest_prediction": None,
            "history": [],  # last 50 predictions
        }

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def start(self, pipeline_model, rows_per_sec: int = 10, trigger_sec: int = 2):
        """
        Start a streaming query that:
        1. Generates rows at `rows_per_sec` rate
        2. Creates random feature vectors matching the trained model's schema
        3. Classifies via MLlib PipelineModel
        4. Sinks results to memory table 'streaming_results'
        """
        if self.running:
            return {"error": "Stream already running"}

        with self._lock:
            self._stats["start_time"] = time.strftime("%H:%M:%S")
            self._stats["batches"] = 0
            self._stats["total_processed"] = 0
            self._stats["class_0_count"] = 0
            self._stats["class_1_count"] = 0
            self._stats["history"] = []

        # Build a rate source + random feature DF
        raw = self.spark.readStream.format("rate") \
            .option("rowsPerSecond", rows_per_sec) \
            .load()

        # Simulate features: map timestamp + random values to approximate schema
        # For demonstration, generate random numeric features
        from pyspark.sql.types import StructType, StructField, DoubleType
        from pyspark.sql.functions import rand as _rand, monotonically_increasing_id

        # Build a simple feature DF with random values
        _feat_count = len(self.feature_cols)
        _exprs = [monotonically_increasing_id().alias("_id")]
        for i in range(_feat_count):
            _exprs.append((_rand() * 100).cast("double").alias(f"f{i}"))
        features_df = raw.select(*_exprs)

        # Assemble into Vector for MLlib
        from pyspark.ml.feature import VectorAssembler
        _feature_names = [f"f{i}" for i in range(_feat_count)]
        assembler = VectorAssembler(inputCols=_feature_names, outputCol="features")
        assembled = assembler.transform(features_df)

        # ── Create a wrapper pipeline that just does classification ──
        # We can't use the full PipelineModel because it expects real column names.
        # Instead, create a simple classification-only pipeline.
        from pyspark.ml import PipelineModel

        try:
            predictions = pipeline_model.transform(assembled)
        except Exception:
            # Fallback: extract just the classifier stage and use it directly
            _clf_stage = pipeline_model.stages[-1] if hasattr(pipeline_model, 'stages') else None
            if _clf_stage:
                predictions = _clf_stage.transform(assembled)
            else:
                return {"error": "Cannot extract classifier from pipeline"}

        # Extract prediction + confidence
        from pyspark.sql.functions import struct, to_json, col, when
        output = predictions.select(
            "prediction",
            col("probability").cast("string").alias("probability_str"),
            when(col("prediction") == 0.0, 1).otherwise(0).alias("class_0"),
            when(col("prediction") == 1.0, 1).otherwise(0).alias("class_1"),
        )

        # Custom foreachBatch to accumulate stats
        def _accumulate(df, epoch_id):
            rows = df.collect()
            batch_size = len(rows)
            with self._lock:
                self._stats["batches"] += 1
                self._stats["total_processed"] += batch_size
                self._stats["class_0_count"] += sum(r["class_0"] for r in rows)
                self._stats["class_1_count"] += sum(r["class_1"] for r in rows)
                if rows:
                    r = rows[-1]
                    self._stats["latest_prediction"] = {
                        "prediction": int(r["prediction"]),
                        "batch": self._stats["batches"],
                        "time": time.strftime("%H:%M:%S"),
                    }
                    # Keep last 50 history entries
                    self._stats["history"].append(self._stats["latest_prediction"])
                    if len(self._stats["history"]) > 50:
                        self._stats["history"] = self._stats["history"][-50:]
            logger.info(
                f"[Stream] Batch {epoch_id}: {batch_size} rows, "
                f"class_0={sum(r['class_0'] for r in rows)}, "
                f"class_1={sum(r['class_1'] for r in rows)}"
            )

        query = output.writeStream \
            .foreachBatch(_accumulate) \
            .trigger(processingTime=f"{trigger_sec} seconds") \
            .start()

        with self._lock:
            self._query = query
            self._running = True

        logger.info(f"[Stream] Started: {rows_per_sec} rows/s, trigger={trigger_sec}s")
        return {"status": "started", "rows_per_sec": rows_per_sec}

    def stop(self):
        """Stop the streaming query."""
        with self._lock:
            if self._query:
                self._query.stop()
                self._query = None
            self._running = False
        logger.info("[Stream] Stopped")
        return {"status": "stopped", "stats": self.stats}

    def status(self) -> dict:
        """Get current streaming status + statistics."""
        s = self.stats
        return {
            "running": self.running,
            "total_processed": s["total_processed"],
            "class_0_pct": round(
                s["class_0_count"] / s["total_processed"] * 100, 1
            ) if s["total_processed"] > 0 else 0,
            "class_1_pct": round(
                s["class_1_count"] / s["total_processed"] * 100, 1
            ) if s["total_processed"] > 0 else 0,
            "batches": s["batches"],
            "start_time": s["start_time"],
            "latest_prediction": s["latest_prediction"],
            "history": s["history"],
        }
