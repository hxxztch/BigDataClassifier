# -*- coding: utf-8 -*-
"""
Spark SQL utilities: SQL-based data quality, skew detection, salt repartition.
Demonstrates Catalyst optimizer awareness via EXPLAIN output.
"""

import textwrap
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, when, lit, rand, floor, concat
from .logger import get_logger

logger = get_logger("spark_sql")


class SparkSQLAnalyzer:
    """SQL-first data analysis with execution plan visibility."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # 鈹€鈹€ Plan visibility 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    def _explain_and_log(self, df: DataFrame, label: str):
        """Output physical plan to log for Catalyst awareness."""
        plan = df._jdf.queryExecution().executedPlan().toString()
        logger.info(f"[PLAN] {label}:\n{plan[:2000]}")

    # 鈹€鈹€ SQL-based column statistics 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    def sql_column_stats(self, table_name: str, columns: list) -> dict:
        """
        Compute per-column statistics using Spark SQL (not DataFrame API).
        Demonstrates: CASE WHEN, aggregate functions, UNION ALL.
        Returns: {col_name: {total_rows, null_count, min, max, mean, std}}
        """
        total = self.spark.sql(f"SELECT COUNT(*) AS cnt FROM {table_name}").collect()[0]["cnt"]
        if total == 0:
            return {"error": "Empty table", "total_rows": 0}

        result = {"total_rows": total, "total_columns": len(columns), "columns": {}}

        for c in columns:
            dtype = self.spark.sql(
                f"DESCRIBE {table_name} {c}"
            ).collect()[0][1]

            # 鈹€鈹€ SQL-based statistics (single pass with CTE) 鈹€鈹€
            sql = textwrap.dedent(f"""
                WITH stats AS (
                    SELECT
                        COUNT(*)                                                          AS total,
                        SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)                     AS null_cnt,
                        MIN({c})                                                          AS min_val,
                        MAX({c})                                                          AS max_val,
                        AVG(CAST({c} AS DOUBLE))                                          AS mean_val,
                        STDDEV(CAST({c} AS DOUBLE))                                       AS std_val
                    FROM {table_name}
                )
                SELECT * FROM stats
            """)

            try:
                row = self.spark.sql(sql).collect()[0]
            except Exception:
                result["columns"][c] = {
                    "name": c, "type": dtype,
                    "null_count": 0, "null_pct": 0.0,
                }
                continue

            null_count = row["null_cnt"] or 0
            result["columns"][c] = {
                "name": c,
                "type": dtype,
                "null_count": null_count,
                "null_pct": round(null_count / total * 100, 1),
                "min": round(float(row["min_val"]), 2) if row["min_val"] is not None else None,
                "max": round(float(row["max_val"]), 2) if row["max_val"] is not None else None,
                "mean": round(float(row["mean_val"]), 2) if row["mean_val"] is not None else None,
                "std": round(float(row["std_val"]), 2) if row["std_val"] is not None else None,
            }

        # Output plan for the last column as a sample
        if columns:
            logger.info(f"[SQL] Column stats computed for {len(columns)} columns via SQL CTE + CASE WHEN")

        return result

    # 鈹€鈹€ Data skew detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    def detect_skew(self, df: DataFrame, partition_col: str = None) -> dict:
        """
        Analyze partition distribution to detect data skew.
        If any partition has >3x the avg size, flag it as skewed.
        Returns: {skewed: bool, partitions: [{id, size_mb}], max_ratio}
        """
        # Check partition-level row distribution
        num_parts = df.rdd.getNumPartitions()
        part_sizes = df.rdd.mapPartitionsWithIndex(
            lambda idx, it: [(idx, sum(1 for _ in it))]
        ).collect()

        sizes = [s[1] for s in part_sizes]
        avg_size = sum(sizes) / len(sizes) if sizes else 1
        max_size = max(sizes) if sizes else 0
        ratio = max_size / avg_size if avg_size > 0 else 1.0
        skewed = ratio > 3.0

        result = {
            "skewed": skewed,
            "num_partitions": num_parts,
            "avg_rows_per_partition": int(avg_size),
            "max_rows_per_partition": max_size,
            "skew_ratio": round(ratio, 2),
            "partitions": [
                {"id": p[0], "rows": p[1]}
                for p in sorted(part_sizes, key=lambda x: -x[1])[:5]
            ],
        }

        if skewed:
            logger.warning(
                f"[SKEW] Detected! max/avg ratio = {ratio:.1f}x. "
                f"Partition {part_sizes[0][0]} has {max_size} rows vs avg {int(avg_size)}. "
                f"Consider salt-based repartition."
            )
        else:
            logger.info(f"[SKEW] Distribution balanced (ratio={ratio:.1f}x, {num_parts} partitions)")

        return result

    # 鈹€鈹€ Salt-based repartition 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    def salt_repartition(self, df: DataFrame, skew_col: str, num_salts: int = 10) -> DataFrame:
        """
        Salt-based repartition for skewed data.
        Adds a salt column (random 0..num_salts-1) to the key,
        repartitions on (salt, key), performs operation,
        then drops the salt.

        Example: if a GROUP BY key has 90% of data in one partition,
        salting spreads it across `num_salts` partitions.
        """
        salt_col_name = f"__salt_{skew_col}"
        salted = df.withColumn(salt_col_name, floor(rand() * num_salts).cast("int"))
        result = salted.repartition(num_salts, salt_col_name, skew_col)
        logger.info(
            f"[SALT] Repartitioned on ({salt_col_name}, {skew_col}) "
            f"with {num_salts} salts 鈫?{result.rdd.getNumPartitions()} partitions"
        )
        return result

    # 鈹€鈹€ SQL-based feature engineering 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    def sql_feature_engineering(self, table_name: str, scene_type: str) -> DataFrame:
        """
        Generate features using Spark SQL CASE WHEN instead of DataFrame withColumn.
        Demonstrates: SQL expressions for feature engineering.
        """
        if scene_type == "shop_shipping":
            sql = textwrap.dedent(f"""
                SELECT *,
                    CASE WHEN Discount_offered > 10 THEN 1 ELSE 0 END AS High_Discount,
                    CASE WHEN Weight_in_gms > 4000 THEN 1 ELSE 0 END AS Is_Heavy
                FROM {table_name}
            """)
            df = self.spark.sql(sql)
            self._explain_and_log(df, "SQL feature engineering (shop_shipping)")
            return df

        elif scene_type == "trans_accident":
            sql = textwrap.dedent(f"""
                SELECT *
                FROM {table_name}
                WHERE Severity IN (2, 3)
            """)
            df = self.spark.sql(sql)
            logger.info("[SQL] trans_accident: filtered to Severity 2,3 via SQL WHERE")
            return df

        elif scene_type == "ind_quality":
            sql = textwrap.dedent(f"""
                SELECT *,
                    CASE WHEN `Pass/Fail` = 1 THEN 1 ELSE 0 END AS Quality
                FROM {table_name}
            """)
            df = self.spark.sql(sql)
            logger.info("[SQL] ind_quality: derived Quality column via SQL CASE WHEN")
            return df

        return self.spark.table(table_name)
