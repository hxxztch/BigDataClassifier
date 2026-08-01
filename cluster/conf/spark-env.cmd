@echo off
REM === Spark Standalone Cluster Environment ===
REM Master settings
set SPARK_MASTER_HOST=127.0.0.1
set SPARK_MASTER_PORT=7077
set SPARK_MASTER_WEBUI_PORT=8080

REM Worker defaults (override per-worker in start script)
set SPARK_WORKER_CORES=4
set SPARK_WORKER_MEMORY=6g

REM Logging
set SPARK_LOG_DIR=%SPARK_HOME%\..\cluster\logs
set SPARK_WORKER_DIR=%SPARK_HOME%\..\cluster

REM Prevent port conflicts with existing Spark
set SPARK_LOCAL_IP=127.0.0.1
set SPARK_PUBLIC_DNS=localhost
