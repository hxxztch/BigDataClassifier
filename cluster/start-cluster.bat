@echo off
set SPARK_HOME=E:\bigdata\spark-3.3.0-bin-hadoop3
set JAVA_HOME=C:\Program Files\Java\jdk-21
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

echo ============================================
echo   Spark Standalone Cluster v3.3.0
echo   Master RPC: localhost:7077
echo   Master Web: http://localhost:9090
echo ============================================

echo [1/4] Starting Master (Web UI on :9090)...
start "SparkMaster" /MIN %SPARK_HOME%\bin\spark-class.cmd org.apache.spark.deploy.master.Master --host localhost --port 7077 --webui-port 9090
timeout /t 8 /nobreak >nul

echo [2/4] Starting Worker 1 (Web UI :8081)...
start "SparkWorker1" /MIN %SPARK_HOME%\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker --cores 4 --memory 6g --webui-port 8081 spark://localhost:7077
timeout /t 5 /nobreak >nul

echo [3/4] Starting Worker 2 (Web UI :8082)...
start "SparkWorker2" /MIN %SPARK_HOME%\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker --cores 4 --memory 6g --webui-port 8082 spark://localhost:7077
timeout /t 5 /nobreak >nul

echo [4/4] Starting Worker 3 (Web UI :8083)...
start "SparkWorker3" /MIN %SPARK_HOME%\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker --cores 4 --memory 6g --webui-port 8083 spark://localhost:7077
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   Cluster started!
echo   Master Web UI : http://localhost:9090
echo   Workers       : 3 x (4 cores / 6 GB)
echo   Total Resources: 12 cores / 18 GB
echo ============================================
endlocal
