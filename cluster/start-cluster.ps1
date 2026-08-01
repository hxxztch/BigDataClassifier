<#
.SYNOPSIS
    启动 Spark 3-Node 本地独立集群 (1 Master + 3 Workers)
.DESCRIPTION
    在同一台机器上启动完整�?Spark standalone 集群�?    每个 Worker 是独�?JVM 进程，模拟真实分布式环境�?#>

$ErrorActionPreference = "Stop"

# ── 路径配置 ────────────────────────────────────
$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$CLUSTER_DIR = $SCRIPT_DIR
$SPARK_HOME  = "E:\bigdata\spark-3.3.0-bin-hadoop3"
$JAVA_HOME   = "C:\Program Files\Java\jdk-21"
$SPARK_CONF  = Join-Path $SCRIPT_DIR "conf\spark-env.cmd"

# 加载环境配置
if (Test-Path $SPARK_CONF) {
    cmd /c "call `"$SPARK_CONF`" && set" | ForEach-Object {
        if ($_ -match "^(SPARK_[A-Z_]+)=(.*)") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

$env:SPARK_HOME = $SPARK_HOME
$env:JAVA_HOME  = $JAVA_HOME
$env:SPARK_MASTER_HOST = "localhost"

$MASTER_PORT       = $env:SPARK_MASTER_PORT       ?? "7077"
$MASTER_WEBUI_PORT = $env:SPARK_MASTER_WEBUI_PORT ?? "8180"
$WORKER_CORES      = $env:SPARK_WORKER_CORES      ?? "4"
$WORKER_MEMORY     = $env:SPARK_WORKER_MEMORY     ?? "6g"

# Executable paths
$SPARK_CLASS = Join-Path $SPARK_HOME "bin\spark-class.cmd"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Spark Standalone Cluster v3.3.0"        -ForegroundColor Cyan
Write-Host "  Master: localhost:$MASTER_PORT"         -ForegroundColor Cyan
Write-Host "  Web UI: http://localhost:$MASTER_WEBUI_PORT" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ── 1. 启动 Master ──────────────────────────────
Write-Host "[1/4] Starting Master..." -ForegroundColor Yellow
$MasterWorkDir = Join-Path $CLUSTER_DIR "master"
New-Item -ItemType Directory -Force -Path $MasterWorkDir | Out-Null

$masterArgs = @(
    "org.apache.spark.deploy.master.Master",
    "--host", "localhost",
    "--port", $MASTER_PORT,
    "--webui-port", $MASTER_WEBUI_PORT
) -join " "

Start-Process -FilePath $SPARK_CLASS -ArgumentList $masterArgs -NoNewWindow -PassThru |
    ForEach-Object { $_.Id } |
    Set-Content -Path (Join-Path $CLUSTER_DIR "master.pid")

Write-Host "  Master PID: $(Get-Content (Join-Path $CLUSTER_DIR 'master.pid'))" -ForegroundColor Green
Start-Sleep -Seconds 4

# ── 2. 启动 3 �?Worker ─────────────────────────
$workerConfigs = @(
    @{ Name = "worker1"; Cores = 4; Memory = "6g"; WebUI = 8081 }
    @{ Name = "worker2"; Cores = 4; Memory = "6g"; WebUI = 8082 }
    @{ Name = "worker3"; Cores = 4; Memory = "6g"; WebUI = 8083 }
)

for ($i = 0; $i -lt $workerConfigs.Count; $i++) {
    $wc = $workerConfigs[$i]
    $step = $i + 2
    Write-Host "[$step/4] Starting $($wc.Name) (cores=$($wc.Cores), mem=$($wc.Memory), UI=:$($wc.WebUI))..." -ForegroundColor Yellow

    $workerDir = Join-Path $CLUSTER_DIR $wc.Name
    New-Item -ItemType Directory -Force -Path $workerDir | Out-Null

    $workerArgs = @(
        "org.apache.spark.deploy.worker.Worker",
        "--cores", $wc.Cores,
        "--memory", $wc.Memory,
        "--webui-port", $wc.WebUI,
        "--work-dir", $workerDir,
        "spark://localhost:$MASTER_PORT"
    ) -join " "

    Start-Process -FilePath $SPARK_CLASS -ArgumentList $workerArgs -NoNewWindow -PassThru |
        ForEach-Object { $_.Id } |
        Set-Content -Path (Join-Path $CLUSTER_DIR "$($wc.Name).pid")

    Write-Host "  $($wc.Name) PID: $(Get-Content (Join-Path $CLUSTER_DIR "$($wc.Name).pid"))" -ForegroundColor Green
    Start-Sleep -Seconds 3
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Cluster started successfully!"            -ForegroundColor Green
Write-Host "  Master UI: http://localhost:8180"          -ForegroundColor Green
Write-Host "  Workers: 3 x ($WORKER_CORES cores / $WORKER_MEMORY)" -ForegroundColor Green
Write-Host "  Total resources: 12 cores / 18 GB"         -ForegroundColor Green
Write-Host "  To stop: .\stop-cluster.ps1"               -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
