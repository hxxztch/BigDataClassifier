<#
.SYNOPSIS
    Launch Flask app with Spark Standalone Cluster mode enabled.
.DESCRIPTION
    Sets USE_SPARK_CLUSTER=true so the app connects to spark://localhost:7077
    instead of running in local[*] mode.
#>
$ErrorActionPreference = "Stop"

# Check cluster
try {
    $r = Invoke-WebRequest -Uri "http://localhost:9090" -TimeoutSec 3 -UseBasicParsing
    Write-Host "[OK] Spark Master reachable on port 9090" -ForegroundColor Green
} catch {
    Write-Host "[ERR] Spark Master not reachable on port 9090!" -ForegroundColor Red
    Write-Host "      Run start-cluster.bat first." -ForegroundColor Yellow
    exit 1
}

$env:USE_SPARK_CLUSTER = "true"
$env:SPARK_MASTER_URL = "spark://localhost:7077"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Flask App (Cluster Mode)"          -ForegroundColor Cyan
Write-Host "  Spark Master: spark://localhost:7077"        -ForegroundColor Cyan
Write-Host "  Web UI      : http://localhost:5000"         -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$backendDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$backendDir = Join-Path $backendDir "backend"
Set-Location $backendDir
python app.py
