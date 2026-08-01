<#
.SYNOPSIS
    停止 Spark Standalone 集群中所有进程
#>
$ErrorActionPreference = "SilentlyContinue"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFiles = @("master.pid", "worker1.pid", "worker2.pid", "worker3.pid")
$stopped = 0
foreach ($file in $pidFiles) {
    $path = Join-Path $SCRIPT_DIR $file
    if (Test-Path $path) {
        $pid = Get-Content $path
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-Host "  [OK] Stopped process $file (PID: $pid)" -ForegroundColor Green
            $stopped++
        } catch {
            Write-Host "  [--] $file (PID: $pid) already stopped" -ForegroundColor DarkGray
        }
        Remove-Item $path -Force
    }
}
Get-Process | Where-Object { $_.ProcessName -like "*java*" } | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force; Write-Host "  [OK] Residual Java PID $($_.Id)" -ForegroundColor Yellow } catch {}
}
Write-Host ""
Write-Host "Cluster stopped. ($stopped processes terminated)" -ForegroundColor Green
