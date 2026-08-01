.SYNOPSIS
    Show Spark cluster status
#>
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Spark Cluster Status" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8080/json/" -TimeoutSec 3 -UseBasicParsing
    $json = $r.Content | ConvertFrom-Json
    Write-Host "  Master: $($json.url)" -ForegroundColor Green
    Write-Host "  Status: $($json.status) - $($json.aliveworkers) active / $($json.workers) total workers" -ForegroundColor Green
    Write-Host "  Cores : $($json.cores) total, $($json.coresused) used" -ForegroundColor Green
    Write-Host "  Memory: $($json.memory)MB total, $($json.memoryused)MB used" -ForegroundColor Green
    Write-Host ""
    foreach ($w in $json.workers) {
        $state = if ($w.state -eq "ALIVE") { "Green" } else { "Red" }
        Write-Host "  [$($w.state)] $($w.host):$($w.port) | Cores: $($w.cores) ($($w.coresused) used) | Mem: $($w.memory)MB ($($w.memoryused)MB used)" -ForegroundColor $state
    }
} catch {
    Write-Host "  Master not reachable on port 8080. Cluster may be stopped." -ForegroundColor Red
}
Write-Host "============================================" -ForegroundColor Cyan
