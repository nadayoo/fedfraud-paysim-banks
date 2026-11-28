Set-Location (Join-Path $PSScriptRoot "flwr_app")
$env:PYTHONHASHSEED = "42"
foreach ($p in @("original", "iid", "fraud_skew", "type_skew_a0.5")) {
    $result = "result_${p}_seed0_r20.pkl"
    if (Test-Path $result) { Write-Host "Skipping $p (found $result)"; continue }
    flwr federation simulation-config @none/default local --num-supernodes 5
    $cfg = 'num-server-rounds=20 seed=42 partitioner=\"lookup_' + $p + '_seed0\" run-name=\"' + $p + '_seed0_r20\"'
    Write-Host "Starting $p at $(Get-Date)"
    flwr run . --run-config $cfg --stream
    Write-Host "Finished $p at $(Get-Date)"
}
