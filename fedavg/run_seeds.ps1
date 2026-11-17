# Run FedAvg for 3 seeds (PowerShell). Activate your Python 3.12 venv first.
foreach ($s in 42, 43, 44) {
    $env:PYTHONHASHSEED = "$s"
    $cfg = "seed=$s run-name=`"fedavg_seed$s`" benchmark-system-metrics=true"
    flwr run . --run-config $cfg --stream
}
python scripts/aggregate.py
