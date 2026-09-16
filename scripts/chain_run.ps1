# Wait for a training run to finish, rebuild the dataset on the latest hand fixes, train the
# next one, then score its checkpoints on the real-world test set.
#
# A running job holds the label cache it took at startup, so corrections made since are
# invisible to it; they only reach a model through a rebuild. Doing the three steps unattended
# keeps the GPU busy without having to watch for the first run to end.
#
#   powershell -File scripts\chain_run.ps1 -WaitFor yolo11s_v5 -Next yolo11s_v6

param(
    [Parameter(Mandatory = $true)][string]$WaitFor,
    [Parameter(Mandatory = $true)][string]$Next,
    [int]$Epochs = 200,
    [int]$SavePeriod = 20
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\서동준\git"
$python = "E:\anaconda\python.exe"
Set-Location $repo

Write-Output "waiting for $WaitFor to finish ..."
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$WaitFor*" }
    if (-not $running) { break }
    Start-Sleep -Seconds 30
}
Write-Output "$WaitFor finished"

Write-Output "rebuilding dataset_v2 on the current label_fixes ..."
& $python scripts\build_dataset_ouo.py 2>&1 | Select-Object -Last 14

Write-Output "starting $Next ..."
& $python scripts\train.py `
    --data "$repo\dataset_v2\data.yaml" `
    --model yolo11s.pt `
    --epochs $Epochs --patience $Epochs --save-period $SavePeriod `
    --name $Next 2>&1 |
    Tee-Object -FilePath "$repo\train_$Next.log" | Select-Object -Last 25

Write-Output "scoring $Next checkpoints on the real-world test set ..."
& $python scripts\eval_checkpoints.py --run $Next 2>&1 | Select-Object -Last 25
