# Wait for the training in progress to finish, rebuild the dataset on the latest hand fixes,
# then start the next run with the same schedule.
#
# The running job holds a label cache taken at its start, so edits made since are invisible to
# it; they only reach a model through a rebuild. Doing both unattended keeps the GPU busy
# without having to watch for the first run to end.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\서동준\git"
$python = "E:\anaconda\python.exe"
$waitFor = "yolo11s_v4"
$nextName = "yolo11s_v5"

Set-Location $repo

Write-Output "waiting for $waitFor to finish ..."
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$waitFor*" }
    if (-not $running) { break }
    Start-Sleep -Seconds 30
}
Write-Output "$waitFor finished"

Write-Output "rebuilding dataset_v2 on the current label_fixes ..."
& $python scripts\build_dataset_ouo.py 2>&1 | Select-Object -Last 12

Write-Output "starting $nextName ..."
& $python scripts\train.py `
    --data "$repo\dataset_v2\data.yaml" `
    --model yolo11s.pt `
    --epochs 200 --patience 200 --save-period 20 `
    --name $nextName 2>&1 |
    Tee-Object -FilePath "$repo\train_v5.log" | Select-Object -Last 30
