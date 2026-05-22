# setup_scheduler.ps1
# D0.2 Daily Snapshot — cai dat Windows Scheduled Task
#
# CHAY 1 LAN DUY NHAT voi quyen Administrator:
#   Right-click PowerShell -> "Run as administrator"
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_scheduler.ps1
#
# Lich: 15:05 moi ngay Thu 2 den Thu 6
# Script: scripts/daily_snapshot.py
# Log: logs/scheduler.log

$ErrorActionPreference = "Stop"

# ── Cau hinh ────────────────────────────────────────────────────────────────
$TaskName    = "AlphaSignal_DailySnapshot"
$RepoRoot    = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe   = "C:\Users\tayd.pyn\AppData\Local\Programs\Python\Python312\python.exe"
$ScriptPath  = Join-Path $RepoRoot "scripts\daily_snapshot.py"
$LogPath     = Join-Path $RepoRoot "logs\scheduler.log"

# ── Kiem tra Python va script ────────────────────────────────────────────────
if (-not (Test-Path $PythonExe)) {
    Write-Error "Khong tim thay Python: $PythonExe"
    Write-Host "Sua bien PythonExe trong file nay cho dung duong dan Python cua ban."
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Khong tim thay script: $ScriptPath"
    exit 1
}

Write-Host "Repo root  : $RepoRoot"
Write-Host "Python     : $PythonExe"
Write-Host "Script     : $ScriptPath"
Write-Host "Log output : $LogPath"
Write-Host ""

# ── Tao thu muc logs neu chua co ─────────────────────────────────────────────
$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "Da tao: $LogDir"
}

# ── Xoa task cu neu ton tai ──────────────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Da xoa task cu: $TaskName"
}

# ── Tao Scheduled Task ───────────────────────────────────────────────────────
$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $RepoRoot

# Thu 2-6, 15:05 (3:05 PM)
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "15:05"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName   $TaskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Principal  $principal `
    -Description "Alpha Signal D0.2: ghi foreign flow + OHLCV snapshot vao SQLite luc 15:05 moi ngay giao dich" | Out-Null

Write-Host ""
Write-Host "=== Scheduled Task da duoc tao thanh cong! ===" -ForegroundColor Green
Write-Host "  Task name : $TaskName"
Write-Host "  Lich chay : 15:05 Mon-Fri"
Write-Host "  Script    : $ScriptPath"
Write-Host ""
Write-Host "De kiem tra: Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "De chay thu: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "De xem log : Get-Content '$LogPath' -Tail 50"
