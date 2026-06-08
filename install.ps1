$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Installing WE.ED.CUT.IT.CLAW dependencies..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python, then run this installer again."
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "FFmpeg was not found. Install FFmpeg or add it to PATH, then run this installer again."
}

if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "FFprobe was not found. Install FFmpeg or add it to PATH, then run this installer again."
}

@'
import importlib.util
import subprocess
import sys

required = ["numpy", "eyed3", "sklearn"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"Installing missing Python packages: {', '.join(missing)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
else:
    print("All Python dependencies are already installed.")
'@ | python -

New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
python -m py_compile .\beat_sync.py

Write-Host ""
Write-Host "WE.ED.CUT.IT.CLAW successfully set up!"
Write-Host "You can run the script using run.ps1 or run.bat."
