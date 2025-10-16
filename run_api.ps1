#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run TravelBot FastAPI backend server with proper Python environment
.DESCRIPTION
    This script ensures the Python environment is set up correctly,
    activates the virtual environment if it exists, and runs the FastAPI backend.
#>

# Configuration
$apiPath = Join-Path $PSScriptRoot "main.py"
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvActivate = Join-Path $venvPath "Scripts/Activate.ps1"
$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"
$port = 8000

# Print banner
Write-Host "`n╔════════════════════════════════════════╗"
Write-Host "║  TravelBot - FastAPI Backend Server     ║"
Write-Host "╚════════════════════════════════════════╝`n"

# Check if API exists
if (-not (Test-Path $apiPath)) {
    Write-Host "❌ FastAPI app not found at: $apiPath" -ForegroundColor Red
    exit 1
}

# Check for virtual environment and activate if it exists
if (Test-Path $venvPath) {
    Write-Host "✓ Found virtual environment at: $venvPath" -ForegroundColor Green
    try {
        Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
        & $venvActivate
    }
    catch {
        Write-Host "❌ Failed to activate virtual environment: $_" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "⚠️ No virtual environment found at: $venvPath" -ForegroundColor Yellow
    Write-Host "   To create one (recommended), run:"
    Write-Host "   python -m venv .venv"
    Write-Host "   .\.venv\Scripts\Activate.ps1"
    Write-Host "   pip install -r requirements.txt`n"
    
    # Check for uvicorn in global environment
    try {
        $uvicornVersion = python -c "import uvicorn; print(f'uvicorn version: {uvicorn.__version__}')" 2>$null
        if (-not $?) {
            Write-Host "❌ uvicorn not found in global environment. Please install:" -ForegroundColor Red
            Write-Host "   pip install -r $requirementsPath`n"
            exit 1
        }
        Write-Host "✓ $uvicornVersion (global)" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to check for uvicorn: $_" -ForegroundColor Red
        exit 1
    }
}

# Run the API
Write-Host "`n✨ Starting FastAPI backend server...`n"

# Configure environment variables if needed
$env:PYTHONUNBUFFERED = "1"

# Run uvicorn
uvicorn main:app --host 0.0.0.0 --port $port --reload