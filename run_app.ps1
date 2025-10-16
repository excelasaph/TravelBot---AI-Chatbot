#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run TravelBot Streamlit App with proper Python environment
.DESCRIPTION
    This script ensures the Python environment is set up correctly,
    activates the virtual environment if it exists, and runs the Streamlit app.
#>

# Configuration
$appPath = Join-Path $PSScriptRoot "app/streamlit_app.py"
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvActivate = Join-Path $venvPath "Scripts/Activate.ps1"
$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"

# Print banner
Write-Host "`n╔════════════════════════════════════════╗"
Write-Host "║  TravelBot - Travel & Geography Demo    ║"
Write-Host "╚════════════════════════════════════════╝`n"

# Check if app exists
if (-not (Test-Path $appPath)) {
    Write-Host "❌ Streamlit app not found at: $appPath" -ForegroundColor Red
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
    
    # Check for streamlit in global environment
    try {
        $streamlitVersion = python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')" 2>$null
        if (-not $?) {
            Write-Host "❌ Streamlit not found in global environment. Please install:" -ForegroundColor Red
            Write-Host "   pip install -r $requirementsPath`n"
            exit 1
        }
        Write-Host "✓ $streamlitVersion (global)" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to check for Streamlit: $_" -ForegroundColor Red
        exit 1
    }
}

# Run the app
Write-Host "`n✨ Starting Streamlit app...`n"

# Configure environment variables if needed
$env:STREAMLIT_THEME = "light"
$env:PYTHONUNBUFFERED = "1"

# Run Streamlit
streamlit run $appPath