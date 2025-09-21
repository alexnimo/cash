#!/usr/bin/env pwsh
<#
.SYNOPSIS
Setup script for the Video Analyzer project on Windows

.DESCRIPTION
This script sets up the Python environment, installs dependencies, 
and validates the setup for the YouTube Video Analysis Platform.

.EXAMPLE
.\setup.ps1
#>

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Video Analyzer..." -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green

# Function to check Python version
function Test-PythonVersion {
    Write-Host "`n📋 Checking Python version..." -ForegroundColor Yellow
    
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green
            
            # Check if version is 3.9 or higher
            if ($pythonVersion -match "Python (\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                
                if ($major -ge 3 -and $minor -ge 9) {
                    Write-Host "✓ Python version meets requirements (≥3.9)" -ForegroundColor Green
                    return $true
                } else {
                    Write-Host "❌ Python version $major.$minor is too old. Please install Python 3.9 or higher." -ForegroundColor Red
                    return $false
                }
            }
        }
    }
    catch {
        Write-Host "❌ Python is not installed or not in PATH. Please install Python 3.9 or higher." -ForegroundColor Red
        return $false
    }
}

# Function to verify project structure
function Test-ProjectStructure {
    Write-Host "`n📁 Verifying project structure..." -ForegroundColor Yellow
    
    # Check if running from project root
    if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "app" -PathType Container)) {
        Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
        Write-Host "   Current directory: $(Get-Location)" -ForegroundColor Red
        return $false
    }
    
    Write-Host "✓ Running from correct directory" -ForegroundColor Green
    
    # Check for required files
    $requiredFiles = @("pyproject.toml", "config.yaml", ".env")
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✓ Found $file" -ForegroundColor Green
        } else {
            Write-Host "❌ Missing $file" -ForegroundColor Red
        }
    }
    
    return $true
}

# Function to setup virtual environment with uv
function Initialize-Environment {
    Write-Host "`n🔧 Setting up Python environment with uv..." -ForegroundColor Yellow
    
    # Check if uv is installed
    try {
        uv --version | Out-Null
        Write-Host "✓ UV package manager found" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ UV package manager not found. Installing..." -ForegroundColor Yellow
        try {
            # Install uv using pip
            python -m pip install uv
            Write-Host "✓ UV installed successfully" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Failed to install UV. Please install manually: pip install uv" -ForegroundColor Red
            return $false
        }
    }
    
    # Install dependencies using uv
    try {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        uv sync
        Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        return $false
    }
}

# Function to test embedding service
function Test-EmbeddingService {
    Write-Host "`n🧠 Testing embedding service..." -ForegroundColor Yellow
    
    try {
        $testResult = uv run python test_embeddings.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Embedding service test passed" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Embedding service test failed" -ForegroundColor Red
            Write-Host "Output: $testResult" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Failed to run embedding service test" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to verify critical imports
function Test-CriticalImports {
    Write-Host "`n🔍 Verifying critical imports..." -ForegroundColor Yellow
    
    $testScript = @'
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing critical imports...")

try:
    # Core imports
    import app
    print("✓ app module")
    
    from app.core.unified_config import ConfigManager
    print("✓ ConfigManager")
    
    from app.services.embedding_service import EmbeddingService
    print("✓ EmbeddingService")
    
    from app.services.video_processor import VideoProcessor
    print("✓ VideoProcessor")
    
    from app.agents.agents import init_agents
    print("✓ Agent system")
    
    print("✓ All critical imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
'@
    
    try {
        $testScript | uv run python -
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ All critical imports successful" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Import test failed" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Failed to run import test" -ForegroundColor Red
        return $false
    }
}

# Function to check external dependencies
function Test-ExternalDependencies {
    Write-Host "`n🔧 Checking external dependencies..." -ForegroundColor Yellow
    
    # Check ffmpeg
    try {
        ffmpeg -version 2>$null | Out-Null
        Write-Host "✓ FFmpeg is installed" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  FFmpeg not found. This may cause issues with video processing." -ForegroundColor Yellow
        Write-Host "   Install from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    }
    
    return $true
}

# Function to show setup completion
function Show-SetupComplete {
    Write-Host "`n✨ Setup completed!" -ForegroundColor Green
    Write-Host "===================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the server, run:" -ForegroundColor Yellow
    Write-Host "  uv run python -m app.main" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run tests:" -ForegroundColor Yellow  
    Write-Host "  uv run python test_embeddings.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Configuration files:" -ForegroundColor Yellow
    Write-Host "  - config.yaml: Application settings" -ForegroundColor Cyan
    Write-Host "  - .env: Environment variables" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🎯 The embedding service is ready and working!" -ForegroundColor Green
}

# Main execution
function Main {
    try {
        # Run all setup steps
        if (-not (Test-PythonVersion)) { return 1 }
        if (-not (Test-ProjectStructure)) { return 1 }
        if (-not (Initialize-Environment)) { return 1 }
        if (-not (Test-EmbeddingService)) { return 1 }
        if (-not (Test-CriticalImports)) { return 1 }
        if (-not (Test-ExternalDependencies)) { return 1 }
        
        # Show completion message
        Show-SetupComplete
        return 0
        
    }
    catch {
        Write-Host "`n❌ Setup failed with error:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        return 1
    }
}

# Run main function and exit with its return code
exit (Main)
