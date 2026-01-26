# Download mmproj file for Qwen3-VL vision support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Qwen3-VL mmproj File Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$modelUrl = "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/qwen3-vl-2b-instruct-mmproj-f16.gguf"
$outputPath = Join-Path $PSScriptRoot "qwen3-vl-2b-instruct-mmproj-f16.gguf"
$fileSize = "~1.7GB"

Write-Host "File to download: qwen3-vl-2b-instruct-mmproj-f16.gguf" -ForegroundColor Yellow
Write-Host "Size: $fileSize" -ForegroundColor Yellow
Write-Host "Destination: $outputPath" -ForegroundColor Yellow
Write-Host ""

# Check if file already exists
if (Test-Path $outputPath) {
    Write-Host "⚠️  File already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to re-download? (y/n)"
    if ($response -ne "y") {
        Write-Host "Download cancelled." -ForegroundColor Red
        exit
    }
}

Write-Host "📥 Starting download..." -ForegroundColor Green
Write-Host "This may take several minutes depending on your internet speed." -ForegroundColor Cyan
Write-Host ""

try {
    # Download with progress
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $modelUrl -OutFile $outputPath -UseBasicParsing
    
    Write-Host ""
    Write-Host "✅ Download completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📂 File saved to: $outputPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Restart your llama-server (close current server terminal)" -ForegroundColor White
    Write-Host "2. Run: start_llama_server.bat" -ForegroundColor White
    Write-Host "3. Test: python verify_model.py" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download option:" -ForegroundColor Yellow
    Write-Host "1. Open: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/main" -ForegroundColor White
    Write-Host "2. Click on: qwen3-vl-2b-instruct-mmproj-f16.gguf" -ForegroundColor White
    Write-Host "3. Click 'Download' button" -ForegroundColor White
    Write-Host "4. Save to: $PSScriptRoot" -ForegroundColor White
    Write-Host ""
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
