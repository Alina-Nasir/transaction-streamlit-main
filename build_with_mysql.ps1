# Build Script for Pakistani Bank Transaction Parser with MySQL Support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Pakistani Bank Transaction Parser" -ForegroundColor Cyan
Write-Host "with MySQL Database Support" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean previous builds
Write-Host "[1/5] Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
    Write-Host "  ✓ Removed build folder" -ForegroundColor Green
}
if (Test-Path "dist") {
    Remove-Item -Path "dist" -Recurse -Force
    Write-Host "  ✓ Removed dist folder" -ForegroundColor Green
}
if (Test-Path "installer_output") {
    Remove-Item -Path "installer_output" -Recurse -Force
    Write-Host "  ✓ Removed installer_output folder" -ForegroundColor Green
}
Write-Host ""

# Step 2: Verify required files
Write-Host "[2/5] Verifying required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "launcher.py",
    "streamlit_app.py",
    "db_manager.py",
    "inference_engine.py",
    "batch_processor_service.py",
    "batch_config.py",
    "validate_mysql.py",
    "PakistanBankParser.spec",
    "setup_mysql.iss",
    "models_latest\model.gguf",
    "models_latest\mmproj.gguf",
    "llama-cpu\llama-server.exe"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (MISSING)" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "ERROR: Some required files are missing!" -ForegroundColor Red
    Write-Host "Please ensure all files are present before building." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Check MySQL connector
Write-Host "[3/5] Checking MySQL connector..." -ForegroundColor Yellow
$mysqlCheck = python -c "import mysql.connector; print('OK')" 2>&1
if ($mysqlCheck -like "*OK*") {
    Write-Host "  ✓ mysql-connector-python is installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ mysql-connector-python not found" -ForegroundColor Red
    Write-Host "  Installing mysql-connector-python..." -ForegroundColor Yellow
    python -m pip install mysql-connector-python
    Write-Host "  ✓ Installed mysql-connector-python" -ForegroundColor Green
}
Write-Host ""

# Step 4: Build with PyInstaller
Write-Host "[4/5] Building executable with PyInstaller..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes depending on your system..." -ForegroundColor Cyan
Write-Host ""

pyinstaller --clean PakistanBankParser.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: PyInstaller build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  ✓ PyInstaller build completed successfully" -ForegroundColor Green
Write-Host ""

# Step 5: Verify build output
Write-Host "[5/5] Verifying build output..." -ForegroundColor Yellow
if (Test-Path "dist\PakistanBankParser\PakistanBankParser.exe") {
    Write-Host "  ✓ Executable created: dist\PakistanBankParser\PakistanBankParser.exe" -ForegroundColor Green
} else {
    Write-Host "  ✗ Executable not found!" -ForegroundColor Red
    exit 1
}

if (Test-Path "dist\PakistanBankParser\_internal") {
    Write-Host "  ✓ _internal folder created with all dependencies" -ForegroundColor Green
} else {
    Write-Host "  ✗ _internal folder not found!" -ForegroundColor Red
    exit 1
}

# Check for validate_mysql.py in _internal
if (Test-Path "dist\PakistanBankParser\_internal\validate_mysql.py") {
    Write-Host "  ✓ MySQL validation script bundled" -ForegroundColor Green
} else {
    Write-Host "  ✗ MySQL validation script missing!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Test the executable: dist\PakistanBankParser\PakistanBankParser.exe" -ForegroundColor White
Write-Host "2. Create installer with Inno Setup:" -ForegroundColor White
Write-Host "   - Open Inno Setup Compiler" -ForegroundColor White
Write-Host "   - Compile: setup_mysql.iss" -ForegroundColor White
Write-Host "3. Installer will be in: installer_output\" -ForegroundColor White
Write-Host ""
Write-Host "Important Notes:" -ForegroundColor Yellow
Write-Host "• The installer will prompt users for MySQL credentials" -ForegroundColor White
Write-Host "• MySQL connection will be validated during installation" -ForegroundColor White
Write-Host "• Users must have MySQL Server running" -ForegroundColor White
Write-Host ""
