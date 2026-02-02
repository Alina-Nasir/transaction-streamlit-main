# Pakistani Bank Transaction Parser - Installation Guide

## Overview
Complete standalone application for extracting transaction details from Pakistani bank slips using AI vision technology. No internet connection required after installation.

## What's Included
- ✅ **Python Runtime**: Complete Python 3.10 environment (no separate installation needed)
- ✅ **AI Vision Model**: Qwen3-VL-2B (1.5GB) for OCR and data extraction
- ✅ **llama.cpp Engine**: Optimized CPU inference (20+ DLLs for various CPU architectures)
- ✅ **SQLite Database**: Persistent storage for all extracted transactions
- ✅ **Streamlit Web UI**: Beautiful browser-based interface
- ✅ **All Dependencies**: requests, pandas, PIL, pypdfium2, etc.

## System Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **Storage**: 3GB free disk space
- **CPU**: Any modern Intel/AMD processor (optimized for all architectures)
- **Internet**: Only needed for installation, not for running the app

## Installation Steps

### Option 1: Using Inno Setup Installer (Recommended)

1. **Download** the installer: `PakistanBankParser_Setup_v1.0.0.exe`

2. **Run the installer**:
   - Double-click the .exe file
   - Click "Yes" if Windows asks for permission
   - Follow the installation wizard
   - Choose installation directory (default: C:\Program Files\Pakistani Bank Transaction Parser)

3. **Launch the application**:
   - From Desktop shortcut (if created during installation)
   - From Start Menu: "Pakistani Bank Transaction Parser"
   - Or run: `C:\Program Files\Pakistani Bank Transaction Parser\PakistanBankParser.exe`

4. **First Launch**:
   - Wait 15-20 seconds for AI model to load
   - Your default browser will automatically open
   - Application URL: http://localhost:8501

### Option 2: Portable Installation (No Installer)

1. **Extract** the `PakistanBankParser` folder to any location

2. **Run** `PakistanBankParser.exe` from the extracted folder

3. **Wait** for the browser to open automatically

## Using the Application

### Quick Start

1. **Upload** a transaction slip image (JPG, PNG, or PDF)
   - Drag & drop or click to browse
   - Supports Pakistani bank slips from all major banks

2. **Click "Process This Slip"**
   - AI will extract all transaction details
   - Processing takes 30-90 seconds (CPU-dependent)

3. **View Results**
   - Extracted data displayed immediately
   - Automatically saved to database

4. **Export Data**
   - Download as CSV or Excel
   - All transactions saved in one place

### Supported Banks
- Meezan Bank
- HBL (Habib Bank Limited)
- UBL (United Bank Limited)
- Bank Alfalah
- Allied Bank
- Standard Chartered
- Bank Islami
- Faysal Bank
- JS Bank
- And all other Pakistani banks

### Extracted Fields
- Bank Name
- Transaction Date
- Transaction ID
- Amount
- Sender Account Name & Number
- Sender Bank Name
- Receiver Account Name & Number
- Receiver Bank Name
- Branch
- Payment Mode
- Customer ID
- Cheque Number
- Remarks

## Database Location
All extracted transactions are stored in:
```
%APPDATA%\PakistanBankParser\transactions.db
```

To access:
1. Press `Win + R`
2. Type: `%APPDATA%\PakistanBankParser`
3. Press Enter

## Troubleshooting

### Application Won't Start
- **Check**: Make sure port 8080 and 8501 are not in use
- **Solution**: Close other applications using these ports
- **Check Task Manager**: Look for "llama-server.exe" processes and end them

### Browser Doesn't Open Automatically
- **Manual Access**: Open browser and go to `http://localhost:8501`
- **Alternative Port**: Check console for actual port if 8501 is busy

### Slow Processing
- **Normal**: First image takes 75-150 seconds (CPU image encoding)
- **Cached**: Repeated images process much faster
- **Performance**: Depends on CPU speed and available RAM
- **Tip**: Close other applications to free up CPU/RAM

### "Server Not Running" Error
- **Wait**: Give 15-20 seconds for server startup
- **Restart**: Close and reopen the application
- **Check Console**: Look for error messages in the black window

### Model Loading Issues
- **Check Files**: Verify `_internal/model/` contains:
  - `model.gguf` (~1.1GB)
  - `mmproj.gguf` (~425MB)
- **Reinstall**: If files are missing, reinstall the application

## Uninstallation

### If Installed via Installer
1. Go to: Settings → Apps → Apps & features
2. Find "Pakistani Bank Transaction Parser"
3. Click "Uninstall"
4. Database will be removed from %APPDATA%

### If Portable Installation
1. Simply delete the `PakistanBankParser` folder
2. Manually delete database if needed: `%APPDATA%\PakistanBankParser`

## Technical Details

### Architecture
```
PakistanBankParser.exe
├── launcher.py (Entry point)
├── llama-server.exe (AI inference engine)
│   └── 20+ CPU-optimized DLLs
├── model.gguf (1.1GB AI model)
├── mmproj.gguf (425MB vision encoder)
├── streamlit_app.py (Web UI)
├── db_manager.py (Database handler)
└── Python 3.10 runtime + all packages
```

### Network Usage
- **Installation**: Downloads ~2.5GB (installer size)
- **Runtime**: No internet required (100% offline)
- **Local Only**: All processing happens on your computer
- **Privacy**: No data sent to external servers

### Performance Optimization
- CPU threading: Uses all available CPU cores
- Context size: 8192 tokens
- Batch size: 2048 (optimized for speed)
- Memory locking: Prevents swapping for faster inference

## Support & Updates

### Getting Help
- Check `QUICKSTART.txt` for quick tips
- Review this README for detailed information
- Contact support (if available)

### Updates
- Download new installer version
- Uninstall old version (data preserved)
- Install new version

## License
See `LICENSE` file in installation directory.

## Credits
- **AI Model**: Qwen3-VL-2B by Alibaba Cloud
- **Inference Engine**: llama.cpp by Georgi Gerganov
- **UI Framework**: Streamlit
- **Database**: SQLite

---

**Version**: 1.0.0  
**Build Date**: January 2026  
**Package Size**: ~2.5 GB installed
