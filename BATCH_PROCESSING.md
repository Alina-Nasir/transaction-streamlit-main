# Batch Processing Guide - Pakistani Bank Transaction Parser

## Overview

The batch processing service enables **24/7 automated processing** of bank transaction slips. When you place receipt images in a monitored folder, the system automatically:

1. **Detects** new image files
2. **Runs inference** on them using the same AI model
3. **Extracts** transaction details
4. **Saves** results to the database
5. **Organizes** files into success/failure folders
6. **Logs** all operations for monitoring

This eliminates manual processing of bulk receipts and enables continuous transaction data collection.

---

## Quick Start

### 1. Enable Batch Processing in Streamlit

1. Open the Pakistani Bank Transaction Parser application
2. Navigate to **"Batch Settings"** tab
3. Configure the folders:
   - **Incoming Folder**: Where you'll place new receipt images
   - **Processed Folder**: Where successful receipts are moved (optional)
   - **Failed Folder**: Where problematic files are moved
4. Click **"Save Settings"**
5. **Restart the application** for changes to take effect

### 2. How It Works

```
Your Receipt Images
        ↓
   Incoming Folder (Monitored 24/7)
        ↓
    File Detected → Debounce Wait (3 seconds for write to complete)
        ↓
  Run Inference (Same AI Model as Manual Processing)
        ↓
  Extract Transaction Details
        ↓
  Save to Database (%APPDATA%\PakistanBankParser\transactions.db)
        ↓
  Move File to Appropriate Folder:
    ✅ Processed Folder (if successful)
    ❌ Failed Folder (if error occurred)
        ↓
  Log Event to: %APPDATA%\PakistanBankParser\logs\
```

---

## Configuration

### Via Streamlit Settings Page (Recommended)

Open the **"Batch Settings"** page in the application to configure:

| Setting | Description | Default |
|---------|-------------|---------|
| **Incoming Folder** | Where to place new receipt images | `%USERPROFILE%\Pictures\BankSlips\incoming` |
| **Auto-move Processed** | Toggle to move successfully processed files | Enabled |
| **Processed Folder** | Where to move successful receipts | `%USERPROFILE%\Pictures\BankSlips\processed` |
| **Failed Folder** | Where to move files that fail processing | `%USERPROFILE%\Pictures\BankSlips\failed` |
| **Debounce Delay** | Wait time before processing new file (seconds) | 3.0 |
| **Inference Timeout** | Maximum time to wait for inference (seconds) | 300 |

### Batch Configuration File

Settings are stored in:
```
%APPDATA%\PakistanBankParser\config\batch_config.json
```

Example configuration:
```json
{
  "incoming_folder": "C:\\Users\\YourName\\Pictures\\BankSlips\\incoming",
  "processed_folder": "C:\\Users\\YourName\\Pictures\\BankSlips\\processed",
  "failed_folder": "C:\\Users\\YourName\\Pictures\\BankSlips\\failed",
  "auto_move_processed": true,
  "debounce_delay": 3.0,
  "enabled": false,
  "inference_timeout": 300
}
```

---

## File Organization

### Recommended Folder Structure

```
C:\Users\YourName\Pictures\BankSlips\
├── incoming/           ← Place new receipts here
├── processed/          ← Successfully processed receipts (auto-moved)
└── failed/             ← Failed receipts (auto-moved with error info)
```

### File Naming

- Files can be: `.jpg`, `.jpeg`, `.png`, or `.pdf`
- Files are auto-renamed if name conflict occurs (appends timestamp)
- Failed files have a `.error` file created with the error details

Example failed file report:
```
transaction_receipt.jpg.error
---
Processing Error: llama.cpp server not responding
Timestamp: 2025-01-15T14:32:45.123456
```

---

## Supported File Types

| Format | Status | Notes |
|--------|--------|-------|
| `.jpg` | ✅ Supported | Recommended format |
| `.jpeg` | ✅ Supported | Same as JPG |
| `.png` | ✅ Supported | Lossless, larger file size |
| `.pdf` | ✅ Supported | First page processed |
| Other formats | ❌ Not supported | Will be moved to failed folder |

---

## Database Integration

### Automatic Storage

All successfully processed transactions are saved to:
```
%APPDATA%\PakistanBankParser\transactions.db
```

Fields extracted and stored:
- Bank Name
- Transaction Date
- Transaction ID / Reference Number
- Amount
- From Account (Sender)
- From Account Number (Sender)
- From Bank Name (Sender)
- To Account (Receiver)
- To Account Number (Receiver)
- To Bank Name (Receiver)
- Branch
- Payment Mode
- Customer ID
- Cheque Number
- Remarks
- File Name (original receipt filename)
- Processed Date (timestamp)

### Viewing Results

1. Open the application
2. Go to **"View Database"** tab
3. Browse all processed transactions
4. Filter by date, bank, or transaction ID
5. Export to CSV or Excel

---

## Logging & Monitoring

### Log Files Location

```
%APPDATA%\PakistanBankParser\logs\
├── launcher_TIMESTAMP.log           ← Application startup/shutdown
├── batch_service_TIMESTAMP.log      ← Batch processor events
└── batch_processor_TIMESTAMP.log    ← Detailed processing logs
```

### Log Levels

- **DEBUG**: Detailed information (debounce checks, file size monitoring)
- **INFO**: General information (file detected, processing started, saved)
- **WARNING**: Warning conditions (file not ready, missing folder)
- **ERROR**: Error conditions (inference failed, database error)

### Viewing Logs

1. **In Streamlit**: Logs automatically displayed in Batch Settings page
2. **Manual**: Open log files with any text editor
3. **Command Line**: 
   ```bash
   type "%APPDATA%\PakistanBankParser\logs\batch_service_*.log"
   ```

---

## Advanced Configuration

### Setting Debounce Delay

The debounce delay prevents processing files that are still being written. Set based on your disk speed:

- **SSD/Fast Drive**: 2-3 seconds
- **HDD/Network Drive**: 5-10 seconds
- **Very Large Files**: 10-30 seconds

### Setting Inference Timeout

Timeout depends on system performance. Defaults to 300 seconds (5 minutes):

- **Modern CPU**: 30-60 seconds
- **Average CPU**: 120-300 seconds
- **Slow CPU**: 300-600 seconds (max)

### Disabling Auto-Move

If you want to keep files in the incoming folder:

1. Go to **Batch Settings**
2. Uncheck **"Auto-move Processed Files"**
3. Click Save

Files will remain in incoming folder but will be tracked in logs.

---

## Troubleshooting

### Problem: Batch Processor Not Starting

**Solution**:
1. Check if batch processing is enabled in settings
2. Verify incoming folder path is valid and accessible
3. Check logs: `%APPDATA%\PakistanBankParser\logs\`
4. Ensure llama.cpp server is running (main Streamlit window)

### Problem: Files Not Processing

**Causes & Solutions**:

| Issue | Solution |
|-------|----------|
| **Server not running** | Make sure llama.cpp server started (green output) |
| **Folder permission denied** | Check folder permissions, use user-writable folder |
| **File format unsupported** | Only .jpg, .jpeg, .png, .pdf supported |
| **Debounce timeout** | Increase debounce delay in settings |
| **Out of disk space** | Free up disk space, check database size |

### Problem: High CPU Usage

**Solutions**:
1. Batch processor uses CPU for inference (normal during processing)
2. Monitor number of files in incoming folder
3. Process files in batches rather than all at once
4. Consider adjusting inference timeout

### Problem: Database Errors

**Troubleshooting Steps**:
1. Check database file exists: `%APPDATA%\PakistanBankParser\transactions.db`
2. Verify disk space available
3. Check file permissions on database
4. Try deleting old log files to free space

**View Database Size**:
```bash
dir "%APPDATA%\PakistanBankParser\" /s
```

---

## Performance Tips

### 1. Batch Processing Strategy

**For Large Batches**:
- Copy files gradually instead of all at once
- Monitor logs to ensure processing speed
- Typical processing time: 30-120 seconds per image

**Estimated Throughput**:
- Modern CPU: 30-40 images/hour
- Average CPU: 15-25 images/hour
- Slow CPU: 5-15 images/hour

### 2. Disk Organization

Keep separate folders for:
- Current work: incoming/
- Archive: processed/ (periodically clean)
- Investigation: failed/ (review errors)

### 3. Database Maintenance

Monitor database size:
```bash
dir %APPDATA%\PakistanBankParser\transactions.db
```

Export old data periodically:
1. Open View Database
2. Use "Export All Records" to CSV/Excel
3. Store backups externally

### 4. System Resources

For optimal performance:
- Allocate 2+ GB RAM
- Keep system temperature normal
- Avoid other heavy tasks while processing
- Use SSD if possible (faster file detection)

---

## Windows Service Installation (Advanced)

### Prerequisites

- Administrator access
- NSSM (Non-Sucking Service Manager) installed: https://nssm.cc/download
- Extract `nssm.exe` to: `C:\Windows\System32\`

### Install as Windows Service

1. Open Command Prompt as **Administrator**
2. Navigate to application folder
3. Run:
   ```bash
   python run_batch_service.py install
   ```

### Manage Service

**Start Service**:
```bash
net start PakistanBankParserBatchService
```

**Stop Service**:
```bash
net stop PakistanBankParserBatchService
```

**Restart Service**:
```bash
net stop PakistanBankParserBatchService
net start PakistanBankParserBatchService
```

**Remove Service**:
```bash
python run_batch_service.py remove
```

**Check Status**:
```bash
nssm status PakistanBankParserBatchService
```

### Service Logs

Logs saved to:
```
%APPDATA%\PakistanBankParser\logs\batch_service_nssm.log
```

---

## Best Practices

### 1. **Start Simple**

- First test with 5-10 images
- Verify correct folder and database setup
- Check logs for any issues
- Gradually increase batch size

### 2. **Regular Monitoring**

- Review logs weekly
- Check failed folder for problem files
- Monitor database growth
- Verify accurate data extraction

### 3. **Data Validation**

- Spot-check extracted data for accuracy
- Review failed files to identify issues
- Keep backup of processed images
- Export database regularly

### 4. **System Maintenance**

- Clean old log files (keep 30 days)
- Archive processed images monthly
- Monitor disk space (keep 5GB+ free)
- Update system software regularly

### 5. **Error Recovery**

- Keep processed folder as backup
- Export database to external drive
- Document any custom folder setup
- Note any configuration changes

---

## FAQ

### Q: Can I process PDFs?
**A**: Yes! First page of PDF is automatically converted to image and processed.

### Q: How long does processing take?
**A**: 30-120 seconds per image depending on system speed and file size.

### Q: Can I move the folders?
**A**: Yes, update paths in Batch Settings and restart application.

### Q: What happens to failed files?
**A**: They're moved to failed folder with a `.error` file containing the error details.

### Q: Can I disable auto-move?
**A**: Yes, uncheck "Auto-move Processed Files" in settings.

### Q: Does processing continue if Streamlit closes?
**A**: Yes, batch processor runs independently and continues processing.

### Q: Where are transactions saved?
**A**: `%APPDATA%\PakistanBankParser\transactions.db`

### Q: How do I backup my data?
**A**: Export from "View Database" tab or copy `transactions.db` file.

### Q: Can I process very large batches?
**A**: Yes, but gradually copy files or increase debounce delay for stability.

### Q: What if I run out of disk space?
**A**: Delete old processed files and check database size. Archive old data.

---

## Support & Troubleshooting

### Useful Paths

| Item | Path |
|------|------|
| Database | `%APPDATA%\PakistanBankParser\transactions.db` |
| Logs | `%APPDATA%\PakistanBankParser\logs\` |
| Config | `%APPDATA%\PakistanBankParser\config\batch_config.json` |
| Batch Script | Same folder as `PakistanBankParser.exe` |

### Debug Information

When reporting issues, include:
1. **Log files** from: `%APPDATA%\PakistanBankParser\logs\`
2. **Config file**: `%APPDATA%\PakistanBankParser\config\batch_config.json`
3. **System info**: Windows version, RAM, disk space
4. **Error messages**: Screenshot or text copy

### Resources

- **Documentation**: This file
- **Settings UI**: "Batch Settings" tab in application
- **Logs**: Check `batch_processor_TIMESTAMP.log` for details
- **Test**: Place a single image in incoming folder to test

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial batch processing release |

---

## License

See LICENSE file in application directory.

---

**Last Updated**: January 2025
**Batch Processing Service v1.0**
