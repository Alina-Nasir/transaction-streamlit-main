# Batch Processing - Quick Reference

## 📁 New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `inference_engine.py` | Shared AI inference logic | ~450 |
| `batch_processor_service.py` | File monitoring & processing | ~450 |
| `batch_config.py` | Configuration management | ~200 |
| `run_batch_service.py` | Windows service wrapper | ~450 |
| `BATCH_PROCESSING.md` | User documentation | ~600 |
| `BATCH_PROCESSING_IMPLEMENTATION.md` | Developer documentation | ~800 |

## 📝 Files Modified

| File | Changes | Impact |
|------|---------|--------|
| `launcher.py` | Added batch processor start/stop | +100 lines |
| `streamlit_app.py` | Shared inference, settings page | +150 lines |
| `db_manager.py` | Added path helpers | +30 lines |
| `PakistanBankParser.spec` | Added modules to build | +5 lines |
| `requirements.txt` | Added watchdog==5.0.3 | +1 line |

## 🔄 Data Flow

```
Incoming Folder
      ↓
Watchdog Detection (FileSystemEventHandler)
      ↓
Debounce Check (3 seconds)
      ↓
File Readiness Check (Size stability)
      ↓
inference_engine.call_local_model_with_image()
      ↓
inference_engine.extract_json_from_response()
      ↓
db_manager.insert_record()
      ↓
Move to Processed/Failed Folder
      ↓
Log Event
```

## 🎯 Key Classes & Functions

### inference_engine.py
```python
class Functions:
    get_pakistani_bank_prompt()          # Get AI prompt
    resize_image_to_512p(image)          # Preprocess image
    encode_image_to_base64(image)        # Convert to base64
    call_local_model_with_image()        # Run inference
    extract_json_from_response()         # Parse AI response
    convert_pdf_to_images(pdf_file)      # PDF handling
    _standardize_fields()                # Field normalization
    _get_empty_response()                # Empty template
```

### batch_processor_service.py
```python
class TransactionFileHandler(FileSystemEventHandler):
    on_created(event)                    # File detected
    on_modified(event)                   # File changed
    _is_supported_file(path)             # Type validation
    _wait_for_file_ready(path)           # Readiness check
    _process_file(path)                  # Main processing
    _move_to_processed(path)             # Success move
    _move_to_failed(path, reason)        # Error move
    get_stats()                          # Get statistics

class BatchProcessorService:
    start()                              # Start monitoring
    stop()                               # Stop monitoring
    is_running()                         # Check status
    get_stats()                          # Get statistics

function run_batch_processor(config, run_in_foreground)
    # Main entry point
```

### batch_config.py
```python
load_config()                           # Load from JSON
save_config(config)                     # Save to JSON
update_config(updates)                  # Update values
validate_config(config)                 # Validate settings
get_config_summary()                    # User-friendly display
validate_folder_path(path)              # Path validation
get_default_config()                    # Default settings
```

### run_batch_service.py
```python
install_service()                       # NSSM install
remove_service()                        # NSSM remove
start_service()                         # net start
stop_service()                          # net stop
check_status()                          # Check if running
run_foreground()                        # Direct execution
```

## 📊 Configuration

### Default Paths (Windows)

```
Incoming:  C:\Users\{UserName}\Pictures\BankSlips\incoming
Processed: C:\Users\{UserName}\Pictures\BankSlips\processed
Failed:    C:\Users\{UserName}\Pictures\BankSlips\failed
Database:  %APPDATA%\PakistanBankParser\transactions.db
Config:    %APPDATA%\PakistanBankParser\config\batch_config.json
Logs:      %APPDATA%\PakistanBankParser\logs\
```

### Configuration File
```json
{
  "incoming_folder": "path/to/incoming",
  "processed_folder": "path/to/processed",
  "failed_folder": "path/to/failed",
  "auto_move_processed": true,
  "debounce_delay": 3.0,
  "inference_timeout": 300,
  "enabled": false
}
```

## 🔌 API Integration Points

### inference_engine → llama.cpp
```
POST http://localhost:8088/v1/chat/completions
- Image: base64 JPEG
- Timeout: 300 seconds (configurable)
- Response: JSON with extracted fields
```

### batch_processor → db_manager
```python
db_manager.insert_record(transaction_data)
# Saves all 17 transaction fields + metadata
```

### batch_processor → batch_config
```python
batch_config.load_config()    # Get settings
batch_config.validate_config() # Verify valid
```

## 🚀 Startup Sequence

1. **launcher.py main()**
2. Start llama.cpp server
3. Start batch processor (if enabled)
4. Start Streamlit app
5. Batch processor runs independently

## 🛑 Shutdown Sequence (Reverse Order)

1. Streamlit closes (Ctrl+C)
2. Stop batch processor gracefully (5s timeout)
3. Stop batch processor forcefully (if needed)
4. Stop llama.cpp server gracefully (5s timeout)
5. Stop llama.cpp server forcefully (if needed)

## 📋 File Processing States

```
1. INCOMING (User places file)
   ↓
2. DETECTED (Watchdog detects change)
   ↓
3. DEBOUNCE (Waiting 3 seconds)
   ↓
4. CHECKING (File readiness verification)
   ↓
5. PROCESSING (Running inference)
   ↓
6. SAVING (Writing to database)
   ↓
   OPTION A: SUCCESS → PROCESSED folder ✅
   OPTION B: FAILED → FAILED folder ❌
```

## 🔍 Logging Levels

| Level | When Used | Example |
|-------|-----------|---------|
| DEBUG | Detailed tracking | Debounce checks, size monitoring |
| INFO | Important events | File detected, processing started |
| WARNING | Attention needed | File not ready, timeout |
| ERROR | Failures | Inference failed, DB error |

## ⚙️ Streamlit Integration

### New Page: "Batch Settings"
- Configure incoming/processed/failed folders
- Toggle auto-move
- Adjust debounce delay
- Set inference timeout
- View current configuration
- Show instructions

### Navigation
```python
page = st.sidebar.radio(
    "Select Page",
    ["Process Transactions", "View Database", "Batch Settings"]
)
```

## 🔧 Service Management (Windows)

```bash
# Install
python run_batch_service.py install

# Start/Stop
net start PakistanBankParserBatchService
net stop PakistanBankParserBatchService

# Status
nssm status PakistanBankParserBatchService

# Remove
python run_batch_service.py remove

# Logs
type %APPDATA%\PakistanBankParser\logs\batch_service_nssm.log
```

## 📦 Dependencies

### New
- `watchdog==5.0.3` - File system monitoring

### Existing (Reused)
- PIL - Image processing
- requests - HTTP calls
- sqlite3 - Database
- streamlit - Web UI
- pypdfium2 - PDF handling

## 🧪 Testing Checklist

- [ ] Detect single image file
- [ ] Process image successfully
- [ ] Save to database
- [ ] Move to processed folder
- [ ] Generate log entry
- [ ] Detect multiple files
- [ ] Handle corrupted file (moved to failed)
- [ ] Handle missing model server
- [ ] Database persist across runs
- [ ] Configuration changes apply
- [ ] Batch processor survives Streamlit close
- [ ] Clean shutdown (no orphan processes)

## 📞 Troubleshooting Quick Tips

| Problem | Quick Fix |
|---------|-----------|
| Files not processing | Check llama.cpp running, check folder permissions |
| Can't find database | Check `%APPDATA%\PakistanBankParser\transactions.db` |
| High CPU | Normal during inference, expected |
| Service won't install | Install NSSM to `C:\Windows\System32\` |
| Permission denied | Use user-writable folder, not C:\ or Program Files |
| Port 8088 in use | Check for existing llama-server process |

## 🔐 Error Handling

All errors caught and logged without crashing:
- File access errors → Move to failed, log error
- Inference timeouts → Log timeout, move to failed
- Database errors → Log error, keep file in incoming
- Service errors → Log, continue monitoring
- Invalid config → Log, use defaults

## 📈 Performance

- **Per Image**: 30-120 seconds (mostly inference)
- **Throughput**: 30-40 images/hour (modern CPU)
- **Memory**: ~500MB for batch processor
- **CPU**: 100% during inference (single-threaded)

## 🎓 Learning Path

1. **User**: Read `BATCH_PROCESSING.md`
2. **Developer**: Read `BATCH_PROCESSING_IMPLEMENTATION.md`
3. **Code Review**: Start with `inference_engine.py`
4. **Integration**: Review `launcher.py` and `streamlit_app.py`
5. **Testing**: See testing checklist above

## 📖 Code Entry Points

**Starting batch processing**:
```python
import batch_processor_service
import batch_config

config = batch_config.load_config()
batch_processor_service.run_batch_processor(config)
```

**Shared inference**:
```python
import inference_engine

response = inference_engine.call_local_model_with_image(image_path)
data = inference_engine.extract_json_from_response(response)
```

**Configuration**:
```python
import batch_config

config = batch_config.load_config()
batch_config.update_config({'enabled': True})
```

## 🚨 Critical Paths

Must exist for batch processing:
- `inference_engine.py` - Core inference logic
- `llama-server.exe` - AI model server
- `model.gguf` - Vision-language model
- `mmproj.gguf` - Multimodal projection

Missing paths will fail gracefully with logging.

## 📞 Support Resources

1. **Logs**: `%APPDATA%\PakistanBankParser\logs\batch_processor_*.log`
2. **Config**: `%APPDATA%\PakistanBankParser\config\batch_config.json`
3. **Database**: `%APPDATA%\PakistanBankParser\transactions.db`
4. **Help**: Read `BATCH_PROCESSING.md` in application directory

---

**Last Updated**: January 2025
**Batch Processing v1.0**
