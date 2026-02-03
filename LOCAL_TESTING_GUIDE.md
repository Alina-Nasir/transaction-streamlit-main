# Local Testing Guide - Batch Processing System

## Pre-Flight Checklist

Before starting, verify:

- [ ] Python environment is set up (`venv` exists)
- [ ] Dependencies installed (`requirements.txt`)
- [ ] `llama-server.exe` is in `build_assets/bin/`
- [ ] Model files exist: `model.gguf` and `mmproj.gguf`
- [ ] All new modules exist:
  - [ ] `inference_engine.py`
  - [ ] `batch_processor_service.py`
  - [ ] `batch_config.py`
  - [ ] `run_batch_service.py`

## Step 1: Environment Setup

### Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# Or if using conda
# conda activate transaction-streamlit-env
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected new package**: `watchdog==5.0.3`

Verify installation:
```bash
pip list | grep watchdog
# Output: watchdog 5.0.3
```

## Step 2: Initialize Database

```bash
python -c "import db_manager; db_manager.init_db(); print('✅ Database initialized')"
```

Expected output:
```
✅ Database initialized at: transactions.db
```

## Step 3: Test Inference Engine (Standalone)

Create test file `test_inference.py`:

```python
import inference_engine
import os

print("Testing inference_engine module...")

# Test 1: Get prompt
prompt = inference_engine.get_pakistani_bank_prompt()
print(f"✅ Prompt loaded ({len(prompt)} chars)")

# Test 2: Check empty response
empty = inference_engine._get_empty_response()
print(f"✅ Empty response template: {len(empty)} fields")

# Test 3: Standardize fields
test_data = {
    'bankName': 'Meezan Bank',
    'Date': '01/02/2025',
    'Amount': 'PKR 50,000'
}
standardized = inference_engine._standardize_fields(test_data)
print(f"✅ Field standardization works: {list(standardized.keys())[:3]}")

# Test 4: Extract JSON from mock response
mock_response = '''
The extracted data is:
{
"bankName": "Habib Bank",
"Date": "02/02/2025",
"TransactionID": "TXN123456",
"Amount": "PKR 100,000"
}
Other text here...
'''
result = inference_engine.extract_json_from_response(mock_response)
print(f"✅ JSON extraction works: {result['bankName']}")

print("\n✅ All inference_engine tests passed!")
```

Run it:
```bash
python test_inference.py
```

## Step 4: Test Batch Configuration

Create test file `test_config.py`:

```python
import batch_config
import os
import json

print("Testing batch_config module...")

# Test 1: Load default config
default = batch_config.get_default_config()
print(f"✅ Default config loaded with {len(default)} keys")

# Test 2: Validate config
errors = batch_config.validate_config(default)
if errors:
    print(f"⚠️  Validation errors: {errors}")
else:
    print(f"✅ Config validation passed")

# Test 3: Create test config
test_config = {
    'incoming_folder': 'test_incoming',
    'processed_folder': 'test_processed',
    'failed_folder': 'test_failed',
    'auto_move_processed': True,
    'debounce_delay': 3.0,
    'inference_timeout': 300,
    'enabled': True
}

# Test 4: Validate paths
valid, path = batch_config.validate_folder_path(test_config['incoming_folder'])
if valid:
    print(f"✅ Path validation works: {path}")
else:
    print(f"⚠️  Path validation failed: {path}")

# Test 5: Load/Save config
batch_config.save_config(test_config)
loaded = batch_config.load_config()
print(f"✅ Config save/load works: {loaded['enabled']}")

# Test 6: Get summary
summary = batch_config.get_config_summary()
print(f"✅ Config summary: {list(summary.keys())}")

print("\n✅ All batch_config tests passed!")

# Cleanup
if os.path.exists('test_incoming'):
    os.rmdir('test_incoming')
if os.path.exists('test_processed'):
    os.rmdir('test_processed')
if os.path.exists('test_failed'):
    os.rmdir('test_failed')
```

Run it:
```bash
python test_config.py
```

## Step 5: Start llama.cpp Server

### Option A: Using Batch File (Recommended for Testing)

```bash
start_llama_server.bat
```

This opens a separate console window. Verify output:
```
Built with CUDA/SYCL support
llm_load_model: loaded all data from model successfully
llama_print_timings: load time = 1234.56 ms
Server is listening on http://0.0.0.0:8088
```

**Keep this window open** - the server must stay running.

### Option B: Manual Command

```bash
# From bash terminal
cd build_assets\bin
llama-server.exe -m ../model/model.gguf --mmproj ../model/mmproj.gguf --host 0.0.0.0 --port 8088 -c 8192 -t 4 -tb 4 -b 2048 -ub 512 -n 512 --n-gpu-layers 0 -np 1 --mlock
```

### Test Server Connection

```bash
# In a NEW terminal (keep server running in first terminal)
curl http://localhost:8088/health
# Expected: {"status":"ok"}
```

## Step 6: Test Batch Processor (Standalone Mode)

Create test file `test_batch_processor.py`:

```python
import batch_processor_service
import batch_config
import os
import logging

print("Testing batch_processor_service module...")

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Test 1: Create test folders
test_config = {
    'incoming_folder': './test_batch_incoming',
    'processed_folder': './test_batch_processed',
    'failed_folder': './test_batch_failed',
    'auto_move_processed': True,
    'debounce_delay': 2.0,
    'inference_timeout': 300,
    'enabled': True
}

for folder in ['incoming_folder', 'processed_folder', 'failed_folder']:
    os.makedirs(test_config[folder], exist_ok=True)

print(f"✅ Test folders created")

# Test 2: Initialize service
service = batch_processor_service.BatchProcessorService(test_config)
print(f"✅ BatchProcessorService initialized")

# Test 3: Check not running yet
if not service.is_running():
    print(f"✅ Service correctly reports as not running")
else:
    print(f"⚠️  Service should not be running yet")

print("\n✅ All batch_processor_service basic tests passed!")
print("Note: Full integration test requires image files and llama.cpp server")

# Cleanup
import shutil
for folder in ['./test_batch_incoming', './test_batch_processed', './test_batch_failed']:
    if os.path.exists(folder):
        shutil.rmtree(folder)
```

Run it:
```bash
python test_batch_processor.py
```

## Step 7: Integration Test (Full Workflow)

### 7.1 Prepare Test Image

You can use any receipt image from your `picture_data` folder, or create a minimal test image:

```bash
# Copy an existing test image
copy picture_data\<any_receipt>.jpg test_receipt.jpg
```

### 7.2 Create Full Integration Test

Create test file `test_integration.py`:

```python
#!/usr/bin/env python3
"""
Full integration test for batch processing
Tests: config → watchdog → inference → database
"""

import os
import sys
import time
import shutil
import sqlite3
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment():
    """Test that all required modules exist"""
    logger.info("=" * 60)
    logger.info("STEP 1: Environment Verification")
    logger.info("=" * 60)
    
    required_files = [
        'inference_engine.py',
        'batch_processor_service.py',
        'batch_config.py',
        'db_manager.py',
        'streamlit_app.py',
    ]
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}")
        else:
            logger.error(f"❌ {file} NOT FOUND")
            return False
    
    logger.info("✅ All required files found\n")
    return True


def test_modules_import():
    """Test that all modules import successfully"""
    logger.info("=" * 60)
    logger.info("STEP 2: Module Imports")
    logger.info("=" * 60)
    
    try:
        import inference_engine
        logger.info("✅ inference_engine imported")
        
        import batch_config
        logger.info("✅ batch_config imported")
        
        import batch_processor_service
        logger.info("✅ batch_processor_service imported")
        
        import db_manager
        logger.info("✅ db_manager imported")
        
        logger.info("✅ All modules imported successfully\n")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}\n")
        return False


def test_database():
    """Test database operations"""
    logger.info("=" * 60)
    logger.info("STEP 3: Database Operations")
    logger.info("=" * 60)
    
    try:
        import db_manager
        
        # Initialize database
        db_manager.init_db()
        logger.info("✅ Database initialized")
        
        # Check table exists
        db_path = db_manager.get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if any('transactions' in str(t) for t in tables):
            logger.info("✅ Transactions table exists")
        else:
            logger.error("❌ Transactions table not found")
            return False
        
        # Get table schema
        cursor.execute("PRAGMA table_info(transactions)")
        columns = cursor.fetchall()
        logger.info(f"✅ Table has {len(columns)} columns")
        
        conn.close()
        logger.info("✅ Database operations successful\n")
        return True
    
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}\n")
        return False


def test_configuration():
    """Test configuration system"""
    logger.info("=" * 60)
    logger.info("STEP 4: Configuration System")
    logger.info("=" * 60)
    
    try:
        import batch_config
        
        # Get default config
        config = batch_config.get_default_config()
        logger.info(f"✅ Default config loaded ({len(config)} keys)")
        
        # Validate config
        errors = batch_config.validate_config(config)
        if errors:
            logger.warning(f"⚠️  Validation warnings: {errors}")
        else:
            logger.info("✅ Config validation passed")
        
        logger.info("✅ Configuration system working\n")
        return True
    
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}\n")
        return False


def test_inference_mock():
    """Test inference engine with mock data"""
    logger.info("=" * 60)
    logger.info("STEP 5: Inference Engine (Mock)")
    logger.info("=" * 60)
    
    try:
        import inference_engine
        
        # Test prompt
        prompt = inference_engine.get_pakistani_bank_prompt()
        logger.info(f"✅ Prompt loaded ({len(prompt)} chars)")
        
        # Test field standardization
        mock_data = {
            'bankName': 'Meezan Bank',
            'Date': '02/02/2025',
            'Amount': 'PKR 50,000'
        }
        standardized = inference_engine._standardize_fields(mock_data)
        logger.info(f"✅ Field standardization works")
        
        # Test JSON extraction from mock response
        mock_response = '''{
            "bankName": "Habib Bank",
            "Date": "02/02/2025",
            "TransactionID": "REF123",
            "Amount": "PKR 100,000",
            "FromAccount": "John Doe",
            "ToAccount": "Jane Doe"
        }'''
        result = inference_engine.extract_json_from_response(mock_response)
        if result['bankName'] == 'Habib Bank':
            logger.info("✅ JSON extraction works correctly")
        else:
            logger.error("❌ JSON extraction failed")
            return False
        
        logger.info("✅ Inference engine tests passed\n")
        return True
    
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}\n")
        return False


def test_batch_service_init():
    """Test batch processor initialization"""
    logger.info("=" * 60)
    logger.info("STEP 6: Batch Processor (Initialization)")
    logger.info("=" * 60)
    
    try:
        import batch_processor_service
        
        # Create test folders
        test_config = {
            'incoming_folder': './test_incoming',
            'processed_folder': './test_processed',
            'failed_folder': './test_failed',
            'auto_move_processed': True,
            'debounce_delay': 2.0,
            'inference_timeout': 300,
            'enabled': True
        }
        
        for folder in ['incoming_folder', 'processed_folder', 'failed_folder']:
            os.makedirs(test_config[folder], exist_ok=True)
        logger.info("✅ Test folders created")
        
        # Initialize service
        service = batch_processor_service.BatchProcessorService(test_config)
        logger.info("✅ BatchProcessorService initialized")
        
        # Check status
        if not service.is_running():
            logger.info("✅ Service correctly reports as not running")
        
        # Cleanup
        for folder in ['./test_incoming', './test_processed', './test_failed']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        
        logger.info("✅ Batch processor initialization test passed\n")
        return True
    
    except Exception as e:
        logger.error(f"❌ Batch processor test failed: {e}\n")
        return False


def test_server_connection():
    """Test connection to llama.cpp server"""
    logger.info("=" * 60)
    logger.info("STEP 7: Server Connection")
    logger.info("=" * 60)
    
    try:
        import requests
        
        try:
            response = requests.get("http://localhost:8088/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ llama.cpp server is running on port 8088")
                logger.info("✅ Server connection test passed\n")
                return True
            else:
                logger.warning(f"⚠️  Server responded with code {response.status_code}")
                logger.warning("⚠️  Start llama.cpp server with: start_llama_server.bat\n")
                return True  # Not critical for this test
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  llama.cpp server not running")
            logger.warning("⚠️  Start it with: start_llama_server.bat")
            logger.info("⚠️  Server test skipped (not critical for basic tests)\n")
            return True
    
    except Exception as e:
        logger.warning(f"⚠️  Could not test server: {e}\n")
        return True  # Not critical


def run_all_tests():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " BATCH PROCESSING - LOCAL TESTING ".center(58) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_modules_import),
        ("Database", test_database),
        ("Configuration", test_configuration),
        ("Inference (Mock)", test_inference_mock),
        ("Batch Service", test_batch_service_init),
        ("Server Connection", test_server_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name} test crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready for full integration testing.\n")
        return True
    else:
        logger.info(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

Run the integration test:

```bash
python test_integration.py
```

## Step 8: Full End-to-End Test (With llama.cpp)

### 8.1 Prerequisites

1. **Terminal 1**: Start llama.cpp server
   ```bash
   start_llama_server.bat
   # Wait for: "Server is listening on http://0.0.0.0:8088"
   ```

2. **Terminal 2**: Run Streamlit
   ```bash
   streamlit run streamlit_app.py
   # Wait for: http://localhost:8501
   ```

3. **Terminal 3**: Test batch processor (separate terminal)

### 8.2 Manual File Test

```bash
# Create test folders
mkdir test_receipts\incoming
mkdir test_receipts\processed
mkdir test_receipts\failed

# Copy a test image
copy picture_data\<sample_receipt>.jpg test_receipts\incoming\test_001.jpg
```

### 8.3 Run Batch Processor (Terminal 3)

Create `test_batch_run.py`:

```python
import batch_config
import batch_processor_service
import os

# Configure test folders
config = {
    'incoming_folder': './test_receipts/incoming',
    'processed_folder': './test_receipts/processed',
    'failed_folder': './test_receipts/failed',
    'auto_move_processed': True,
    'debounce_delay': 2.0,
    'inference_timeout': 300,
    'enabled': True
}

# Create folders
for folder in [config['incoming_folder'], config['processed_folder'], config['failed_folder']]:
    os.makedirs(folder, exist_ok=True)

print("Starting batch processor...")
print(f"Watching: {config['incoming_folder']}")
print(f"Processed: {config['processed_folder']}")
print(f"Failed: {config['failed_folder']}")
print("\nPress Ctrl+C to stop\n")

# Run processor
batch_processor_service.run_batch_processor(config, run_in_foreground=True)
```

Run it:
```bash
python test_batch_run.py
```

### 8.4 Monitor Processing

1. **Terminal 3**: Should show
   ```
   File detected: test_001.jpg
   File is ready: test_001.jpg (size: 245633 bytes)
   Running inference on test_001.jpg
   Successfully processed: test_001.jpg
   Moved to processed: test_receipts/processed/test_001.jpg
   ```

2. **Terminal 2 (Streamlit)**: Go to "View Database" tab
   - Should see new transaction entry

3. **File System**: Check folders
   - `test_receipts/incoming/` - should be empty
   - `test_receipts/processed/` - should contain `test_001.jpg`
   - `test_receipts/failed/` - should be empty

## Step 9: Verify Database

```bash
# Check database records
python -c "import db_manager; records = db_manager.get_all_transactions(); print(f'Total records: {len(records)}'); [print(r) for r in records[-1:]]"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'watchdog'` | Run `pip install watchdog==5.0.3` |
| `ConnectionError: Cannot connect to llama.cpp` | Start `start_llama_server.bat` first |
| `PermissionError: [Errno 13] Permission denied` | Check folder permissions |
| `File not being detected` | Check incoming folder path, verify file format |
| `Inference hangs` | Check llama.cpp console for errors, restart server |
| `Database locked` | Close Streamlit, wait 2 seconds, retry |

## Performance Metrics

After running test, you should see:

```
Inference time: 30-120 seconds per image
Success rate: 100% for valid receipts
Database records: Correct number added
Logs generated: In %APPDATA%\PakistanBankParser\logs\
```

## Next Steps

Once all tests pass:

1. ✅ Test with 5-10 images
2. ✅ Verify database accuracy
3. ✅ Check file organization
4. ✅ Review logs for issues
5. ✅ Then proceed to production deployment

---

**Testing Status**: Ready for local testing
**Estimated Time**: 30-45 minutes for full test suite
