# Build Verification Checklist

## Files Required for Build

### Core Application Files
✅ launcher.py
✅ streamlit_app.py
✅ db_manager.py
✅ inference_engine.py
✅ batch_processor_service.py
✅ batch_config.py
✅ run_batch_service.py
✅ batch_processor_start.py

### MySQL Support Files
✅ validate_mysql.py
✅ validate_mysql.bat

### Model Files
✅ models_latest/model.gguf
✅ models_latest/mmproj.gguf

### llama.cpp Files
✅ llama-cpu/llama-server.exe
✅ llama-cpu/*.dll files

### Configuration Files
✅ PakistanBankParser.spec (updated with MySQL support)
✅ setup.iss (updated with MySQL credential collection)
✅ requirements.txt (includes mysql-connector-python)

## Build Commands

### 1. Clean Previous Builds
```powershell
Remove-Item -Path "build","dist" -Recurse -Force -ErrorAction SilentlyContinue
```

### 2. Build with PyInstaller
```powershell
python -m pyinstaller --clean PakistanBankParser.spec
```

### 3. Compile Installer
```powershell
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
```

## Installer Features

### MySQL Configuration During Installation
1. User enters:
   - Host (default: localhost)
   - Database Name (default: transactions_db)
   - Username (default: root)
   - Password
   
2. Validation using mysql.exe:
   - Tests connection with provided credentials
   - Creates database if it doesn't exist
   - Shows success/error messages
   - Always continues installation

3. Configuration saved to:
   - %APPDATA%\PakistanBankParser\config\db_config.json

### Batch Processing Configuration
1. User selects folder for incoming bank slips
2. Configuration saved to:
   - %APPDATA%\PakistanBankParser\config\batch_config.json

## Key Changes Made

### PakistanBankParser.spec
- Added mysql-connector-python to hiddenimports
- Added validate_mysql.py to bundled files
- Added validate_mysql.bat to bundled files

### setup.iss
- Added MySQL configuration page with 4 input fields
- Added validation using mysql.exe command line
- Bundled validate_mysql.bat
- Creates db_config.json with user credentials
- Graceful fallback if mysql.exe not in PATH

### db_manager.py
- Added MySQL connector support
- Configuration loading from db_config.json
- Auto-fallback removed (MySQL only)
- Compatible with both MySQL and SQLite query syntax

### streamlit_app.py
- Updated to use db_manager.get_connection()
- Works with MySQL database
- Auto-refresh with fragments maintained

## Expected Output

### After PyInstaller Build
```
dist/
└── PakistanBankParser/
    ├── PakistanBankParser.exe
    └── _internal/
        ├── python310.dll
        ├── validate_mysql.py
        ├── validate_mysql.bat
        ├── db_manager.py
        ├── streamlit_app.py
        ├── model/
        │   ├── model.gguf
        │   └── mmproj.gguf
        ├── bin/
        │   └── llama-server.exe
        └── [all other dependencies]
```

### After Inno Setup Compilation
```
installer_output/
└── PakistanBankParser_Setup_v1.0.0.exe
```

## Installation Flow

1. Welcome message (mentions MySQL requirement)
2. License agreement
3. Installation directory selection
4. **MySQL Configuration** (NEW)
   - Host, Database, Username, Password
   - Validation attempt
5. **Bank Slips Folder** selection
6. File copying
7. Configuration file creation
8. Completion message with summary

## Testing Checklist

### Before Building
- [ ] All files present
- [ ] MySQL installed and running
- [ ] requirements.txt up to date

### After Building
- [ ] PakistanBankParser.exe created
- [ ] _internal folder contains all files
- [ ] validate_mysql.py and .bat present

### After Installer Creation
- [ ] Installer runs without errors
- [ ] MySQL config page displays correctly
- [ ] All 4 fields visible and accessible
- [ ] Validation works (if mysql.exe in PATH)
- [ ] Installation completes successfully
- [ ] Application launches and connects to MySQL

## Notes

- Client system needs MySQL Server installed and running
- mysql.exe must be in system PATH for validation (optional)
- If validation fails, installation continues with saved config
- User can manually edit config at %APPDATA%\PakistanBankParser\config\db_config.json
