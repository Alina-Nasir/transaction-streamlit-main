# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Get DLL files from llama-cpu directory
dll_files = []
bin_path = 'llama-cpu'
for file in os.listdir(bin_path):
    if file.endswith('.dll'):
        dll_files.append((os.path.join(bin_path, file), 'bin'))

# Collect metadata and data files for packages
datas_with_metadata = [
    ('streamlit_app.py', '.'),
    ('db_manager.py', '.'),
    ('models_latest/model.gguf', 'model'),
    ('models_latest/mmproj.gguf', 'model'),
]

# Add package metadata
for pkg in ['streamlit', 'altair', 'pandas', 'pillow']:
    try:
        datas_with_metadata += copy_metadata(pkg)
    except:
        pass

# CRITICAL: Collect Streamlit's static files (HTML, CSS, JS)
try:
    datas_with_metadata += collect_data_files('streamlit', include_py_files=False)
except Exception as e:
    print(f"Warning: Could not collect streamlit data files: {e}")

# Also collect tornado templates used by Streamlit
try:
    datas_with_metadata += collect_data_files('streamlit.web.server', include_py_files=False)
except:
    pass

# Add batch processing modules
datas_with_metadata += [
    ('inference_engine.py', '.'),
    ('batch_processor_service.py', '.'),
    ('batch_config.py', '.'),
    ('run_batch_service.py', '.'),
    ('batch_processor_start.py', '.'),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[
        ('llama-cpu/llama-server.exe', 'bin'),
    ] + dll_files,
    datas=datas_with_metadata,
    hiddenimports=[
        'streamlit',
        'streamlit.web',
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.scriptrunner.magic_funcs',
        'streamlit.runtime.scriptrunner.script_runner',
        'streamlit.runtime.scriptrunner.exec_code',
        'streamlit.runtime.state',
        'streamlit.runtime.state.session_state',
        'streamlit.runtime.caching',
        'streamlit.runtime.legacy_caching',
        'streamlit.runtime.media_file_manager',
        'streamlit.runtime.uploaded_file_manager',
        'streamlit.elements',
        'streamlit.elements.form',
        'streamlit.elements.widgets',
        'streamlit.components',
        'streamlit.components.v1',
        'streamlit.web.server',
        'streamlit.web.server.server',
        'streamlit.web.server.routes',
        'streamlit.logger',
        'pandas',
        'altair',
        'sqlite3',
        'PIL',
        'PIL.Image',
        'requests',
        'uuid',
        'base64',
        'json',
        're',
        'io',
        'tempfile',
        'hashlib',
        'datetime',
        'pypdfium2',
        'db_manager',
        'inference_engine',
        'batch_processor_service',
        'batch_config',
        'run_batch_service',
        'batch_processor_start',
        'tornado',
        'tornado.web',
        'tornado.ioloop',
        'tornado.httpserver',
        'tornado.template',
        'watchdog',
        'watchdog.observers',
        'watchdog.events',
        'watchdog.observers.polling',
        'validators',
        'packaging',
        'packaging.version',
        'pyarrow',
        'click',
        'toml',
        'typing_extensions',
        'importlib_metadata',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PakistanBankParser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PakistanBankParser',
)