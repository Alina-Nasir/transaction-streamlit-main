"""
Pakistani Bank Transaction Parser - Launcher
Main entry point for bundled executable
"""

import sys
import os
import subprocess
import time
import atexit
import multiprocessing
import logging
import traceback
import webbrowser
from datetime import datetime

# Global process references for cleanup
llama_process = None
batch_processor_process = None

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode characters on Windows"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Safely encode for console output
            if isinstance(msg, str):
                msg = msg.encode('ascii', 'replace').decode('ascii')
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Setup logging
def setup_logging():
    """Setup comprehensive logging to file in AppData"""
    try:
        # Get AppData directory
        appdata = os.getenv('APPDATA')
        log_dir = os.path.join(appdata, 'PakistanBankParser', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Different log file for batch processor mode
        if '--batch-processor' in sys.argv:
            log_file = os.path.join(log_dir, f'batch_processor_{timestamp}.log')
        else:
            log_file = os.path.join(log_dir, f'launcher_{timestamp}.log')
        
        # Configure logging with file handler (always UTF-8)
        # Use SafeStreamHandler for console to avoid Unicode errors
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info(f"Log file created: {log_file}")
        logging.info("="*60)
        if '--batch-processor' in sys.argv:
            logging.info("BATCH PROCESSOR MODE - Launcher Started")
        else:
            logging.info("Pakistani Bank Transaction Parser - Launcher Started")
        logging.info("="*60)
        logging.info(f"Python version: {sys.version}")
        logging.info(f"Platform: {sys.platform}")
        logging.info(f"Frozen mode: {getattr(sys, 'frozen', False)}")
        logging.info(f"Command line args: {sys.argv}")
        if hasattr(sys, '_MEIPASS'):
            logging.info(f"PyInstaller temp dir: {sys._MEIPASS}")
        
        return log_file
    except Exception as e:
        # Even if logging fails, write to a basic file
        try:
            appdata = os.getenv('APPDATA')
            emergency_log = os.path.join(appdata, 'PakistanBankParser', 'logs', 'emergency.log')
            os.makedirs(os.path.dirname(emergency_log), exist_ok=True)
            with open(emergency_log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now()}: Failed to setup logging: {e}\n")
                import traceback
                f.write(traceback.format_exc() + "\n")
        except:
            pass
        return None

# Initialize logging immediately
log_file = setup_logging()


def resource_path(relative_path):
    """
    Get absolute path to resource - works for dev and PyInstaller bundled mode
    
    Args:
        relative_path: Relative path to resource
    
    Returns:
        Absolute path to resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logging.debug(f"Using PyInstaller temp path: {base_path}")
    except Exception:
        # Running in development mode
        base_path = os.path.abspath(".")
        logging.debug(f"Using development path: {base_path}")
    
    full_path = os.path.join(base_path, relative_path)
    logging.debug(f"Resource path for '{relative_path}': {full_path}")
    return full_path


def get_cpu_count():
    """Get number of CPU cores"""
    try:
        return str(multiprocessing.cpu_count())
    except:
        return "4"  # Fallback to 4 cores


def start_llama_server():
    """
    Start llama.cpp server as subprocess
    Replicates the functionality of start_llama_server.bat
    """
    global llama_process
    
    try:
        logging.info("Starting llama.cpp server initialization")
        
        # Get paths to bundled resources
        llama_exe = resource_path(os.path.join("bin", "llama-server.exe"))
        model_path = resource_path(os.path.join("model", "model.gguf"))
        mmproj_path = resource_path(os.path.join("model", "mmproj.gguf"))
        bin_dir = resource_path("bin")
        
        logging.info(f"llama-server.exe path: {llama_exe}")
        logging.info(f"model.gguf path: {model_path}")
        logging.info(f"mmproj.gguf path: {mmproj_path}")
        logging.info(f"bin directory: {bin_dir}")
        
        # Verify files exist
        if not os.path.exists(llama_exe):
            error_msg = f"llama-server.exe not found at: {llama_exe}"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return False
        
        if not os.path.exists(model_path):
            error_msg = f"model.gguf not found at: {model_path}"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return False
        
        if not os.path.exists(mmproj_path):
            error_msg = f"mmproj.gguf not found at: {mmproj_path}"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return False
        
        logging.info("All required files verified")
        
        # Check for essential DLLs
        essential_dlls = ["ggml.dll", "llama.dll", "libomp140.x86_64.dll"]
        missing_dlls = []
        for dll in essential_dlls:
            dll_path = os.path.join(bin_dir, dll)
            logging.debug(f"Checking DLL: {dll_path}")
            if not os.path.exists(dll_path):
                missing_dlls.append(dll)
                logging.error(f"Missing DLL: {dll}")
        
        if missing_dlls:
            error_msg = f"Missing DLLs: {', '.join(missing_dlls)}"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            print(f"   Expected location: {bin_dir}")
            return False
        
        logging.info("All essential DLLs verified")
        print("✅ llama-server.exe found")
        print("✅ model.gguf found")
        print("✅ mmproj.gguf found")
        print("✅ Essential DLLs verified")
        print("\n🚀 Starting llama.cpp server...")
        
        cpu_count = get_cpu_count()
        logging.info(f"CPU count: {cpu_count}")
        
        # Build command with optimized flags (from start_llama_server.bat)
        command = [
            llama_exe,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "--host", "0.0.0.0",
            "--port", "8088",
            "-c", "8192",
            "-t", cpu_count,
            "-tb", cpu_count,
            "-b", "2048",
            "-ub", "512",
            "-n", "512",
            "--n-gpu-layers", "0",
            "-np", "1",
            "--mlock"
        ]
        
        # Start server as subprocess with independent console
        # CREATE_NEW_CONSOLE creates new console window for server output
        # CREATE_NEW_PROCESS_GROUP makes it independent from parent process
        # See: https://github.com/pyinstaller/pyinstaller/issues/7118
        if sys.platform == 'win32':
            # Windows: Create new console in separate process group
            # Note: DETACHED_PROCESS conflicts with CREATE_NEW_CONSOLE, don't use both
            creation_flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
            logging.info(f"Using Windows creation flags: CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP")
        else:
            creation_flags = 0
            logging.info("Non-Windows platform, no special flags")
        
        logging.info(f"Command: {' '.join(command)}")
        logging.info(f"Working directory: {resource_path('bin')}")
        
        # Don't pipe output - let it go to the console window
        # Piping creates parent-child dependency
        llama_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            cwd=resource_path("bin"),  # Set working directory to bin folder
            creationflags=creation_flags  # Critical: creates independent process
        )
        
        logging.info(f"llama-server process created with PID: {llama_process.pid}")
        
        print(f"✅ llama.cpp server started (PID: {llama_process.pid})")
        print(f"📁 Log file: {log_file}")
        print("⏳ Waiting for server to initialize (15 seconds)...")
        print("📋 Server is running in separate console window")
        print("-" * 60)
        
        # Monitor the process for 15 seconds to catch early crashes
        start_time = time.time()
        check_interval = 1.0
        
        while time.time() - start_time < 15:
            # Check if process crashed
            poll_result = llama_process.poll()
            if poll_result is not None:
                error_msg = f"llama.cpp server crashed with exit code: {poll_result}"
                logging.error(error_msg)
                print("-" * 60)
                print(f"❌ ERROR: {error_msg}")
                print("\n🔍 Troubleshooting:")
                print("1. Check if all DLL files are in the bin folder")
                print("2. Verify model files are not corrupted")
                print("3. Check if port 8088 is already in use")
                print("4. Install Visual C++ Redistributable 2015-2022 (x64)")
                print(f"5. Check log file for details: {log_file}")
                print("\nPress Enter to exit...")
                input()
                return False
            
            elapsed = time.time() - start_time
            logging.debug(f"Server check {elapsed:.1f}s - Process still alive (PID: {llama_process.pid})")
            time.sleep(check_interval)
        
        print("-" * 60)
        
        # Final check if process is still running
        final_poll = llama_process.poll()
        if final_poll is not None:
            error_msg = f"llama.cpp server stopped unexpectedly (exit code: {final_poll})"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            print(f"Check log file: {log_file}")
            print("Press Enter to exit...")
            input()
            return False
        
        logging.info(f"Server successfully started and stable (PID: {llama_process.pid})")
        print("✅ Server ready on http://localhost:8088")
        print("ℹ️  Server is running independently in separate console")
        print(f"ℹ️  All events logged to: {log_file}")
        return True
        
    except Exception as e:
        error_msg = f"Exception starting llama server: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        print(f"❌ ERROR: {error_msg}")
        print(f"Check log file: {log_file}")
        return False


def cleanup_llama_server():
    """Kill llama server process on exit"""
    global llama_process
    if llama_process is not None:
        try:
            logging.info("Initiating server shutdown")
            print("\n🛑 Shutting down llama.cpp server...")
            llama_process.terminate()
            llama_process.wait(timeout=5)
            logging.info("Server terminated gracefully")
            print("✅ Server stopped")
        except subprocess.TimeoutExpired:
            # Force kill if termination takes too long
            logging.warning("Server termination timeout, forcing kill")
            try:
                llama_process.kill()
                logging.info("Server forcefully killed")
                print("✅ Server forcefully stopped")
            except Exception as kill_error:
                logging.error(f"Failed to kill server: {kill_error}")
        except Exception as e:
            # Try force kill on any other error
            logging.error(f"Error during cleanup: {e}")
            try:
                llama_process.kill()
                logging.info("Server killed after error")
            except Exception as kill_error:
                logging.error(f"Failed to kill server after error: {kill_error}")


def start_batch_processor():
    """Start batch processor as background subprocess if enabled"""
    global batch_processor_process
    
    try:
        # Try to import batch modules
        try:
            import batch_config
        except ImportError:
            logging.warning("batch_config module not available - batch processing disabled")
            return False
        
        # Load configuration
        config = batch_config.load_config()
        
        if not config.get('enabled', False):
            logging.info("Batch processing disabled in configuration")
            return False
        
        logging.info("Starting batch processor service")
        
        # Get path to batch processor script
        batch_script = resource_path("batch_processor_start.py")
        logging.info(f"Batch processor script: {batch_script}")
        
        if not os.path.exists(batch_script):
            logging.warning(f"Batch processor script not found: {batch_script}")
            return False
        
        # Start batch processor as subprocess
        # It will run independently even if Streamlit closes
        batch_log_file = os.path.join(
            os.path.dirname(log_file),
            f'batch_startup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        try:
            batch_stdout = open(batch_log_file, 'w', encoding='utf-8')
            batch_stderr = open(batch_log_file, 'a', encoding='utf-8')
            
            # Setup environment for subprocess
            env = os.environ.copy()
            
            # Get the directory where batch_script is located
            batch_dir = os.path.dirname(batch_script)
            
            # Add batch directory to PYTHONPATH so imports work correctly
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{batch_dir}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = batch_dir
            
            # Set UTF-8 encoding
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # CRITICAL: Set PYINSTALLER_RESET_ENVIRONMENT to 1
            # This tells PyInstaller to spawn the subprocess as an independent instance
            # Otherwise the subprocess reuses the parent's activation context and fails
            env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
            
            logging.info(f"Batch subprocess PYTHONPATH: {env.get('PYTHONPATH')}")
            logging.info(f"Batch subprocess PYINSTALLER_RESET_ENVIRONMENT: {env.get('PYINSTALLER_RESET_ENVIRONMENT')}")
            logging.info(f"Batch subprocess script: {batch_script}")
            logging.info(f"Batch subprocess executable: {sys.executable}")
            
            # In bundled mode, we need to call ourselves with --batch-processor argument
            # because PyInstaller bundles always run the entry point script
            batch_processor_process = subprocess.Popen(
                [sys.executable, '--batch-processor'],
                stdin=subprocess.DEVNULL,
                stdout=batch_stdout,
                stderr=batch_stderr,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
                env=env
            )
            
            logging.info(f"Batch processor started with PID: {batch_processor_process.pid}")
            logging.info(f"Batch processor logs: {batch_log_file}")
            print(f"✅ Batch processor service started (PID: {batch_processor_process.pid})")
            
            # Wait a moment and check if process is still alive
            time.sleep(2)
            if batch_processor_process.poll() is not None:
                # Process exited immediately - something went wrong
                logging.error(f"Batch processor exited immediately with code: {batch_processor_process.returncode}")
                logging.error(f"Check logs: {batch_log_file}")
                batch_stdout.close()
                batch_stderr.close()
                with open(batch_log_file, 'r', encoding='utf-8') as f:
                    error_content = f.read()
                    logging.error(f"Batch processor error output:\n{error_content}")
                return False
            
            return True
        
        except Exception as e:
            logging.error(f"Exception starting batch processor subprocess: {e}")
            logging.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logging.warning(f"Failed to start batch processor: {e}")
        return False


def cleanup_batch_processor():
    """Stop batch processor process on exit"""
    global batch_processor_process
    if batch_processor_process is not None:
        try:
            logging.info("Stopping batch processor service")
            print("\n🛑 Shutting down batch processor...")
            batch_processor_process.terminate()
            batch_processor_process.wait(timeout=5)
            logging.info("Batch processor terminated gracefully")
            print("✅ Batch processor stopped")
        except subprocess.TimeoutExpired:
            logging.warning("Batch processor termination timeout, forcing kill")
            try:
                batch_processor_process.kill()
                logging.info("Batch processor forcefully killed")
                print("✅ Batch processor forcefully stopped")
            except Exception as e:
                logging.error(f"Failed to kill batch processor: {e}")
        except Exception as e:
            logging.error(f"Error stopping batch processor: {e}")


def start_streamlit_app():
    """
    Start Streamlit application
    Uses streamlit.web.cli to run in bundled mode
    """
    try:
        logging.info("Initializing Streamlit application")
        
        # Import database manager and initialize
        import db_manager
        logging.info("Importing database manager")
        db_manager.init_db()
        logging.info("Database initialized")
        
        # Get path to streamlit_app.py
        app_path = resource_path("streamlit_app.py")
        logging.info(f"Streamlit app path: {app_path}")
        
        if not os.path.exists(app_path):
            error_msg = f"streamlit_app.py not found at: {app_path}"
            logging.error(error_msg)
            print(f"❌ ERROR: {error_msg}")
            return False
        
        print(f"✅ Loading Streamlit app from: {app_path}")
        print("\n🌐 Starting Pakistani Bank Transaction Parser...")
        print("=" * 60)
        
        # Run Streamlit using its CLI module (works in bundled mode)
        from streamlit.web import cli as stcli
        
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--global.developmentMode=false",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none"
        ]
        
        logging.info(f"Starting Streamlit with args: {sys.argv}")
        
        # Open browser after a short delay (let Streamlit initialize)
        def open_browser():
            time.sleep(3)  # Wait 3 seconds for Streamlit to start
            logging.info("Opening browser at http://localhost:8501")
            webbrowser.open('http://localhost:8501')
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        logging.info("Browser auto-launch scheduled")
        
        logging.info("Transferring control to Streamlit CLI")
        
        sys.exit(stcli.main())
        
    except Exception as e:
        error_msg = f"Exception starting Streamlit: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        print(f"❌ ERROR: {error_msg}")
        print(f"Check log file: {log_file}")
        return False


def run_batch_processor_mode():
    """Run in batch processor mode (called when --batch-processor argument is passed)"""
    # This runs the batch processor directly without starting llama or streamlit
    logging.info("="*60)
    logging.info("BATCH PROCESSOR MODE - STARTING")
    logging.info("="*60)
    logging.info(f"sys.executable: {sys.executable}")
    logging.info(f"sys.argv: {sys.argv}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    logging.info(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'NOT SET')}")
    
    try:
        logging.info("Step 1: Importing batch_config module...")
        import batch_config
        logging.info("Step 1: SUCCESS - batch_config imported")
        
        logging.info("Step 2: Importing batch_processor_service module...")
        import batch_processor_service
        logging.info("Step 2: SUCCESS - batch_processor_service imported")
        
        logging.info("Step 3: Loading configuration...")
        config = batch_config.load_config()
        logging.info(f"Step 3: SUCCESS - Config loaded")
        logging.info(f"  - enabled: {config.get('enabled')}")
        logging.info(f"  - incoming_folder: {config.get('incoming_folder')}")
        logging.info(f"  - processed_folder: {config.get('processed_folder')}")
        logging.info(f"  - failed_folder: {config.get('failed_folder')}")
        logging.info(f"  - auto_move_processed: {config.get('auto_move_processed')}")
        logging.info(f"  - debounce_delay: {config.get('debounce_delay')}")
        
        # Verify folders exist
        incoming_folder = config.get('incoming_folder')
        if not incoming_folder:
            logging.error("Step 4: FAILED - No incoming folder configured!")
            return
        
        logging.info(f"Step 4: Creating incoming folder if needed: {incoming_folder}")
        os.makedirs(incoming_folder, exist_ok=True)
        logging.info(f"Step 4: SUCCESS - Folder exists: {os.path.exists(incoming_folder)}")
        logging.info(f"Step 4: Folder writable: {os.access(incoming_folder, os.W_OK)}")
        
        # List files currently in incoming folder
        try:
            files = os.listdir(incoming_folder)
            logging.info(f"Step 5: Files currently in incoming folder: {files}")
        except Exception as e:
            logging.warning(f"Step 5: Could not list files: {e}")
        
        # Run batch processor
        logging.info("Step 6: Starting batch processor service...")
        logging.info("="*60)
        logging.info("BATCH PROCESSOR SERVICE RUNNING - Monitoring for files...")
        logging.info("="*60)
        batch_processor_service.run_batch_processor(config, run_in_foreground=False)
        logging.info("BATCH PROCESSOR SERVICE STOPPED")
        
    except ImportError as e:
        logging.error(f"IMPORT ERROR in batch processor mode: {e}")
        logging.error(f"Module search paths: {sys.path}")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"CRITICAL ERROR in batch processor mode: {e}")
        logging.error(traceback.format_exc())
        raise


def main():
    """Main entry point"""
    logging.info("Main function started")
    logging.info(f"Arguments received: {sys.argv}")
    
    # Check if running in batch processor mode - MUST BE FIRST!
    if '--batch-processor' in sys.argv:
        logging.info("="*60)
        logging.info("BATCH PROCESSOR MODE DETECTED")
        logging.info("="*60)
        run_batch_processor_mode()
        logging.info("Batch processor mode completed, exiting")
        return
    
    # Only print fancy UI for normal mode
    print("=" * 60)
    print("Pakistani Bank Transaction Parser")
    print("=" * 60)
    print()
    
    # Register cleanup functions
    atexit.register(cleanup_llama_server)
    atexit.register(cleanup_batch_processor)
    logging.info("Cleanup handlers registered")
    
    # Step 1: Start llama.cpp server
    logging.info("Attempting to start llama.cpp server")
    if not start_llama_server():
        logging.error("Failed to start llama.cpp server")
        print("\n❌ Failed to start llama.cpp server")
        print(f"📁 Check log file: {log_file}")
        print("Press Enter to exit...")
        input()
        sys.exit(1)
    
    logging.info("llama.cpp server started successfully")
    print("\n" + "=" * 60)
    
    # Step 1.5: Start batch processor (if enabled)
    logging.info("Attempting to start batch processor")
    if start_batch_processor():
        print("=" * 60)
    else:
        logging.info("Batch processor not started (disabled or unavailable)")
    
    # Step 2: Start Streamlit app
    try:
        logging.info("Attempting to start Streamlit application")
        start_streamlit_app()
    except KeyboardInterrupt:
        logging.info("Application interrupted by user (Ctrl+C)")
        print("\n\n⚠️  Application interrupted by user")
    except Exception as e:
        error_msg = f"Streamlit error: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        print(f"\n❌ {error_msg}")
        print(f"📁 Check log file: {log_file}")
        print("\nllama-server is still running. Check if there's a Streamlit issue.")
        print("Press Enter to exit...")
        input()
    finally:
        logging.info("Entering cleanup phase")
        cleanup_batch_processor()
        cleanup_llama_server()
        logging.info("Application shutdown complete")
        print("\n👋 Application closed")


if __name__ == "__main__":
    # Prevent multiple processes when frozen
    if getattr(sys, 'frozen', False):
        multiprocessing.freeze_support()
    
    main()
