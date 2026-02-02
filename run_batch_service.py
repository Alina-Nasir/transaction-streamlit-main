"""
Windows Service Wrapper for Batch Processor
Allows running batch processor as a Windows service using NSSM
(Non-Sucking Service Manager)

Usage:
  python run_batch_service.py [start|stop|restart|install|remove]
  
  install  - Install as Windows service (requires admin)
  remove   - Remove Windows service (requires admin)
  start    - Start the batch processor
  stop     - Stop the batch processor
  restart  - Restart the batch processor
  status   - Check if running
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import db_manager
import batch_config
import batch_processor_service

# Setup logging
log_dir = db_manager.get_log_dir()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'batch_service_wrapper.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Service configuration
SERVICE_NAME = "PakistanBankParserBatchService"
SERVICE_DISPLAY_NAME = "Pakistani Bank Parser - Batch Processor"
SERVICE_DESCRIPTION = "24/7 batch processing service for bank transaction slips"

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
SERVICE_SCRIPT = os.path.join(SCRIPT_DIR, "run_batch_service.py")


def check_admin():
    """Check if running with administrator privileges"""
    try:
        import ctypes
        return ctypes.windll.shell.IsUserAnAdmin()
    except:
        return False


def run_nssm_command(command):
    """Run NSSM command"""
    try:
        # Try using NSSM if available in PATH
        result = subprocess.run(
            ["nssm", command, SERVICE_NAME],
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        logger.error("NSSM not found. Install with: https://nssm.cc/download")
        return False, "", "NSSM not found"


def install_service():
    """Install batch processor as Windows service"""
    if not check_admin():
        logger.error("Service installation requires administrator privileges")
        print("ERROR: Administrator privileges required. Run as Administrator.")
        return False
    
    logger.info(f"Installing {SERVICE_DISPLAY_NAME}...")
    
    # Check if NSSM is available
    try:
        subprocess.run(["nssm", "version"], capture_output=True, timeout=5, check=True)
    except Exception as e:
        logger.error(f"NSSM not available: {e}")
        print("\nERROR: NSSM (Non-Sucking Service Manager) is required.")
        print("Download from: https://nssm.cc/download")
        print("Extract nssm.exe to C:\\Windows\\System32\\ or add to PATH")
        return False
    
    try:
        # Install service
        cmd = [
            "nssm", "install", SERVICE_NAME,
            PYTHON_EXE, SERVICE_SCRIPT, "run_foreground"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Service installed: {result.stdout}")
        
        # Set service description
        subprocess.run([
            "nssm", "set", SERVICE_NAME, "Description",
            SERVICE_DESCRIPTION
        ], capture_output=True, check=True)
        
        # Set to auto-start
        subprocess.run([
            "nssm", "set", SERVICE_NAME, "Start", "SERVICE_AUTO_START"
        ], capture_output=True, check=True)
        
        # Set to restart on failure
        subprocess.run([
            "nssm", "set", SERVICE_NAME, "AppExit", "Default", "Restart"
        ], capture_output=True, check=True)
        
        # Set output files for logging
        log_file = os.path.join(log_dir, "batch_service_nssm.log")
        subprocess.run([
            "nssm", "set", SERVICE_NAME, "AppStdout", log_file
        ], capture_output=True, check=True)
        subprocess.run([
            "nssm", "set", SERVICE_NAME, "AppStderr", log_file
        ], capture_output=True, check=True)
        
        print(f"\n✅ Service installed successfully!")
        print(f"   Service Name: {SERVICE_NAME}")
        print(f"   Start: net start {SERVICE_NAME}")
        print(f"   Stop:  net stop {SERVICE_NAME}")
        print(f"   Remove: nssm remove {SERVICE_NAME} confirm")
        
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install service: {e.stderr}")
        print(f"ERROR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"ERROR: {e}")
        return False


def remove_service():
    """Remove batch processor Windows service"""
    if not check_admin():
        logger.error("Service removal requires administrator privileges")
        print("ERROR: Administrator privileges required. Run as Administrator.")
        return False
    
    logger.info(f"Removing {SERVICE_DISPLAY_NAME}...")
    
    try:
        result = subprocess.run(
            ["nssm", "remove", SERVICE_NAME, "confirm"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Service removed: {result.stdout}")
        print(f"✅ Service removed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to remove service: {e.stderr}")
        print(f"ERROR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"ERROR: {e}")
        return False


def start_service():
    """Start the batch processor service"""
    logger.info("Starting batch processor service...")
    
    try:
        result = subprocess.run(
            ["net", "start", SERVICE_NAME],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or "already been started" in result.stdout:
            print(f"✅ Service started")
            logger.info("Service started successfully")
            return True
        else:
            print(f"ERROR: {result.stderr}")
            logger.error(f"Failed to start service: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        print(f"ERROR: {e}")
        return False


def stop_service():
    """Stop the batch processor service"""
    logger.info("Stopping batch processor service...")
    
    try:
        result = subprocess.run(
            ["net", "stop", SERVICE_NAME],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or "has not been started" in result.stdout:
            print(f"✅ Service stopped")
            logger.info("Service stopped successfully")
            return True
        else:
            print(f"ERROR: {result.stderr}")
            logger.error(f"Failed to stop service: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error stopping service: {e}")
        print(f"ERROR: {e}")
        return False


def restart_service():
    """Restart the batch processor service"""
    logger.info("Restarting batch processor service...")
    stop_service()
    time.sleep(2)
    start_service()


def check_status():
    """Check if service is running"""
    try:
        result = subprocess.run(
            ["nssm", "status", SERVICE_NAME],
            capture_output=True,
            text=True
        )
        
        if "SERVICE_RUNNING" in result.stdout:
            print(f"✅ Service is RUNNING")
            return True
        elif "SERVICE_STOPPED" in result.stdout:
            print(f"⚫ Service is STOPPED")
            return False
        else:
            print(f"❓ Service status unknown: {result.stdout}")
            return False
    
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        print(f"ERROR: {e}")
        return False


def run_foreground():
    """Run batch processor in foreground (used by service)"""
    logger.info("="*60)
    logger.info("BATCH PROCESSOR SERVICE STARTING")
    logger.info("="*60)
    
    try:
        # Load configuration
        logger.info("Loading batch processor configuration...")
        config = batch_config.load_config()
        
        logger.info(f"Incoming folder: {config.get('incoming_folder')}")
        logger.info(f"Enabled: {config.get('enabled')}")
        logger.info(f"Auto-move: {config.get('auto_move_processed')}")
        
        # Verify folders exist
        incoming_folder = config.get('incoming_folder')
        if not incoming_folder:
            raise ValueError("No incoming folder configured!")
        
        os.makedirs(incoming_folder, exist_ok=True)
        logger.info(f"✅ Verified incoming folder: {incoming_folder}")
        
        # Run processor
        logger.info("Starting batch processor service...")
        batch_processor_service.run_batch_processor(config, run_in_foreground=False)
        logger.info("Batch processor service stopped")
    
    except Exception as e:
        logger.error(f"CRITICAL ERROR in batch processor: {e}", exc_info=True)
        logger.error("="*60)
        raise


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        command = "run_foreground"  # Default to running processor
    else:
        command = sys.argv[1].lower()
    
    logger.info(f"Command: {command}")
    
    if command == "install":
        success = install_service()
        sys.exit(0 if success else 1)
    
    elif command == "remove":
        success = remove_service()
        sys.exit(0 if success else 1)
    
    elif command == "start":
        success = start_service()
        sys.exit(0 if success else 1)
    
    elif command == "stop":
        success = stop_service()
        sys.exit(0 if success else 1)
    
    elif command == "restart":
        restart_service()
        sys.exit(0)
    
    elif command == "status":
        check_status()
        sys.exit(0)
    
    elif command in ["run_foreground", "run"]:
        # Run batch processor directly
        run_foreground()
    
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
