"""
Batch Processing Service for Pakistani Bank Transaction Parser
Monitors a folder 24/7 for new receipt images and automatically runs inference
Saves results to the same SQLite database used by Streamlit app
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import db_manager
import inference_engine

# Configure logging
def setup_logging():
    """Setup logging to file with timestamps"""
    log_dir = db_manager.get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"batch_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


class TransactionFileHandler(FileSystemEventHandler):
    """Handles file system events in the incoming folder"""
    
    # Debounce tracking to prevent double-processing
    file_modified_times = {}
    DEBOUNCE_DELAY = 3.0  # seconds
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.failed_count = 0
        self.success_count = 0
        logger.info(f"TransactionFileHandler initialized with config: {config}")
    
    def on_created(self, event):
        """Called when a file is created"""
        if event.is_directory:
            logger.debug(f"Directory created (ignored): {event.src_path}")
            return
        
        logger.debug(f"File created event: {event.src_path}")
        
        # Only process image and PDF files
        if not self._is_supported_file(event.src_path):
            logger.debug(f"File ignored (not supported format): {event.src_path}")
            return
        
        logger.info(f"🎯 SUPPORTED FILE DETECTED: {event.src_path}")
        
        # Debounce: wait for file write to complete
        file_key = event.src_path
        current_time = time.time()
        
        if file_key in self.file_modified_times:
            elapsed = current_time - self.file_modified_times[file_key]
            if elapsed < self.DEBOUNCE_DELAY:
                logger.debug(f"Debouncing: {file_key} (elapsed: {elapsed:.1f}s)")
                return
        
        self.file_modified_times[file_key] = current_time
        
        # Schedule processing after debounce delay
        logger.debug(f"Scheduling processing for {event.src_path} after {self.DEBOUNCE_DELAY}s")
        time.sleep(self.DEBOUNCE_DELAY)
        
        self._process_file(event.src_path)
    
    def on_modified(self, event):
        """Called when a file is modified - also trigger processing"""
        if event.is_directory:
            return
        
        if not self._is_supported_file(event.src_path):
            return
        
        # Debounce check
        file_key = event.src_path
        current_time = time.time()
        
        if file_key in self.file_modified_times:
            elapsed = current_time - self.file_modified_times[file_key]
            if elapsed < self.DEBOUNCE_DELAY:
                return
        
        self.file_modified_times[file_key] = current_time
        
        logger.debug(f"Scheduling processing for modified file {event.src_path}")
        time.sleep(self.DEBOUNCE_DELAY)
        
        self._process_file(event.src_path)
    
    def _is_supported_file(self, file_path):
        """Check if file is a supported image or PDF"""
        supported_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.JPG', '.JPEG', '.PNG', '.PDF')
        return file_path.lower().endswith(supported_extensions)
    
    def _process_file(self, file_path):
        """Process a single transaction file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {file_path}")
                return
            
            # Wait for file to be fully written (size stable check)
            if not self._wait_for_file_ready(file_path):
                logger.error(f"File never became ready: {file_path}")
                self.failed_count += 1
                self._move_to_failed(file_path, "File never became ready after retries")
                return
            
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file: {file_name}")
            
            # Run inference
            logger.debug(f"Running inference on {file_name}")
            response = inference_engine.call_local_model_with_image(
                file_path,
                timeout=300
            )
            
            if response is None:
                logger.error(f"Inference failed for {file_name} (model returned None)")
                self.failed_count += 1
                self._move_to_failed(file_path, "Inference failed (model returned None)")
                return
            
            # Extract JSON from response
            logger.debug(f"Extracting JSON from response")
            transaction_data = inference_engine.extract_json_from_response(response)
            
            if not transaction_data or all(v == "Not Found" for v in transaction_data.values()):
                logger.warning(f"No valid transaction data extracted from {file_name}")
                self.failed_count += 1
                self._move_to_failed(file_path, "No valid transaction data extracted")
                return
            
            # Add metadata
            transaction_data['FileName'] = file_name
            
            # Save to database
            logger.debug(f"Saving transaction to database")
            db_manager.insert_record(transaction_data)
            
            logger.info(f"Successfully processed: {file_name}")
            logger.debug(f"Extracted data: {transaction_data}")
            self.success_count += 1
            
            # Move to processed folder if configured
            if self.config.get('auto_move_processed', True):
                self._move_to_processed(file_path)
            else:
                logger.debug(f"Auto-move disabled, keeping file in place: {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            self.failed_count += 1
            try:
                self._move_to_failed(file_path, f"Processing error: {str(e)}")
            except Exception as move_error:
                logger.error(f"Failed to move file to failed folder: {move_error}")
    
    def _wait_for_file_ready(self, file_path, max_retries=5):
        """Wait for file to be fully written (size stable)"""
        try:
            previous_size = -1
            stable_count = 0
            
            for attempt in range(max_retries):
                try:
                    current_size = os.path.getsize(file_path)
                    
                    if current_size == previous_size and current_size > 0:
                        stable_count += 1
                        if stable_count >= 2:
                            logger.debug(f"File is ready: {file_path} (size: {current_size} bytes)")
                            return True
                    else:
                        stable_count = 0
                    
                    previous_size = current_size
                    time.sleep(0.5)
                
                except OSError:
                    logger.debug(f"File not accessible yet (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.5)
            
            logger.warning(f"File readiness check timed out: {file_path}")
            return False
        
        except Exception as e:
            logger.error(f"Error checking file readiness: {e}")
            return False
    
    def _move_to_processed(self, file_path):
        """Move successfully processed file to processed folder"""
        try:
            processed_dir = self.config.get('processed_folder', None)
            if not processed_dir:
                logger.debug("Processed folder not configured, file stays in place")
                return
            
            os.makedirs(processed_dir, exist_ok=True)
            
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(processed_dir, file_name)
            
            # Handle filename conflict
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(file_name)
                dest_path = os.path.join(processed_dir, f"{name}_{int(time.time())}{ext}")
            
            os.rename(file_path, dest_path)
            logger.info(f"Moved to processed: {dest_path}")
        
        except Exception as e:
            logger.error(f"Error moving file to processed folder: {e}")
    
    def _move_to_failed(self, file_path, reason):
        """Move failed file to failed folder with error info"""
        try:
            failed_dir = self.config.get('failed_folder', None)
            if not failed_dir:
                logger.debug("Failed folder not configured")
                return
            
            os.makedirs(failed_dir, exist_ok=True)
            
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(failed_dir, file_name)
            
            # Handle filename conflict
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(file_name)
                dest_path = os.path.join(failed_dir, f"{name}_{int(time.time())}{ext}")
            
            os.rename(file_path, dest_path)
            
            # Create error report file
            report_path = dest_path + ".error"
            with open(report_path, 'w') as f:
                f.write(f"Processing Error: {reason}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            logger.info(f"Moved to failed: {dest_path}")
        
        except Exception as e:
            logger.error(f"Error moving file to failed folder: {e}")
    
    def get_stats(self):
        """Return processing statistics"""
        return {
            'success': self.success_count,
            'failed': self.failed_count,
            'total': self.success_count + self.failed_count
        }


class BatchProcessorService:
    """Main batch processor service"""
    
    def __init__(self, config):
        self.config = config
        self.observer = None
        self.event_handler = None
        logger.info("BatchProcessorService initialized")
    
    def start(self):
        """Start monitoring the incoming folder"""
        try:
            incoming_dir = self.config.get('incoming_folder')
            
            if not incoming_dir:
                logger.error("❌ No incoming folder configured")
                return False
            
            # Create incoming folder if it doesn't exist
            os.makedirs(incoming_dir, exist_ok=True)
            logger.info(f"✅ Monitoring folder: {incoming_dir}")
            logger.info(f"   Folder exists: {os.path.exists(incoming_dir)}")
            logger.info(f"   Folder writable: {os.access(incoming_dir, os.W_OK)}")
            
            # Initialize event handler
            self.event_handler = TransactionFileHandler(self.config)
            logger.info("✅ Event handler initialized")
            
            # Create and start observer
            self.observer = Observer()
            self.observer.schedule(self.event_handler, incoming_dir, recursive=False)
            self.observer.start()
            logger.info("✅ File system observer started")
            
            logger.info("✅ Batch processor service started successfully - monitoring for files...")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error starting batch processor: {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop monitoring"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
                logger.info("Batch processor service stopped")
        except Exception as e:
            logger.error(f"Error stopping batch processor: {e}", exc_info=True)
    
    def is_running(self):
        """Check if observer is running"""
        return self.observer is not None and self.observer.is_alive()
    
    def get_stats(self):
        """Get processing statistics"""
        if self.event_handler:
            return self.event_handler.get_stats()
        return {'success': 0, 'failed': 0, 'total': 0}


def run_batch_processor(config, run_in_foreground=False):
    """
    Main entry point for batch processor
    
    Args:
        config: Configuration dictionary with keys:
            - incoming_folder: folder to monitor
            - processed_folder: where to move successful files
            - failed_folder: where to move failed files
            - auto_move_processed: whether to move successfully processed files
        run_in_foreground: If True, runs in foreground (for testing/debugging)
                          If False, runs as background daemon
    """
    service = BatchProcessorService(config)
    
    if not service.start():
        logger.error("Failed to start batch processor service")
        return False
    
    if run_in_foreground:
        try:
            logger.info("Running batch processor in foreground mode (Ctrl+C to stop)")
            while service.is_running():
                time.sleep(10)
                stats = service.get_stats()
                logger.info(f"Processing stats - Success: {stats['success']}, Failed: {stats['failed']}")
        except KeyboardInterrupt:
            logger.info("Batch processor interrupted by user")
        finally:
            service.stop()
    else:
        # Background mode - keep running
        try:
            logger.info("Batch processor running in background mode")
            while service.is_running():
                time.sleep(60)
                stats = service.get_stats()
                logger.info(f"Processing stats - Success: {stats['success']}, Failed: {stats['failed']}")
        except KeyboardInterrupt:
            logger.info("Batch processor interrupted")
        except Exception as e:
            logger.error(f"Error in batch processor loop: {e}", exc_info=True)
        finally:
            service.stop()


if __name__ == "__main__":
    # Default configuration
    default_config = {
        'incoming_folder': os.path.join(
            os.environ.get('USERPROFILE', os.path.expanduser('~')),
            'Pictures', 'BankSlips', 'incoming'
        ),
        'processed_folder': os.path.join(
            os.environ.get('USERPROFILE', os.path.expanduser('~')),
            'Pictures', 'BankSlips', 'processed'
        ),
        'failed_folder': os.path.join(
            os.environ.get('USERPROFILE', os.path.expanduser('~')),
            'Pictures', 'BankSlips', 'failed'
        ),
        'auto_move_processed': True
    }
    
    # Run batch processor
    run_batch_processor(default_config, run_in_foreground=True)
