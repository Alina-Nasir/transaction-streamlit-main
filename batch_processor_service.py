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
import queue
import threading
import json
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
    
    # Configuration constants
    DEBOUNCE_DELAY = 3.0  # seconds
    MAX_DEBOUNCE_ENTRIES = 1000  # Limit debounce dictionary size
    NUM_WORKER_THREADS = 3  # Number of concurrent processing threads
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Thread-safe work queue
        self.work_queue = queue.Queue()
        
        # Thread-safe tracking structures
        self.lock = threading.Lock()  # General lock for shared data
        self.stats_lock = threading.Lock()  # Lock for statistics
        
        # Debounce tracking (protected by self.lock)
        self.file_modified_times = {}
        
        # Pre-existing files tracking (protected by self.lock)
        self.existing_files = set()
        
        # Statistics (protected by self.stats_lock)
        self.failed_count = 0
        self.success_count = 0
        
        # Failed files tracking
        self.failed_files_log = self._get_failed_files_log_path()
        
        # Processed files tracking (for restart recovery)
        self.processed_files_log = self._get_processed_files_log_path()
        self.processed_files = self._load_processed_files()
        
        # Worker threads
        self.workers = []
        self.shutdown_event = threading.Event()
        
        # Periodic cleanup timer (runs every 24 hours)
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 24 * 60 * 60  # 24 hours in seconds
        
        # Start worker threads
        for i in range(self.NUM_WORKER_THREADS):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"TransactionFileHandler initialized with {self.NUM_WORKER_THREADS} worker threads")
        logger.info(f"Failed files log: {self.failed_files_log}")
        logger.info(f"Processed files log: {self.processed_files_log} (auto-cleanup every 24h)")
        logger.info(f"Loaded {len(self.processed_files)} previously processed files from log")
    
    def _get_processed_files_log_path(self):
        """Get path for processed files tracking JSON"""
        log_dir = db_manager.get_log_dir()
        return os.path.join(log_dir, 'processed_files.json')
    
    def _load_processed_files(self):
        """Load set of previously processed filenames and clean old entries (>30 days)"""
        try:
            if os.path.exists(self.processed_files_log):
                with open(self.processed_files_log, 'r') as f:
                    data = json.load(f)
                
                # Filter: keep only entries from last 30 days
                cutoff_time = datetime.now().timestamp() - (30 * 24 * 60 * 60)
                recent_entries = []
                
                for entry in data:
                    try:
                        # Parse timestamp
                        processed_at = datetime.fromisoformat(entry['processed_at']).timestamp()
                        if processed_at > cutoff_time:
                            recent_entries.append(entry)
                    except:
                        # If timestamp parsing fails, keep the entry (safer)
                        recent_entries.append(entry)
                
                # If we filtered old entries, save the cleaned version
                if len(recent_entries) < len(data):
                    removed_count = len(data) - len(recent_entries)
                    logger.info(f"🧹 Cleaned {removed_count} old entries (>30 days) from processed files log")
                    with open(self.processed_files_log, 'w') as f:
                        json.dump(recent_entries, f, indent=2)
                
                # Return set of filenames
                processed = set(entry['filename'] for entry in recent_entries)
                logger.info(f"📂 Loaded {len(processed)} processed files from log (last 30 days)")
                return processed
            else:
                logger.info(f"📂 No processed files log found - starting fresh")
                return set()
        except Exception as e:
            logger.error(f"Error loading processed files log: {e}")
            return set()
    
    def _mark_file_processed(self, file_path):
        """Mark a file as successfully processed (thread-safe)"""
        try:
            filename = os.path.basename(file_path)
            
            with self.lock:
                # Add to in-memory set
                self.processed_files.add(filename)
            
            # Periodic cleanup check (every 24 hours)
            current_time = time.time()
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                logger.info("⏰ 24 hours elapsed - triggering cleanup of old processed files")
                self._cleanup_old_processed_files()
                self.last_cleanup_time = current_time
            
            # CRITICAL: Lock the entire JSON read-modify-write operation
            # to prevent race conditions between worker threads
            with self.lock:
                # Load existing log
                if os.path.exists(self.processed_files_log):
                    with open(self.processed_files_log, 'r') as f:
                        processed_log = json.load(f)
                else:
                    processed_log = []
                
                # Add new entry
                processed_log.append({
                    'filename': filename,
                    'processed_at': datetime.now().isoformat(),
                    'full_path': file_path
                })
                
                # Write back atomically
                with open(self.processed_files_log, 'w') as f:
                    json.dump(processed_log, f, indent=2)
            
            logger.debug(f"✅ Marked as processed: {filename}")
        except Exception as e:
            logger.error(f"Error marking file as processed: {e}")
            # Even if JSON write fails, file is still in in-memory set
            # This prevents loss - file won't be reprocessed in current session
    
    def _is_already_processed(self, file_path):
        """Check if file was already processed (thread-safe)"""
        filename = os.path.basename(file_path)
        with self.lock:
            return filename in self.processed_files
    
    def _cleanup_old_processed_files(self):
        """Remove entries older than 30 days from processed files log"""
        try:
            if not os.path.exists(self.processed_files_log):
                return
            
            with open(self.processed_files_log, 'r') as f:
                data = json.load(f)
            
            # Filter: keep only entries from last 30 days
            cutoff_time = datetime.now().timestamp() - (30 * 24 * 60 * 60)
            recent_entries = []
            
            for entry in data:
                try:
                    processed_at = datetime.fromisoformat(entry['processed_at']).timestamp()
                    if processed_at > cutoff_time:
                        recent_entries.append(entry)
                except:
                    recent_entries.append(entry)  # Keep if timestamp parsing fails
            
            removed_count = len(data) - len(recent_entries)
            if removed_count > 0:
                logger.info(f"🧹 Cleanup: Removed {removed_count} old entries (>30 days), kept {len(recent_entries)}")
                
                # Update in-memory set
                with self.lock:
                    self.processed_files = set(entry['filename'] for entry in recent_entries)
                
                # Save cleaned version
                with open(self.processed_files_log, 'w') as f:
                    json.dump(recent_entries, f, indent=2)
            else:
                logger.info(f"🧹 Cleanup: No old entries to remove (all {len(data)} entries within 30 days)")
        
        except Exception as e:
            logger.error(f"Error during cleanup of processed files: {e}")
    
    def _get_failed_files_log_path(self):
        """Get path for failed files tracking JSON"""
        log_dir = db_manager.get_log_dir()
        return os.path.join(log_dir, 'failed_files.json')
    
    def _log_failed_file(self, file_path, reason):
        """Log a failed file with timestamp and reason"""
        try:
            # Load existing failed files
            if os.path.exists(self.failed_files_log):
                with open(self.failed_files_log, 'r') as f:
                    failed_files = json.load(f)
            else:
                failed_files = []
            
            # Add new entry
            failed_files.append({
                'file': file_path,
                'timestamp': datetime.now().isoformat(),
                'reason': reason
            })
            
            # Write back
            with open(self.failed_files_log, 'w') as f:
                json.dump(failed_files, f, indent=2)
            
            logger.info(f"📝 Logged failed file: {file_path}")
        except Exception as e:
            logger.error(f"Error logging failed file: {e}")
    
    def _worker_loop(self):
        """Worker thread loop - processes files from queue"""
        thread_name = threading.current_thread().name
        logger.info(f"🚀 {thread_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get file from queue (with timeout to allow checking shutdown)
                try:
                    file_path = self.work_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                logger.info(f"🔧 {thread_name} processing: {os.path.basename(file_path)}")
                
                # Process the file
                self._process_file(file_path)
                
                # Mark task as done
                self.work_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in {thread_name}: {e}", exc_info=True)
        
        logger.info(f"🛑 {thread_name} stopped")
    
    def shutdown(self):
        """Shutdown worker threads gracefully"""
        logger.info("Shutting down worker threads...")
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
        
        logger.info("All worker threads stopped")
    
    def _clean_debounce_dict(self):
        """Clean old entries from debounce dictionary (must be called with lock held)"""
        if len(self.file_modified_times) > self.MAX_DEBOUNCE_ENTRIES:
            # Sort by timestamp and remove oldest half
            sorted_entries = sorted(self.file_modified_times.items(), key=lambda x: x[1])
            entries_to_remove = len(self.file_modified_times) - (self.MAX_DEBOUNCE_ENTRIES // 2)
            
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.file_modified_times[key]
            
            logger.info(f"🧹 Cleaned debounce dictionary: removed {entries_to_remove} old entries")
    
    def _increment_success(self):
        """Thread-safe increment of success counter"""
        with self.stats_lock:
            self.success_count += 1
    
    def _increment_failed(self):
        """Thread-safe increment of failed counter"""
        with self.stats_lock:
            self.failed_count += 1
    
    def on_created(self, event):
        """Called when a file is created - NON-BLOCKING, just queues the file"""
        if event.is_directory:
            logger.debug(f"Directory created (ignored): {event.src_path}")
            return
        
        logger.debug(f"File created event: {event.src_path}")
        
        # Only process image and PDF files
        if not self._is_supported_file(event.src_path):
            logger.debug(f"File ignored (not supported format): {event.src_path}")
            return
        
        # Check if already processed (resume after crash/power failure)
        if self._is_already_processed(event.src_path):
            logger.info(f"⏭️ SKIPPING already processed file: {event.src_path}")
            return
        
        # Thread-safe check: Skip if this file existed before service started
        with self.lock:
            if event.src_path in self.existing_files:
                logger.info(f"⏭️ SKIPPING pre-existing file: {event.src_path}")
                return
        
        logger.info(f"🎯 NEW FILE DETECTED: {event.src_path}")
        
        # Thread-safe debounce check
        file_key = event.src_path
        current_time = time.time()
        
        should_queue = False
        with self.lock:
            if file_key in self.file_modified_times:
                elapsed = current_time - self.file_modified_times[file_key]
                if elapsed < self.DEBOUNCE_DELAY:
                    logger.debug(f"Debouncing: {file_key} (elapsed: {elapsed:.1f}s)")
                    return
            
            # Update timestamp and queue the file
            self.file_modified_times[file_key] = current_time
            should_queue = True
            
            # Clean debounce dictionary if needed
            self._clean_debounce_dict()
        
        if should_queue:
            # Add to work queue (non-blocking!)
            self.work_queue.put(event.src_path)
            logger.info(f"📥 QUEUED for processing: {os.path.basename(event.src_path)} (Queue size: {self.work_queue.qsize()})")
    
    def _is_supported_file(self, file_path):
        """Check if file is a supported image or PDF"""
        supported_extensions = ('.jpg', '.jpeg', '.png', '.pdf', '.JPG', '.JPEG', '.PNG', '.PDF')
        is_supported = file_path.lower().endswith(supported_extensions)
        
        if not is_supported:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.debug(f"⚠️ Unsupported file type '{file_ext}' - only images (JPG, PNG) and PDFs are supported")
        
        return is_supported
    
    def _process_file(self, file_path):
        """Process a single transaction file (called by worker threads)"""
        try:
            # Wait for file to be fully written (debounce delay)
            time.sleep(self.DEBOUNCE_DELAY)
            
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {file_path}")
                return
            
            # Wait for file to be fully written (size stable check)
            if not self._wait_for_file_ready(file_path):
                logger.error(f"File never became ready: {file_path}")
                self._increment_failed()
                self._log_failed_file(file_path, "File readiness check timeout")
                logger.error(f"File kept in place (failed processing): {file_path}")
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
                self._increment_failed()
                self._log_failed_file(file_path, "Inference returned None")
                logger.error(f"File kept in place (failed inference): {file_path}")
                return
            
            # Extract JSON from response
            logger.debug(f"Extracting JSON from response")
            transaction_data = inference_engine.extract_json_from_response(response)
            
            if not transaction_data or all(v == "Not Found" for v in transaction_data.values()):
                logger.warning(f"No valid transaction data extracted from {file_name}")
                self._increment_failed()
                self._log_failed_file(file_path, "No valid data extracted")
                logger.error(f"File kept in place (no valid data): {file_path}")
                return
            
            # Add metadata
            transaction_data['FileName'] = file_name
            
            # ACCOUNT NUMBER LOOKUP: Match ToAccountNumber to find ToBankName
            to_account_num = transaction_data.get('ToAccountNumber', 'Not Found')
            if to_account_num and to_account_num != 'Not Found':
                matched_bank = db_manager.find_bank_by_account_number(to_account_num)
                if matched_bank:
                    transaction_data['ToBankName'] = matched_bank
                    logger.info(f"🔍 Matched account {to_account_num} → {matched_bank}")
                else:
                    transaction_data['ToBankName'] = 'Not Found'
                    logger.warning(f"⚠️ No bank match found for account: {to_account_num}")
            else:
                transaction_data['ToBankName'] = 'Not Found'
            
            # Save to database
            logger.debug(f"Saving transaction to database")
            db_manager.insert_record(transaction_data)
            
            logger.info(f"✅ Successfully processed: {file_name}")
            logger.debug(f"Extracted data: {transaction_data}")
            self._increment_success()
            
            # Mark file as processed (for restart recovery)
            self._mark_file_processed(file_path)
            
            # Write notification file to trigger Streamlit refresh
            self._write_update_notification()
            
            # Files are kept in the incoming folder - no movement operations
            logger.info(f"File kept in place: {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            self._increment_failed()
            self._log_failed_file(file_path, f"Exception: {str(e)}")
            # Log the error but don't move the file
            logger.error(f"Processing failed for {file_path} - file kept in place for manual review")
    
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
    
    def _write_update_notification(self):
        """Write a notification file to signal Streamlit to refresh
        Uses atomic write (temp file + rename) to prevent race conditions"""
        try:
            notification_dir = os.path.join(
                os.environ.get('APPDATA', os.path.expanduser('~')),
                'PakistanBankParser', 'config'
            )
            os.makedirs(notification_dir, exist_ok=True)
            
            notification_file = os.path.join(notification_dir, '.db_updated')
            
            # Write atomically: write to temp file first, then rename
            # This prevents readers from getting partial/corrupted timestamps
            temp_file = notification_file + '.tmp'
            try:
                with open(temp_file, 'w') as f:
                    f.write(str(time.time()))
                # Atomic rename (on Windows, this replaces the old file)
                os.replace(temp_file, notification_file)
                logger.debug(f"✉️ Notification written: Database updated")
            except Exception as e:
                # Clean up temp file if rename failed
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise e
                
        except Exception as e:
            logger.warning(f"⚠️ Could not write update notification: {e}")
    
    def get_stats(self):
        """Return processing statistics (thread-safe)"""
        with self.stats_lock:
            return {
                'success': self.success_count,
                'failed': self.failed_count,
                'total': self.success_count + self.failed_count,
                'pending': self.work_queue.qsize()
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
            
            # Scan and mark all existing files as "already seen"
            self._scan_existing_files(incoming_dir)
            
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
    
    def _scan_existing_files(self, folder_path):
        """
        Scan existing files in folder and process unprocessed ones (restart recovery)
        Thread-safe - marks old files as existing, queues new unprocessed files
        """
        try:
            new_files_count = 0
            skipped_count = 0
            
            if os.path.isdir(folder_path):
                logger.info(f"📂 Scanning existing files in: {folder_path}")
                
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Only process files (not directories)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # Only process supported file types
                    if not self.event_handler._is_supported_file(file_path):
                        continue
                    
                    # Check if already processed in JSON
                    if self.event_handler._is_already_processed(file_path):
                        # Mark as existing (won't process again)
                        with self.event_handler.lock:
                            self.event_handler.existing_files.add(file_path)
                        skipped_count += 1
                        logger.debug(f"⏭️ Skipping already processed: {filename}")
                    else:
                        # Not in JSON - but could be old file that was cleaned up
                        # Check file age: if >30 days old, assume it was already processed
                        try:
                            file_mtime = os.path.getmtime(file_path)
                            file_age_days = (time.time() - file_mtime) / (24 * 60 * 60)
                            
                            if file_age_days > 30:
                                # File is old (>30 days) - must have been processed before
                                # Skip it to prevent re-processing
                                with self.event_handler.lock:
                                    self.event_handler.existing_files.add(file_path)
                                skipped_count += 1
                                logger.info(f"⏭️ Skipping old file (>{int(file_age_days)} days): {filename}")
                            else:
                                # File is recent (<30 days) and not in JSON - NEW FILE!
                                # Queue it for processing
                                self.event_handler.work_queue.put(file_path)
                                new_files_count += 1
                                logger.info(f"📥 QUEUED unprocessed file from restart: {filename}")
                        except Exception as e:
                            # If we can't get file time, queue it to be safe
                            logger.warning(f"⚠️ Could not check age of {filename}, queueing: {e}")
                            self.event_handler.work_queue.put(file_path)
                            new_files_count += 1
                
                logger.info(f"📊 Scan complete: {skipped_count} already processed, {new_files_count} queued for processing")
        except Exception as e:
            logger.warning(f"⚠️ Error scanning existing files: {e}")
    
    def stop(self):
        """Stop monitoring and shutdown worker threads"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
                logger.info("Observer stopped")
            
            if self.event_handler:
                self.event_handler.shutdown()
                logger.info("Worker threads stopped")
            
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
                logger.info(f"Processing stats - Success: {stats['success']}, Failed: {stats['failed']}, Pending: {stats['pending']}")
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
                logger.info(f"Processing stats - Success: {stats['success']}, Failed: {stats['failed']}, Pending: {stats['pending']}")
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
