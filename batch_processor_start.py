"""
Simple batch processor starter
Runs ONLY the batch processor without launcher dependencies
"""

import sys
import os

# Ensure UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("="*60)
print("BATCH PROCESSOR STARTUP WRAPPER")
print("="*60)

try:
    # Import only what we need
    import batch_config
    import batch_processor_service
    import logging
    from datetime import datetime
    import db_manager
    
    # Setup logging
    log_dir = db_manager.get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"batch_processor_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("BATCH PROCESSOR STARTING")
    logger.info("="*60)
    
    # Load config
    config = batch_config.load_config()
    logger.info(f"Config loaded: enabled={config.get('enabled')}")
    logger.info(f"Incoming folder: {config.get('incoming_folder')}")
    
    # Run batch processor
    logger.info("Running batch processor in background mode...")
    batch_processor_service.run_batch_processor(config, run_in_foreground=False)
    
    logger.info("Batch processor service stopped")

except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
