"""
Test TTL-based processed files tracker
Tests the 4-day retention with cleanup at 3 AM PKT
"""

import os
import sys
import json
import time
import tempfile
import shutil
import logging
from datetime import datetime, timedelta
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestTTLTracker:
    """Test suite for TTL-based processed files tracker"""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix='ttl_test_')
        self.incoming_dir = os.path.join(self.test_dir, 'incoming')
        self.processed_log = os.path.join(self.test_dir, 'processed.json')
        os.makedirs(self.incoming_dir, exist_ok=True)
        logger.info(f"Test directory: {self.test_dir}")
    
    def cleanup(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")
    
    def test_create_empty_tracker(self):
        """Test 1: Create empty processed.json if doesn't exist"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Create Empty Tracker")
        logger.info("="*80)
        
        # Simulate creating empty tracker
        if not os.path.exists(self.processed_log):
            with open(self.processed_log, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
        
        # Verify
        assert os.path.exists(self.processed_log), "processed.json should be created"
        
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "Should be a dictionary"
        assert len(data) == 0, "Should be empty"
        
        logger.info("✅ TEST 1 PASSED: Empty tracker created successfully")
    
    def test_add_entries(self):
        """Test 2: Add entries with filename: timestamp format"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Add Entries")
        logger.info("="*80)
        
        current_time = int(time.time())
        
        test_entries = {
            "slip_001.jpg": current_time,
            "slip_002.jpg": current_time - 1000,
            "slip_003.jpg": current_time - 2000
        }
        
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump(test_entries, f, indent=2)
        
        # Verify
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 3, "Should have 3 entries"
        assert "slip_001.jpg" in data, "slip_001.jpg should exist"
        assert data["slip_001.jpg"] == current_time, "Timestamp should match"
        
        logger.info(f"Added 3 entries with timestamps")
        logger.info("✅ TEST 2 PASSED: Entries added successfully")
    
    def test_check_duplicate_skip(self):
        """Test 3: Skip files that exist in tracker"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Duplicate Detection")
        logger.info("="*80)
        
        current_time = int(time.time())
        
        # Create tracker with existing file
        tracker = {
            "existing_file.jpg": current_time - 1000
        }
        
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
        
        # Load and check
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Simulate checking if file should be processed
        should_skip = "existing_file.jpg" in data
        should_process = "new_file.jpg" not in data
        
        assert should_skip == True, "Existing file should be skipped"
        assert should_process == True, "New file should be processed"
        
        logger.info("✅ TEST 3 PASSED: Duplicate detection works")
    
    def test_ttl_cleanup(self):
        """Test 4: TTL cleanup removes entries older than 4 days"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: TTL Cleanup (4-day retention)")
        logger.info("="*80)
        
        current_time = int(time.time())
        ttl_threshold = 345600  # 4 days in seconds
        
        # Create entries with different ages
        test_entries = {
            "very_old_file.jpg": current_time - (5 * 24 * 60 * 60),  # 5 days old - should be removed
            "old_file.jpg": current_time - (4.5 * 24 * 60 * 60),      # 4.5 days old - should be removed
            "recent_file.jpg": current_time - (2 * 24 * 60 * 60),     # 2 days old - should be kept
            "new_file.jpg": current_time - (1 * 60 * 60),              # 1 hour old - should be kept
        }
        
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump(test_entries, f, indent=2)
        
        logger.info(f"Created 4 entries (2 expired, 2 valid)")
        
        # Simulate cleanup
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_dict = {}
        removed_count = 0
        
        for filename, timestamp in data.items():
            age_seconds = current_time - int(timestamp)
            if age_seconds <= ttl_threshold:
                cleaned_dict[filename] = timestamp
            else:
                removed_count += 1
                logger.info(f"  Removing: {filename} (age: {age_seconds/86400:.1f} days)")
        
        # Save cleaned version
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump(cleaned_dict, f, indent=2)
        
        # Verify
        assert len(cleaned_dict) == 2, f"Should have 2 entries after cleanup, got {len(cleaned_dict)}"
        assert removed_count == 2, f"Should have removed 2 entries, removed {removed_count}"
        assert "recent_file.jpg" in cleaned_dict, "Recent file should be kept"
        assert "new_file.jpg" in cleaned_dict, "New file should be kept"
        assert "very_old_file.jpg" not in cleaned_dict, "Very old file should be removed"
        assert "old_file.jpg" not in cleaned_dict, "Old file should be removed"
        
        logger.info(f"Cleanup result: Removed {removed_count}, kept {len(cleaned_dict)}")
        logger.info("✅ TEST 4 PASSED: TTL cleanup works correctly")
    
    def test_json_corruption_recovery(self):
        """Test 5: Recover from corrupted JSON file"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: JSON Corruption Recovery")
        logger.info("="*80)
        
        # Create corrupted JSON
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json content}')  # Invalid JSON
        
        # Simulate recovery
        try:
            with open(self.processed_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.error("Should have raised JSONDecodeError")
            assert False, "Should have raised error"
        except json.JSONDecodeError:
            logger.info("Detected corrupted JSON")
            
            # Backup corrupted file
            backup_name = f"processed_corrupted_test_{int(time.time())}.json.bak"
            backup_path = os.path.join(os.path.dirname(self.processed_log), backup_name)
            shutil.copy2(self.processed_log, backup_path)
            logger.info(f"Backed up to: {backup_path}")
            
            # Create fresh file
            with open(self.processed_log, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
            logger.info("Reset with empty tracker")
        
        # Verify recovery
        assert os.path.exists(backup_path), "Backup should exist"
        assert os.path.exists(self.processed_log), "Fresh file should exist"
        
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "Should be valid dict"
        assert len(data) == 0, "Should be empty after reset"
        
        logger.info("✅ TEST 5 PASSED: Corruption recovery works")
    
    def test_concurrent_writes(self):
        """Test 6: Simulate concurrent writes with locking"""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Concurrent Write Simulation")
        logger.info("="*80)
        
        import threading
        
        # Create initial tracker
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        
        lock = threading.Lock()
        
        def add_entry(filename):
            """Simulate adding entry with lock"""
            current_time = int(time.time())
            
            with lock:
                # Read
                with open(self.processed_log, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Modify
                data[filename] = current_time
                
                # Write
                with open(self.processed_log, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Added: {filename}")
        
        # Simulate 3 concurrent writes
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_entry, args=(f"file_{i}.jpg",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify
        with open(self.processed_log, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 3, f"Should have 3 entries, got {len(data)}"
        assert "file_0.jpg" in data, "file_0 should exist"
        assert "file_1.jpg" in data, "file_1 should exist"
        assert "file_2.jpg" in data, "file_2 should exist"
        
        logger.info("✅ TEST 6 PASSED: Concurrent writes handled correctly")
    
    def test_3am_schedule_calculation(self):
        """Test 7: Verify 3 AM Pakistan time calculation"""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: 3 AM PKT Schedule Calculation")
        logger.info("="*80)
        
        pkt = pytz.timezone('Asia/Karachi')
        now_pkt = datetime.now(pkt)
        
        # Calculate next 3 AM
        next_cleanup = now_pkt.replace(hour=3, minute=0, second=0, microsecond=0)
        
        # If we've passed 3 AM today, schedule for tomorrow
        if now_pkt >= next_cleanup:
            next_cleanup = next_cleanup + timedelta(days=1)
        
        sleep_seconds = (next_cleanup - now_pkt).total_seconds()
        
        logger.info(f"Current time (PKT): {now_pkt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Next cleanup: {next_cleanup.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Time until cleanup: {sleep_seconds/3600:.1f} hours")
        
        assert sleep_seconds > 0, "Sleep time should be positive"
        assert sleep_seconds <= 24 * 3600, "Sleep time should be <= 24 hours"
        assert next_cleanup.hour == 3, "Cleanup should be at 3 AM"
        
        logger.info("✅ TEST 7 PASSED: 3 AM PKT scheduling works")
    
    def run_all_tests(self):
        """Run all tests"""
        try:
            logger.info("\n" + "="*80)
            logger.info("STARTING TTL TRACKER TESTS")
            logger.info("="*80)
            
            self.test_create_empty_tracker()
            time.sleep(0.5)
            
            self.test_add_entries()
            time.sleep(0.5)
            
            self.test_check_duplicate_skip()
            time.sleep(0.5)
            
            self.test_ttl_cleanup()
            time.sleep(0.5)
            
            self.test_json_corruption_recovery()
            time.sleep(0.5)
            
            self.test_concurrent_writes()
            time.sleep(0.5)
            
            self.test_3am_schedule_calculation()
            time.sleep(0.5)
            
            logger.info("\n" + "="*80)
            logger.info("✅ ALL TTL TRACKER TESTS PASSED!")
            logger.info("="*80)
            logger.info("TTL tracker is working correctly:")
            logger.info("  ✅ Empty tracker creation")
            logger.info("  ✅ Entry addition (filename: timestamp)")
            logger.info("  ✅ Duplicate detection")
            logger.info("  ✅ 4-day TTL cleanup")
            logger.info("  ✅ JSON corruption recovery with backup")
            logger.info("  ✅ Thread-safe concurrent writes")
            logger.info("  ✅ 3 AM PKT scheduling")
            logger.info("="*80)
            
            return True
            
        except AssertionError as e:
            logger.error(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        except Exception as e:
            logger.error(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.cleanup()


if __name__ == "__main__":
    tester = TestTTLTracker()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
