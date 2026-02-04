"""
Configuration management for Batch Processing Service
Stores and retrieves user-configurable settings for batch processing
"""

import os
import sys
import json
import logging
from pathlib import Path
import db_manager

logger = logging.getLogger(__name__)

# Configuration file location
CONFIG_DIR = db_manager.get_config_dir()
CONFIG_FILE = os.path.join(CONFIG_DIR, 'batch_config.json')


def get_default_config():
    """Get default batch processor configuration"""
    # Check if running in bundled mode
    is_bundled = getattr(sys, 'frozen', False)
    
    return {
        'incoming_folder': os.path.join(
            os.environ.get('USERPROFILE', os.path.expanduser('~')),
            'Pictures', 'BankSlips', 'incoming'
        ),
        'debounce_delay': 3.0,
        'enabled': is_bundled,  # Auto-enable batch processing in bundled installations
        'inference_timeout': 300
    }


def load_config():
    """Load batch processor configuration from file"""
    try:
        # Check if running in bundled mode
        is_bundled = getattr(sys, 'frozen', False)
        
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                logger.debug(f"Loaded batch config from {CONFIG_FILE}")
                
                # Ensure backward compatibility - add missing fields with defaults
                defaults = get_default_config()
                for key, value in defaults.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                        logger.debug(f"Added missing config field: {key} = {value}")
                
                # Force enable batch processing in bundled installations
                if is_bundled:
                    loaded_config['enabled'] = True
                    logger.info("Batch processing auto-enabled in bundled installation")
                
                return loaded_config
        else:
            logger.info("No config file found, using defaults")
            return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return get_default_config()


def save_config(config):
    """Save batch processor configuration to file"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Batch config saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def update_config(updates):
    """Update specific configuration values"""
    try:
        config = load_config()
        config.update(updates)
        
        if save_config(config):
            logger.info(f"Config updated: {updates}")
            return config
        else:
            logger.error("Failed to save updated config")
            return None
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return None


def validate_folder_path(path):
    """Validate that a folder path is accessible and create if needed"""
    try:
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(path, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        return True, path
    except PermissionError:
        return False, f"Permission denied: {path}"
    except Exception as e:
        return False, str(e)


def validate_config(config):
    """Validate all configuration values"""
    errors = []
    
    # Check incoming folder
    if not config.get('incoming_folder'):
        errors.append("Incoming folder not configured")
    else:
        valid, msg = validate_folder_path(config['incoming_folder'])
        if not valid:
            errors.append(f"Incoming folder error: {msg}")
    
    # Check processed folder
    if config.get('auto_move_processed'):
        if not config.get('processed_folder'):
            errors.append("Processed folder not configured but auto-move enabled")
        else:
            valid, msg = validate_folder_path(config['processed_folder'])
            if not valid:
                errors.append(f"Processed folder error: {msg}")
    
    # Check failed folder
    if not config.get('failed_folder'):
        errors.append("Failed folder not configured")
    else:
        valid, msg = validate_folder_path(config['failed_folder'])
        if not valid:
            errors.append(f"Failed folder error: {msg}")
    
    # Check debounce delay
    try:
        debounce = float(config.get('debounce_delay', 3.0))
        if debounce < 0.5 or debounce > 30:
            errors.append("Debounce delay must be between 0.5 and 30 seconds")
    except (TypeError, ValueError):
        errors.append("Debounce delay must be a number")
    
    # Check inference timeout
    try:
        timeout = float(config.get('inference_timeout', 300))
        if timeout < 30 or timeout > 900:
            errors.append("Inference timeout must be between 30 and 900 seconds")
    except (TypeError, ValueError):
        errors.append("Inference timeout must be a number")
    
    return errors if errors else None


def get_config_summary():
    """Get a user-friendly summary of current configuration"""
    config = load_config()
    return {
        'incoming_folder': config.get('incoming_folder'),
        'processed_folder': config.get('processed_folder'),
        'failed_folder': config.get('failed_folder'),
        'auto_move_processed': config.get('auto_move_processed', True),
        'debounce_delay': config.get('debounce_delay', 3.0),
        'enabled': config.get('enabled', False),
        'inference_timeout': config.get('inference_timeout', 300)
    }


# Helper functions for db_manager integration
def get_config_dir():
    """Get configuration directory"""
    if hasattr(db_manager, 'get_config_dir'):
        return db_manager.get_config_dir()
    
    # Fallback
    app_data = os.environ.get('APPDATA')
    if app_data:
        return os.path.join(app_data, 'PakistanBankParser', 'config')
    else:
        return os.path.expanduser('~/.PakistanBankParser/config')
