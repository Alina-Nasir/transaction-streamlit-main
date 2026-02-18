"""
Database Manager for Pakistani Bank Transaction Parser
Supports MS SQL Server database
"""

import os
import sys
import json
from datetime import datetime
import logging

# Try importing pyodbc for MS SQL Server
try:
    import pyodbc
    MSSQL_AVAILABLE = True
except ImportError:
    MSSQL_AVAILABLE = False
    pyodbc = None

logger = logging.getLogger(__name__)


def get_app_dir():
    """Get application directory - AppData in bundled mode, current directory in dev mode"""
    if getattr(sys, 'frozen', False):
        # Running in bundled executable - use AppData
        appdata = os.getenv('APPDATA')
        app_dir = os.path.join(appdata, 'PakistanBankParser')
        os.makedirs(app_dir, exist_ok=True)
        return app_dir
    else:
        # Running in development mode - use current directory
        return os.getcwd()


def get_db_path():
    """Get database path - uses %APPDATA% in bundled mode"""
    return os.path.join(get_app_dir(), 'transactions.db')


def get_log_dir():
    """Get logs directory"""
    log_dir = os.path.join(get_app_dir(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_config_dir():
    """Get configuration directory"""
    config_dir = os.path.join(get_app_dir(), 'config')
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_db_config():
    """Load database configuration (MS SQL only)"""
    config_file = os.path.join(get_config_dir(), 'db_config.json')
    
    # Check if config file exists
    if not os.path.exists(config_file):
        raise Exception(f"Database configuration not found at: {config_file}. Please create db_config.json with MS SQL credentials.")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
            # Validate MS SQL configuration
            if config.get('type') != 'mssql':
                raise Exception(f"Invalid database type: {config.get('type')}. Only 'mssql' is supported.")
            
            required = ['host', 'port', 'database']
            missing = [key for key in required if key not in config]
            if missing:
                raise Exception(f"Incomplete MS SQL configuration. Missing fields: {', '.join(missing)}")
            
            # Check if using Windows Authentication (Trusted Connection)
            if not config.get('trusted_connection', False):
                # SQL Server Authentication requires user and password
                if 'user' not in config or 'password' not in config:
                    raise Exception("MS SQL configuration missing user/password. Set trusted_connection=true for Windows Authentication.")
            
            return config
    except Exception as e:
        logger.error(f"Error loading DB config: {e}")
        raise


def save_db_config(config):
    """Save database configuration"""
    try:
        config_file = os.path.join(get_config_dir(), 'db_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving DB config: {e}")
        return False


def get_connection():
    """Get database connection based on configuration"""
    config = get_db_config()
    
    if not MSSQL_AVAILABLE:
        raise Exception("pyodbc is not installed. Please install pyodbc for MS SQL Server support")
    
    if config['type'] != 'mssql':
        raise Exception(f"Only MS SQL Server is supported. Current config type: {config.get('type')}")
    
    try:
        # Build MS SQL connection string
        if config.get('trusted_connection', False):
            # Windows Authentication (Trusted Connection)
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config['host']},{config.get('port', 1433)};"
                f"DATABASE={config['database']};"
                f"Trusted_Connection=yes;"
            )
        else:
            # SQL Server Authentication (user/password)
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config['host']},{config.get('port', 1433)};"
                f"DATABASE={config['database']};"
                f"UID={config['user']};"
                f"PWD={config['password']}"
            )
        conn = pyodbc.connect(conn_str)
        return conn, 'mssql'
    except pyodbc.Error as e:
        logger.error(f"MS SQL connection failed: {e}")
        raise Exception(f"Failed to connect to MS SQL Server: {e}")


def init_db():
    """Initialize database and create transactions table if it doesn't exist"""
    try:
        config = get_db_config()
        
        if not MSSQL_AVAILABLE:
            raise Exception("pyodbc is not installed")
        
        # First, connect to master database to create transactions_db if needed
        try:
            if config.get('trusted_connection', False):
                master_conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={config['host']},{config.get('port', 1433)};"
                    f"DATABASE=master;"
                    f"Trusted_Connection=yes;"
                )
            else:
                master_conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={config['host']},{config.get('port', 1433)};"
                    f"DATABASE=master;"
                    f"UID={config['user']};"
                    f"PWD={config['password']}"
                )
            
            master_conn = pyodbc.connect(master_conn_str)
            master_conn.autocommit = True
            master_cursor = master_conn.cursor()
            
            # Create database if it doesn't exist
            db_name = config['database']
            master_cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '{db_name}')
                CREATE DATABASE [{db_name}]
            """)
            
            master_cursor.close()
            master_conn.close()
            print(f"✅ Database '{db_name}' is ready")
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            # Continue anyway - database might already exist
        
        # Now connect to the transactions database
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        # MS SQL table creation only
        # Create transactions table
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='transactions' AND xtype='U')
            CREATE TABLE transactions (
                id INT IDENTITY(1,1) PRIMARY KEY,
                bankName NVARCHAR(255),
                Date NVARCHAR(50),
                TransactionID NVARCHAR(255),
                Amount NVARCHAR(50),
                FromAccount NVARCHAR(255),
                FromAccountNumber NVARCHAR(100),
                ToAccount NVARCHAR(255),
                ToAccountNumber NVARCHAR(100),
                ToBankName NVARCHAR(255),
                PaymentMode NVARCHAR(100),
                FileName NVARCHAR(255),
                ProcessedDate NVARCHAR(50),
                CreatedAt DATETIME DEFAULT GETDATE()
            )
        ''')
        
        # Create user_banks table for bank account management
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_banks' AND xtype='U')
            CREATE TABLE user_banks (
                id INT IDENTITY(1,1) PRIMARY KEY,
                bank_name NVARCHAR(255) NOT NULL,
                account_number NVARCHAR(100) NOT NULL,
                CreatedAt DATETIME DEFAULT GETDATE()
            )
        ''')
        
        # Check if user_banks table is empty and add default banks
        cursor.execute("SELECT COUNT(*) FROM user_banks")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert default bank entries
            cursor.execute('''
                INSERT INTO user_banks (bank_name, account_number) VALUES (?, ?)
            ''', ('Habib Metro', '6017'))
            
            cursor.execute('''
                INSERT INTO user_banks (bank_name, account_number) VALUES (?, ?)
            ''', ('Meezan Bank', '9339'))
            
            print("✅ Default banks added: Habib Metro (6017), Meezan Bank (9339)")
        
        print(f"✅ MS SQL database initialized: {config['database']}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        logger.error(f"Database initialization error: {e}", exc_info=True)
        return False


def insert_record(data):
    """Insert transaction record into database"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        # MS SQL insert query with ? placeholders (removed obsolete fields)
        cursor.execute('''
            INSERT INTO transactions (
                bankName, Date, TransactionID, Amount, FromAccount, FromAccountNumber,
                ToAccount, ToAccountNumber, ToBankName, PaymentMode,
                FileName, ProcessedDate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('bankName', 'Not Found'),
            data.get('Date', 'Not Found'),
            data.get('TransactionID', 'Not Found'),
            data.get('Amount', 'Not Found'),
            data.get('FromAccount', 'Not Found'),
            data.get('FromAccountNumber', 'Not Found'),
            data.get('ToAccount', 'Not Found'),
            data.get('ToAccountNumber', 'Not Found'),
            data.get('ToBankName', 'Not Found'),
            data.get('PaymentMode', 'Not Found'),
            data.get('FileName', 'Not Found'),
            data.get('ProcessedDate', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error inserting record: {str(e)}")
        return False


def get_all_transactions():
    """Retrieve all transactions from database"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM transactions ORDER BY id DESC')
        rows = cursor.fetchall()
        
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Error retrieving transactions: {str(e)}")
        return []


# ============================================================================
# BANK MANAGEMENT FUNCTIONS
# ============================================================================

def add_bank(bank_name, account_number):
    """Add a new bank account to user_banks table"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_banks (bank_name, account_number)
            VALUES (?, ?)
        ''', (bank_name.strip(), account_number.strip()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error adding bank: {e}")
        return False


def get_all_banks():
    """Retrieve all banks from user_banks table"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, bank_name, account_number, CreatedAt FROM user_banks ORDER BY id DESC')
        rows = cursor.fetchall()
        
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error retrieving banks: {e}")
        return []


def update_bank(bank_id, bank_name, account_number):
    """Update an existing bank account"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_banks
            SET bank_name = ?, account_number = ?
            WHERE id = ?
        ''', (bank_name.strip(), account_number.strip(), bank_id))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    except Exception as e:
        logger.error(f"Error updating bank: {e}")
        return False


def delete_bank(bank_id):
    """Delete a bank account from user_banks table"""
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_banks WHERE id = ?', (bank_id,))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    except Exception as e:
        logger.error(f"Error deleting bank: {e}")
        return False


def find_bank_by_account_number(account_number_partial):
    """
    Find bank name by matching account number (supports partial matching)
    Extracts visible digits from masked numbers (e.g., '******232' -> '232')
    Then matches against bank accounts that end with those digits
    
    Args:
        account_number_partial: Full or partial account number from slip (may contain *, X, or other masking)
        
    Returns:
        Bank name if match found, None otherwise
    """
    import re
    
    if not account_number_partial or account_number_partial == "Not Found":
        return None
    
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT bank_name, account_number FROM user_banks')
        banks = cursor.fetchall()
        conn.close()
        
        # Step 1: Extract the LAST continuous group of digits (trailing digits)
        # This handles patterns like:
        #   - ******232 → 232
        #   - XXXX-1234 → 1234
        #   - 01*********39 → 39 (trailing digits only)
        #   - ***-***-9339 → 9339
        # Bank slips typically show the last few digits, so we match on trailing digits
        
        input_str = str(account_number_partial)
        
        # Find all continuous digit groups
        digit_groups = re.findall(r'\d+', input_str)
        
        if not digit_groups:
            logger.warning(f"No digits found in account number: {account_number_partial}")
            return None
        
        # Use the LAST group of digits (trailing digits)
        clean_input = digit_groups[-1]
        
        logger.info(f"Extracted trailing digits from '{account_number_partial}': '{clean_input}'")
        
        # Step 2: Match against each bank's account number
        # Strategy: Check if any stored account ENDS with the extracted digits
        for row in banks:
            bank_name = row[0]
            stored_account = row[1]
            
            # Clean the stored account number (remove spaces, dashes, etc.)
            clean_stored = re.sub(r'[^0-9]', '', str(stored_account))
            
            # Check if stored account ends with the extracted digits
            if clean_stored.endswith(clean_input):
                logger.info(f"✅ Matched '{account_number_partial}' (digits: {clean_input}) → Bank: {bank_name} (Account: {stored_account} ends with {clean_input})")
                return bank_name
        
        logger.warning(f"❌ No bank match found for account number: {account_number_partial} (extracted digits: {clean_input})")
        return None
        
    except Exception as e:
        logger.error(f"Error finding bank by account number: {e}")
        return None


if __name__ == "__main__":
    # Test database operations
    init_db()
    print("Database manager test completed")
