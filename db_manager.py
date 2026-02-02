"""
Database Manager for Pakistani Bank Transaction Parser
Handles SQLite database operations in bundled executable mode
"""

import sqlite3
import os
import sys
from datetime import datetime


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


def init_db():
    """Initialize SQLite database and create transactions table if it doesn't exist"""
    try:
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bankName TEXT,
                Date TEXT,
                TransactionID TEXT,
                Amount TEXT,
                FromAccount TEXT,
                FromAccountNumber TEXT,
                FromBankName TEXT,
                ToAccount TEXT,
                ToAccountNumber TEXT,
                ToBankName TEXT,
                Branch TEXT,
                PaymentMode TEXT,
                CustomerID TEXT,
                ChequeNo TEXT,
                Remarks TEXT,
                FileName TEXT,
                ProcessedDate TEXT,
                CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at: {db_path}")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        return False


def insert_record(data):
    """Insert transaction record into database"""
    try:
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transactions (
                bankName, Date, TransactionID, Amount, FromAccount, FromAccountNumber,
                FromBankName, ToAccount, ToAccountNumber, ToBankName, Branch, PaymentMode,
                CustomerID, ChequeNo, Remarks, FileName, ProcessedDate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('bankName', 'Not Found'),
            data.get('Date', 'Not Found'),
            data.get('TransactionID', 'Not Found'),
            data.get('Amount', 'Not Found'),
            data.get('FromAccount', 'Not Found'),
            data.get('FromAccountNumber', 'Not Found'),
            data.get('FromBankName', 'Not Found'),
            data.get('ToAccount', 'Not Found'),
            data.get('ToAccountNumber', 'Not Found'),
            data.get('ToBankName', 'Not Found'),
            data.get('Branch', 'Not Found'),
            data.get('PaymentMode', 'Not Found'),
            data.get('CustomerID', 'Not Found'),
            data.get('ChequeNo', 'Not Found'),
            data.get('Remarks', 'Not Found'),
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
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM transactions ORDER BY id DESC')
        rows = cursor.fetchall()
        
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Error retrieving transactions: {str(e)}")
        return []


if __name__ == "__main__":
    # Test database operations
    init_db()
    print("Database manager test completed")
