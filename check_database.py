"""
Script to check SQLite database entries and verify data integrity
"""

import sqlite3
import pandas as pd
from datetime import datetime

def check_database():
    """Check database entries and display statistics"""
    try:
        # Connect to database
        conn = sqlite3.connect('transactions.db')
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("❌ ERROR: 'transactions' table does not exist!")
            return
        
        print("✅ Database connection successful")
        print("=" * 80)
        
        # Get table schema
        print("\n📋 TABLE SCHEMA:")
        cursor.execute("PRAGMA table_info(transactions)")
        columns = cursor.fetchall()
        for col in columns:
            col_id, col_name, col_type, not_null, default, primary_key = col
            print(f"  - {col_name}: {col_type} (PK: {bool(primary_key)})")
        
        print("\n" + "=" * 80)
        
        # Count total entries
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_entries = cursor.fetchone()[0]
        print(f"\n📊 TOTAL ENTRIES: {total_entries}")
        
        if total_entries == 0:
            print("\n⚠️  No entries found in database yet.")
            conn.close()
            return
        
        # Get latest entry with ALL fields
        print(f"\n📝 LATEST ENTRY (ALL FIELDS):")
        print("=" * 80)
        
        cursor.execute("""
            SELECT bankName, Date, TransactionID, Amount, FromAccount, FromAccountNumber,
                   FromBankName, ToAccount, ToAccountNumber, ToBankName, Branch, PaymentMode,
                   CustomerID, ChequeNo, Remarks, FileName, ProcessedDate, CreatedAt
            FROM transactions
            ORDER BY id DESC
            LIMIT 1
        """)
        
        latest = cursor.fetchone()
        if latest:
            (bank, date, trans_id, amount, from_acc, from_acc_num, from_bank, 
             to_acc, to_acc_num, to_bank, branch, mode, cust_id, cheque, remarks, 
             filename, processed, created) = latest
            
            print(f"  Bank Name: {bank}")
            print(f"  Date: {date}")
            print(f"  Transaction ID: {trans_id}")
            print(f"  Amount: {amount}")
            print(f"  From Account: {from_acc}")
            print(f"  From Account Number: {from_acc_num}")
            print(f"  From Bank Name: {from_bank}")
            print(f"  To Account: {to_acc}")
            print(f"  To Account Number: {to_acc_num}")
            print(f"  To Bank Name: {to_bank}")
            print(f"  Branch: {branch}")
            print(f"  Payment Mode: {mode}")
            print(f"  Customer ID: {cust_id}")
            print(f"  Cheque No: {cheque}")
            print(f"  Remarks: {remarks}")
            print(f"  File Name: {filename}")
            print(f"  Processed Date: {processed}")
            print(f"  Created At: {created}")
        
        print("\n" + "=" * 80)
        
        # Data completeness check
        print("\n✔️  FIELD COMPLETENESS CHECK:")
        print("=" * 80)
        
        fields_to_check = [
            'bankName', 'Date', 'Amount', 'FromAccount', 'FromAccountNumber',
            'ToAccount', 'ToAccountNumber', 'Branch', 'PaymentMode', 'TransactionID'
        ]
        
        for field in fields_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM transactions WHERE {field} != 'Not Found' AND {field} IS NOT NULL")
            filled_count = cursor.fetchone()[0]
            percentage = (filled_count / total_entries) * 100 if total_entries > 0 else 0
            status = "✅" if percentage >= 80 else "⚠️ " if percentage >= 50 else "❌"
            print(f"{status} {field}: {filled_count}/{total_entries} ({percentage:.1f}%)")
        
        print("\n" + "=" * 80)
        
        # Show full DataFrame
        print("\n📑 FULL DATA TABLE (ALL FIELDS):")
        print("=" * 80)
        
        df = pd.read_sql_query("""
            SELECT id, bankName, Date, TransactionID, Amount, FromAccount, FromAccountNumber,
                   FromBankName, ToAccount, ToAccountNumber, ToBankName, Branch, PaymentMode,
                   CustomerID, ChequeNo, Remarks, FileName, ProcessedDate, CreatedAt
            FROM transactions
            ORDER BY id DESC
        """, conn)
        
        # Display with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        print(df.to_string(index=False))
        
        print("\n" + "=" * 80)
        print(f"\n✅ Database check completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database Error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def export_data_to_csv():
    """Export database to CSV for inspection"""
    try:
        conn = sqlite3.connect('transactions.db')
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY id DESC", conn)
        
        if len(df) > 0:
            output_file = f"transactions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ Data exported to: {output_file}")
        else:
            print("\n⚠️  No data to export")
        
        conn.close()
    except Exception as e:
        print(f"❌ Export Error: {str(e)}")

def clear_database():
    """Clear all entries from database (use with caution)"""
    try:
        response = input("\n⚠️  WARNING: This will DELETE all entries! Continue? (yes/no): ")
        if response.lower() == 'yes':
            conn = sqlite3.connect('transactions.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions")
            conn.commit()
            conn.close()
            print("✅ Database cleared successfully")
        else:
            print("❌ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🏦 PAKISTANI BANK TRANSACTION DATABASE CHECKER")
    print("=" * 80)
    
    while True:
        print("\nOptions:")
        print("1. Check database entries")
        print("2. Export data to CSV")
        print("3. Clear database")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            check_database()
        elif choice == '2':
            export_data_to_csv()
        elif choice == '3':
            clear_database()
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
