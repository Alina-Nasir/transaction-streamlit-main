"""
Generate a beautiful Markdown report from SQLite database entries
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

def generate_markdown_report():
    """Generate a markdown report of all database entries"""
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
        
        # Start markdown content
        md_content = """# 🏦 Pakistani Bank Transaction Database Report

"""
        md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_entries = cursor.fetchone()[0]
        
        if total_entries == 0:
            md_content += "⚠️ **No entries found in database yet.**\n"
            print(md_content)
            return
        
        md_content += f"## 📊 Database Summary\n\n"
        md_content += f"- **Total Transactions:** {total_entries}\n"
        md_content += f"- **Report Generated:** {datetime.now().strftime('%d %B %Y at %H:%M:%S')}\n\n"
        
        # Get unique banks
        cursor.execute("SELECT DISTINCT bankName FROM transactions WHERE bankName != 'Not Found'")
        unique_banks = [row[0] for row in cursor.fetchall()]
        md_content += f"- **Banks in Database:** {', '.join(unique_banks) if unique_banks else 'None'}\n\n"
        
        # Field completeness
        md_content += "## ✔️ Field Completeness\n\n"
        
        fields_to_check = [
            'bankName', 'Date', 'Amount', 'TransactionID',
            'FromAccount', 'FromAccountNumber', 'FromBankName',
            'ToAccount', 'ToAccountNumber', 'ToBankName',
            'Branch', 'PaymentMode', 'CustomerID', 'ChequeNo', 'Remarks'
        ]
        
        md_content += "| Field | Filled | Empty | Completion % |\n"
        md_content += "|-------|--------|-------|----------|\n"
        
        for field in fields_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM transactions WHERE {field} != 'Not Found' AND {field} IS NOT NULL")
            filled_count = cursor.fetchone()[0]
            empty_count = total_entries - filled_count
            percentage = (filled_count / total_entries) * 100 if total_entries > 0 else 0
            
            status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
            md_content += f"| {field} | {filled_count} | {empty_count} | {status} {percentage:.1f}% |\n"
        
        md_content += "\n"
        
        # All transactions
        md_content += "## 📋 All Transactions\n\n"
        
        df = pd.read_sql_query("""
            SELECT id, bankName, Date, TransactionID, Amount, FromAccount, FromAccountNumber,
                   FromBankName, ToAccount, ToAccountNumber, ToBankName, Branch, PaymentMode,
                   CustomerID, ChequeNo, Remarks, FileName, ProcessedDate, CreatedAt
            FROM transactions
            ORDER BY id DESC
        """, conn)
        
        # Generate individual transaction cards
        for idx, row in df.iterrows():
            entry_id = row['id']
            md_content += f"### Transaction #{entry_id}\n\n"
            
            md_content += f"**File Name:** `{row['FileName']}`  \n"
            md_content += f"**Saved At:** {row['CreatedAt']}\n"
            md_content += f"**Processed Date:** {row['ProcessedDate']}\n\n"
            
            md_content += "#### Bank Information\n"
            md_content += f"- **Bank Name:** {row['bankName']}\n"
            md_content += f"- **Branch:** {row['Branch']}\n\n"
            
            md_content += "#### Transaction Details\n"
            md_content += f"- **Date:** {row['Date']}\n"
            md_content += f"- **Transaction ID:** {row['TransactionID']}\n"
            md_content += f"- **Amount:** {row['Amount']}\n"
            md_content += f"- **Payment Mode:** {row['PaymentMode']}\n"
            md_content += f"- **Cheque No:** {row['ChequeNo']}\n"
            md_content += f"- **Remarks:** {row['Remarks']}\n\n"
            
            md_content += "#### Sender Details\n"
            md_content += f"- **Name:** {row['FromAccount']}\n"
            md_content += f"- **Account Number:** {row['FromAccountNumber']}\n"
            md_content += f"- **Bank:** {row['FromBankName']}\n\n"
            
            md_content += "#### Receiver Details\n"
            md_content += f"- **Name:** {row['ToAccount']}\n"
            md_content += f"- **Account Number:** {row['ToAccountNumber']}\n"
            md_content += f"- **Bank:** {row['ToBankName']}\n\n"
            
            md_content += "#### Additional Information\n"
            md_content += f"- **Customer ID:** {row['CustomerID']}\n"
            md_content += f"- **Processed Date:** {row['ProcessedDate']}\n\n"
            
            md_content += "---\n\n"
        
        # Summary table
        md_content += "## 📊 Summary Table (All Fields)\n\n"
        
        summary_df = df[['id', 'bankName', 'Date', 'TransactionID', 'Amount', 
                         'FromAccount', 'FromAccountNumber', 'FromBankName',
                         'ToAccount', 'ToAccountNumber', 'ToBankName',
                         'Branch', 'PaymentMode', 'CustomerID', 'ChequeNo', 'Remarks', 'CreatedAt']].copy()
        md_content += summary_df.to_markdown(index=False)
        md_content += "\n\n"
        
        # Save to file
        report_filename = f"database_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ Report generated successfully!")
        print(f"📄 Saved as: {report_filename}")
        print(f"📊 Total transactions: {total_entries}")
        
        conn.close()
        
        # Option to open the file
        response = input("\nWould you like to open the report? (yes/no): ").strip().lower()
        if response == 'yes':
            os.system(f"start {report_filename}")  # Windows
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database Error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🏦 DATABASE MARKDOWN REPORT GENERATOR")
    print("=" * 80 + "\n")
    generate_markdown_report()
