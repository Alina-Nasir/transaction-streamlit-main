import os
import streamlit as st
import base64
import io
import tempfile
import json
import re
import pandas as pd
from PIL import Image
import hashlib
from datetime import datetime
import requests
import uuid
import sqlite3
import logging

# Import database manager for bundled mode compatibility
import db_manager

# Import shared inference engine (used by both streamlit and batch processor)
import inference_engine

# Import batch configuration
try:
    import batch_config
    BATCH_CONFIG_AVAILABLE = True
except ImportError:
    BATCH_CONFIG_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Pakistani Bank Transaction Parser",
    page_icon="🏦",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 30px;
    }
    .pakistan-flag {
        background: linear-gradient(90deg, #115740 0%, #115740 33%, white 33%, white 66%, #115740 66%, #115740 100%);
        padding: 10px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .transaction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #3B82F6;
    }
    .bank-meezan {
        border-left-color: #00A651 !important;
    }
    .bank-habib {
        border-left-color: #7C0A02 !important;
    }
    .bank-ubl {
        border-left-color: #FF6B35 !important;
    }
    .bank-alfalah {
        border-left-color: #F9A826 !important;
    }
    .upload-box {
        border: 2px dashed #00A651;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background-color: #F8FFF8;
        margin: 20px 0;
    }
    .file-preview {
        max-width: 150px !important;  /* small thumbnail */
        max-height: 150px !important; /* keep it square-ish */
        object-fit: contain !important; /* scale without cropping */
        border-radius: 8px;
        margin: 0 auto;
        display: block;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def convert_pdf_to_images(pdf_file):
    """Convert PDF file to list of PIL Images using pypdfium2"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        pdf = pdfium.PdfDocument(tmp_path)
        images = []
        
        for page_number in range(len(pdf)):
            page = pdf.get_page(page_number)
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        
        pdf.close()
        os.unlink(tmp_path)
        
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return None

# ---------- FUNCTIONS ----------
# Use shared inference engine functions
def convert_pdf_to_images(pdf_file):
    """Convert PDF file to list of PIL Images using pypdfium2"""
    return inference_engine.convert_pdf_to_images(pdf_file)

# Use db_manager functions
init_sqlite_db = db_manager.init_db
insert_transaction_to_db = db_manager.insert_record

# Import inference functions from shared module
get_pakistani_bank_prompt = inference_engine.get_pakistani_bank_prompt
resize_image_to_512p = inference_engine.resize_image_to_512p
encode_image_to_base64 = inference_engine.encode_image_to_base64
call_local_model_with_image = inference_engine.call_local_model_with_image
extract_json_from_response = inference_engine.extract_json_from_response

def get_bank_style_class(bank_name):
    """Get CSS class based on bank name"""
    bank_name = str(bank_name).upper()
    
    if 'MEEZAN' in bank_name:
        return 'bank-meezan'
    elif 'HABIB' in bank_name or 'HBL' in bank_name:
        return 'bank-habib'
    elif 'UBL' in bank_name:
        return 'bank-ubl'
    elif 'ALFALAH' in bank_name:
        return 'bank-alfalah'
    elif 'HBL' in bank_name:
        return 'bank-hbl'
    elif 'ALLIED' in bank_name or 'ABL' in bank_name:
        return 'bank-allied'
    elif 'STANDARD CHARTERED' in bank_name:
        return 'bank-sc'
    elif 'BANK ISLAMI' in bank_name:
        return 'bank-islami'
    elif 'FAYSAL' in bank_name:
        return 'bank-faysal'
    elif 'ASKARI' in bank_name:
        return 'bank-askari'
    elif 'JS BANK' in bank_name or 'JSB' in bank_name:
        return 'bank-js'
    elif 'DUBAI ISLAMIC' in bank_name:
        return 'bank-dib'
    elif 'SONERI' in bank_name:
        return 'bank-soneri'
    else:
        return ''

def clean_amount(amount_str):
    """Clean and convert amount string to numeric"""
    try:
        if isinstance(amount_str, str):
            # Remove currency symbols, commas, and spaces
            cleaned = re.sub(r'[^\d.]', '', amount_str)
            return float(cleaned) if cleaned else 0.0
        elif isinstance(amount_str, (int, float)):
            return float(amount_str)
    except:
        return 0.0
    return 0.0

def display_single_transaction(data):
    """Display single transaction data"""
    bank_name = data.get('bankName', 'Unknown Bank')
    bank_class = get_bank_style_class(bank_name)
    
    with st.container():
        
        # Header with bank info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### Bank Name: {bank_name}")
        with col2:
            st.markdown(f"**Processed:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Account Information Section
        st.markdown("#### Account Information")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sender Information
            st.markdown("**Sender Details**")
            from_acc_name = data.get('FromAccount', 'Not Found')
            from_acc_num = data.get('FromAccountNumber', 'Not Found')
            from_bank_name = data.get('FromBankName', 'Not Found')
            
            if from_acc_num != 'Not Found':
                # Show the masked account number
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin: 10px 0;'>
                    <div style='font-size: 0.9rem; color: #666;'>Account Number</div>
                    <div style='font-size: 1.3rem; font-weight: 600; color: #dc3545; font-family: monospace;'>{from_acc_num}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if from_bank_name != 'Not Found':
                st.metric("Sender Bank", from_bank_name)
            
            if from_acc_name != 'Not Found':
                st.metric("Sender Name", from_acc_name)
        
        with col2:
            # Receiver Information
            st.markdown("**Receiver Details**")
            to_acc_name = data.get('ToAccount', 'Not Found')
            to_acc_num = data.get('ToAccountNumber', 'Not Found')
            to_bank_name = data.get('ToBankName', 'Not Found')
            
            if to_acc_num != 'Not Found':
                # Show the masked account number
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin: 10px 0;'>
                    <div style='font-size: 0.9rem; color: #666;'>Account Number</div>
                    <div style='font-size: 1.3rem; font-weight: 600; color: #28a745; font-family: monospace;'>{to_acc_num}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if to_bank_name != 'Not Found':
                st.metric("Receiver Bank", to_bank_name)
            
            if to_acc_name != 'Not Found':
                st.metric("Receiver Name", to_acc_name)
        
        # Transaction Details Section
        st.markdown("#### Transaction Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Transaction Date", data.get('Date', 'Not Found'))
            st.metric("Branch", data.get('Branch', 'Not Found'))
        
        with col2:
            amount = data.get('Amount', 'Not Found')
            if amount != 'Not Found':
                # Show amount heading and value
                st.markdown("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <div style='font-size: 0.85rem; color: #666; margin-bottom: 5px;'>Amount</div>
                    <div style='font-size: 1.5rem; font-weight: 600; color: #00A651;'>{}</div>
                </div>
                """.format(amount), unsafe_allow_html=True)
            else:
                st.metric("Amount", "Not Found")
            
            st.metric("Payment Mode", data.get('PaymentMode', 'Not Found'))
        
        with col3:
            st.metric("Transaction ID", data.get('TransactionID', 'Not Found'))
            st.metric("Customer ID", data.get('CustomerID', 'Not Found'))
            if data.get('ChequeNo', 'Not Found') != 'Not Found':
                st.metric("Cheque No", data.get('ChequeNo', 'Not Found'))
        
        # Remarks section
        if data.get('Remarks', 'Not Found') != 'Not Found':
            st.markdown("---")
            st.markdown(f"**Remarks:** {data.get('Remarks')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_dataframe(data_list):
    """Create DataFrame from extracted data"""
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list)
    
    # Reorder columns
    column_order = [
        'bankName', 'Date', 'TransactionID', 'Amount', 
        'FromAccount', 'FromAccountNumber', 'FromBankName',
        'ToAccount', 'ToAccountNumber', 'ToBankName',
        'Branch', 'PaymentMode', 'CustomerID', 'ChequeNo', 'Remarks', 
        'FileName', 'ProcessedDate'
    ]
    
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    return df

def export_to_csv(dataframe):
    """Export DataFrame to CSV"""
    if dataframe.empty:
        return None
    
    csv_string = dataframe.to_csv(index=False, encoding='utf-8')
    return csv_string

def export_to_excel(dataframe):
    """Export DataFrame to Excel"""
    if dataframe.empty:
        return None
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Transactions')
    
    processed_data = output.getvalue()
    return processed_data

# ---------- BATCH SETTINGS PAGE ----------
def batch_settings_page():
    """Simplified read-only settings page for batch processing configuration"""
    st.markdown("# Batch Processing Status")
    
    if not BATCH_CONFIG_AVAILABLE:
        st.error("Batch configuration module not available.")
        return
    
    # Get current configuration
    current_config = batch_config.load_config()
    
    st.markdown("## Configuration")
    st.info("""
    The batch processor automatically monitors a folder for new bank slips and processes them 24/7.
    These settings are managed by the system administrator.
    """)
    
    # Display configuration as read-only metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Folder Paths")
        st.code(f"""Incoming:
{current_config.get('incoming_folder', 'Not configured')}

Processed:
{current_config.get('processed_folder', 'Not configured')}

Failed:
{current_config.get('failed_folder', 'Not configured')}""")
    
    with col2:
        st.markdown("### Processing Settings")
        is_enabled = current_config.get('enabled', False)
        
        st.write(f"**Service Status:** {'🟢 Enabled' if is_enabled else '🔴 Disabled'}")
        st.write(f"**Auto-move processed files:** {current_config.get('auto_move_processed', True)}")
        st.write(f"**Debounce delay:** {current_config.get('debounce_delay', 3.0)} seconds")
        st.write(f"**Inference timeout:** {current_config.get('inference_timeout', 300)} seconds")
    
    # Information section
    st.markdown("## How It Works")
    with st.expander("Batch Processing Details"):
        st.markdown("""
        **Incoming Folder:** Drop new bank slip images here
        - The system monitors this folder 24/7
        - Automatically detects new files
        
        **Processing:** Files are analyzed automatically
        - Uses the same AI model as manual processing
        - Results saved to database
        
        **Organization:**
        - Processed files → Processed folder
        - Failed files → Failed folder
        - All activity logged
        
        **No action needed** - just add images to the incoming folder!
        """)
    
    # Log files viewer
    st.markdown("## Recent Activity")
    try:
        log_dir = db_manager.get_log_dir()
        log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('batch_processor_')], reverse=True)
        
        if log_files:
            selected_log = st.selectbox("View log file:", log_files[:5])
            log_path = os.path.join(log_dir, selected_log)
            
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Show last 50 lines
            lines = log_content.split('\n')
            recent_lines = '\n'.join(lines[-50:])
            st.text_area("Log preview (last 50 lines):", recent_lines, height=300, disabled=True)
        else:
            st.info("No batch processor logs found yet. Processing will begin when files are added.")
    except Exception as e:
        st.warning(f"Could not read logs: {e}")


# ---------- MAIN APP ----------
def main():
    # Initialize SQLite database
    init_sqlite_db()
    
    # Initialize session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = []
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = pd.DataFrame()
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    # Header with Pakistan theme
    st.markdown('<div class="pakistan-flag">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header"> Bank Transaction Parser</h1>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **Process one Pakistani bank transaction slip at a time**
    
    Upload a single slip, extract details, and add to your collection.
    """)
    
    # Current Slip Processing Section
    st.markdown("### Step 1: Upload Single Slip")
    
    # Upload box
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=False,
        help="Upload ONE image or PDF of a Pakistani bank transaction slip",
        key="single_upload"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # Store current file
        st.session_state.current_file = uploaded_file
        
        # Preview current file
        st.markdown("### File Preview")
        st.markdown('<div class="file-preview">', unsafe_allow_html=True)
        
        file_col1, file_col2 = st.columns([2, 1])
        
        with file_col1:
            if uploaded_file.type == "application/pdf" and PDF_SUPPORT:
                pdf_images = convert_pdf_to_images(uploaded_file)
                if pdf_images:
                    st.image(pdf_images[0], use_container_width=True)
            else:
                st.image(uploaded_file, use_container_width=True)
        
        with file_col2:
            st.markdown("**File Information:**")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Type:** {uploaded_file.type.split('/')[-1].upper()}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        st.markdown("### Step 2: Extract Data")
        
        if st.button("Process This Slip", type="primary", use_container_width=True):
            with st.spinner("Extracting transaction details..."):
                response = call_local_model_with_image(uploaded_file)
                
                if response:
                    print(response)
                    extracted_data = extract_json_from_response(response)
                    extracted_data['FileName'] = uploaded_file.name
                    extracted_data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Update session state and display first
                    st.session_state.current_result = extracted_data
                    st.session_state.extracted_data.append(extracted_data)
                    st.session_state.dataframe = create_dataframe(st.session_state.extracted_data)
                    
                    st.success(f"✅ Transaction extracted! Total: {len(st.session_state.extracted_data)}")
                    
                    # Display the extracted data immediately
                    st.markdown("### 🎯 Extracted Transaction")
                    display_single_transaction(extracted_data)
                    
                    # Store to SQLite database in background (after display)
                    try:
                        insert_transaction_to_db(extracted_data)
                    except Exception as e:
                        st.warning(f"Data displayed but database save failed: {str(e)}")
                else:
                    st.error("Failed to extract data. Please try again.")
    

    
    # Display extracted data with table and summary
    if not st.session_state.dataframe.empty:
        st.markdown("---")
        st.markdown(f"### 📋 Extracted Transactions ({len(st.session_state.dataframe)} total)")
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Transactions", len(st.session_state.dataframe))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_banks = st.session_state.dataframe['bankName'].nunique()
            st.metric("Different Banks", unique_banks)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            try:
                amounts = st.session_state.dataframe['Amount'].apply(clean_amount)
                total_amount = amounts.sum()
                st.metric("Total Amount", f"PKR {total_amount:,.0f}")
            except:
                st.metric("Total Amount", "PKR -")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            complete = st.session_state.dataframe.apply(
                lambda x: all(x[f] != "Not Found" for f in ['bankName', 'Date', 'Amount']), 
                axis=1
            ).sum()
            st.metric("Complete Records", complete)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display all transactions in cards
        st.markdown("#### Transaction Details")
        for i, row in st.session_state.dataframe.iterrows():
            display_single_transaction(row.to_dict())
        
        # Data table view
        st.markdown("#### Data Table View")
        display_df = st.session_state.dataframe.copy()
        if 'ProcessedDate' in display_df.columns and 'FileName' in display_df.columns:
            display_df = display_df.drop(columns=['ProcessedDate', 'FileName'])
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export section
        st.markdown("---")
        st.markdown("### Export Your Data")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            st.download_button(
                label="Download as CSV",
                data=export_to_csv(st.session_state.dataframe),
                file_name="pakistan_transactions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            excel_data = export_to_excel(st.session_state.dataframe)
            if excel_data:
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="pakistan_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with export_col3:
            if st.button("Clear All Data", type="secondary", use_container_width=True):
                st.session_state.extracted_data = []
                st.session_state.dataframe = pd.DataFrame()
                st.session_state.current_file = None
                st.session_state.current_result = None
                st.success("All data cleared!")
                st.rerun()

                st.session_state.dataframe = pd.DataFrame()
                st.session_state.current_file = None
                st.session_state.current_result = None
                st.success("All data cleared!")
                st.rerun()

def view_database():
    """View all saved transactions from SQLite database"""
    
    # Initialize SQLite database
    init_sqlite_db()
    
    # Header
    st.markdown('<div class="pakistan-flag">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">📊 Saved Transactions Database</h1>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("View all transaction records saved in the database")
    
    try:
        # Get database path
        db_path = db_manager.get_db_path()
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all transactions
        cursor.execute('SELECT * FROM transactions ORDER BY CreatedAt DESC')
        rows = cursor.fetchall()
        
        if not rows:
            st.info("📭 No transactions saved yet. Start processing slips to populate the database!")
            conn.close()
            return
        
        # Convert to DataFrame
        columns = [description[0] for description in cursor.description]
        df = pd.DataFrame([dict(row) for row in rows], columns=columns)
        
        # Display summary metrics
        st.markdown("### 📈 Database Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_banks = df['bankName'].nunique()
            st.metric("Unique Banks", unique_banks)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Filter and Search Section
        st.markdown("### 🔍 Filter & Search")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            search_term = st.text_input("Search by bank name, account, or transaction ID", "")
        
        with filter_col2:
            date_from = st.date_input("From Date", value=None)
        
        with filter_col3:
            date_to = st.date_input("To Date", value=None)
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_term:
            search_lower = search_term.lower()
            filtered_df = filtered_df[
                filtered_df.astype(str).apply(
                    lambda x: x.str.contains(search_lower, case=False).any(), axis=1
                )
            ]
        
        if date_from:
            try:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['Date'], format='%d/%m/%Y', errors='coerce') >= pd.Timestamp(date_from)]
            except:
                pass
        
        if date_to:
            try:
                filtered_df = filtered_df[pd.to_datetime(filtered_df['Date'], format='%d/%m/%Y', errors='coerce') <= pd.Timestamp(date_to)]
            except:
                pass
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        # Data table view with all columns
        st.markdown("### 📊 Transactions Table")
        
        # Create display dataframe
        display_df = filtered_df.copy()
        
        # Reorder columns for better visibility
        column_order = [
            'bankName', 'Date', 'TransactionID', 'Amount', 
            'FromAccount', 'FromAccountNumber', 'FromBankName',
            'ToAccount', 'ToAccountNumber', 'ToBankName',
            'Branch', 'PaymentMode', 'CustomerID', 'ChequeNo', 'Remarks'
        ]
        
        existing_cols = [col for col in column_order if col in display_df.columns]
        display_df = display_df[existing_cols]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.TextColumn(width="medium"),
                "Remarks": st.column_config.TextColumn(width="large"),
            }
        )
        
        # Export section
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            # Export filtered as CSV
            csv_data = filtered_df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="Download Filtered as CSV",
                data=csv_data,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # Export all as CSV
            csv_all_data = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="Download All as CSV",
                data=csv_all_data,
                file_name=f"all_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col3:
            # Export filtered as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Transactions')
            excel_data = output.getvalue()
            st.download_button(
                label="Download Filtered as Excel",
                data=excel_data,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with export_col4:
            # Export all as Excel
            output_all = io.BytesIO()
            with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Transactions')
            excel_all_data = output_all.getvalue()
            st.download_button(
                label="Download All as Excel",
                data=excel_all_data,
                file_name=f"all_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Database management section
        st.markdown("---")
        st.markdown("### ⚙️ Database Management")
        
        db_info_col1, db_info_col2 = st.columns(2)
        
        with db_info_col1:
            st.info(f"""
            **Database Location:** 
            `{db_path}`
            
            **Total Records:** {len(df)}
            **Database Size:** {os.path.getsize(db_path) / 1024 / 1024:.2f} MB
            """)
        
        with db_info_col2:
            if st.button("🗑️ Delete All Records", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    try:
                        cursor.execute('DELETE FROM transactions')
                        conn.commit()
                        st.success("✅ All records deleted!")
                        st.session_state.confirm_delete = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting records: {e}")
                else:
                    st.session_state.confirm_delete = True
                    st.warning("⚠️ Click again to confirm deletion of ALL records")
        
        conn.close()
        
    except Exception as e:
        st.error(f"❌ Error accessing database: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Run the app
if __name__ == "__main__":
    # Create navigation
    st.sidebar.markdown("# Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Process Transactions", "View Database", "Auto Invocation Feature"],
        label_visibility="collapsed"
    )
    
    if page == "Process Transactions":
        main()
    elif page == "View Database":
        view_database()
    else:
        batch_settings_page()