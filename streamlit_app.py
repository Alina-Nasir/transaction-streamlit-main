import streamlit as st
import base64
import io
import tempfile
import os
import json
import re
import pandas as pd
from PIL import Image
from openai import OpenAI
import hashlib
from datetime import datetime

# Try to import PDF libraries
try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PyPDFium2 not available - PDF support limited")

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

def get_pakistani_bank_prompt():
    """Return prompt optimized for Pakistani bank transaction slips"""
    return """
    You are an expert in Pakistani bank transaction slip analysis. Extract ALL visible information from this Pakistani bank transaction slip.
    
    IMPORTANT: Focus on PAKISTANI BANKS like Meezan Bank, Habib Metro, United Bank (UBL), Bank Alfalah, MCB Bank, Allied Bank, Standard Chartered, HBL, Faysal Bank, Soneri Bank, Dubai Islamic Bank, Bank Alfalah (Alfa), JS Bank, easypaisa, JazzCash etc.
    
    CRITICAL FOR ACCOUNT NUMBERS: 
    1. Look for account numbers that are partially masked with **** or XXXX patterns. 
    2. Common patterns: ****1234, XXX-XXX-1234, ******5678, XXXX-XXXX-XXXX-1234
    3. Account numbers may appear as: "A/C No: ****5678" or "Account: XXX-XXX-7890" or "Account Number: 1234********"
    4. IMPORTANT: Extract SEPARATE account numbers AND bank names for SENDER and RECEIVER
    
    Extract these SPECIFIC FIELDS:
    1. bankName - Name of the bank issuing this slip (e.g., MEEZAN BANK, HABIB BANK LIMITED, UBL, BANK ALFALAH, HBL)
    2. Date - Transaction date in DD/MM/YYYY format
    3. TransactionID - Transaction reference number, Chq #, Document Code, or any ID
    4. Amount - Transaction amount with currency (PKR) - look for "Actual Amount" or similar
    5. ToAccount - Name of recipient (look for Customer Name, Beneficiary Name)
    6. ToAccountNumber - Recipient's account number (look for "Beneficiary Account", "Credit To", "To A/C", "Receiver Account")
    7. ToBankName - Recipient's bank name (look for "Beneficiary Bank", "Receiver Bank", "Credit Bank")
    8. FromAccount - Name of sender (look for Sender Name, Payer Name)
    9. FromAccountNumber - Sender's account number (look for "Sender Account", "Debit From", "From A/C", "Payer Account")
    10. FromBankName - Sender's bank name (look for "Sender Bank", "Payer Bank", "Debit Bank")
    11. Branch - Branch code or name
    12. PaymentMode - Payment mode (Online, Cash, Cheque, Transfer)
    13. CustomerID - Customer ID or Account number if visible
    14. ChequeNo - Cheque number if present
    15. Remarks - Any additional notes or remarks
    
    SPECIFIC INSTRUCTIONS FOR PAKISTANI SLIPS:
    - Look for fields like: "Branch", "Date", "Chq #", "Customer Name", "Amount", "Account No"
    - For Meezan Bank: Look for green color themes, "MEEZAN BANK" text
    - Amounts are usually in PKR (Pakistani Rupees)
    - Dates are usually in DD/MM/YYYY format
    - Common terms: "Branch", "Customer", "Amount", "Cheque", "Transfer", "Online"
    
    BANK NAME HINTS:
    - Look for labels like: "Sender Bank", "Payer Bank", "From Bank", "Debit Bank"
    - Look for labels like: "Receiver Bank", "Beneficiary Bank", "To Bank", "Credit Bank"
    - Bank names might appear in sections like "Sender Details" or "Beneficiary Details"
    
    ACCOUNT NUMBER HINTS:
    - Sender account usually near: "From", "Debit From", "Sender A/C", "Payer Account"
    - Receiver account usually near: "To", "Credit To", "Beneficiary A/C", "Receiver Account"
    
    RETURN FORMAT:
    Return ONLY valid JSON with these exact field names:
    {
        "bankName": "extracted value or 'Not Found'",
        "Date": "extracted value or 'Not Found'",
        "TransactionID": "extracted value or 'Not Found'",
        "Amount": "extracted value or 'Not Found'",
        "ToAccount": "extracted value or 'Not Found'",
        "ToAccountNumber": "extracted value or 'Not Found'",
        "ToBankName": "extracted value or 'Not Found'",
        "FromAccount": "extracted value or 'Not Found'",
        "FromAccountNumber": "extracted value or 'Not Found'",
        "FromBankName": "extracted value or 'Not Found'",
        "Branch": "extracted value or 'Not Found'",
        "PaymentMode": "extracted value or 'Not Found'",
        "CustomerID": "extracted value or 'Not Found'",
        "ChequeNo": "extracted value or 'Not Found'",
        "Remarks": "extracted value or 'Not Found'"
    }
    
    Extract ONLY what is visible. If field not found, use "Not Found".
    IMPORTANT: Extract account numbers exactly as shown, including **** or XXXX masking.
    """

def call_openai_api_with_image(image_file, prompt=None, model="gpt-4o"):
    """Call OpenAI GPT-4o API with uploaded image/PDF"""
    try:
        # Get API key
        OPENAI_API_KEY = st.secrets.get("API_KEY", "")
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
            if not OPENAI_API_KEY and 'api_key' in st.session_state:
                OPENAI_API_KEY = st.session_state.api_key
        
        if not OPENAI_API_KEY:
            st.error("OpenAI API key not found. Please add it in the sidebar.")
            return None
        
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Use Pakistani bank specific prompt
        effective_prompt = prompt if prompt else get_pakistani_bank_prompt()

        # Check if file is PDF
        if hasattr(image_file, 'type') and image_file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("PDF processing requires pypdfium2. Install with: pip install pypdfium2")
                return None
                
            pdf_images = convert_pdf_to_images(image_file)
            if pdf_images and len(pdf_images) > 0:
                image = pdf_images[0]
            else:
                return None
        else:
            image = Image.open(image_file)
        
        # Convert and optimize image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize large images to save tokens
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def extract_json_from_response(response_text):
    """Extract JSON from API response"""
    try:
        response_text = response_text.strip()
        
        # Find JSON pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            json_str = max(matches, key=len)
            json_str = json_str.replace('\\', '').replace('\n', ' ')
            data = json.loads(json_str)
            
            # Standardize field names
            standardized_data = {}
            field_mappings = {
                'bankName': ['bankName', 'bank', 'bank_name', 'Bank Name'],
                'Date': ['Date', 'date', 'Transaction Date', 'DATE'],
                'TransactionID': ['TransactionID', 'transactionID', 'Transaction ID', 'Reference No', 'Chq #', 'Document Code'],
                'Amount': ['Amount', 'amount', 'Transaction Amount', 'AMOUNT', 'Actual Amount'],
                'ToAccount': ['ToAccount', 'toAccount', 'To Account', 'Beneficiary', 'Customer Name', 'Customer', 'Receiver Name'],
                'ToAccountNumber': ['ToAccountNumber', 'toAccountNumber', 'To Account Number', 'Beneficiary Account', 'Credit To', 'To A/C', 'Receiver Account'],
                'ToBankName': ['ToBankName', 'toBankName', 'To Bank Name', 'Beneficiary Bank', 'Receiver Bank', 'Credit Bank'],
                'FromAccount': ['FromAccount', 'fromAccount', 'From Account', 'Sender', 'Payer', 'Sender Name'],
                'FromAccountNumber': ['FromAccountNumber', 'fromAccountNumber', 'From Account Number', 'Sender Account', 'Debit From', 'From A/C', 'Payer Account'],
                'FromBankName': ['FromBankName', 'fromBankName', 'From Bank Name', 'Sender Bank', 'Payer Bank', 'Debit Bank'],
                'Branch': ['Branch', 'branch', 'BRANCH'],
                'PaymentMode': ['PaymentMode', 'paymentMode', 'Payment Mode', 'Mode', 'PaymentMode'],
                'CustomerID': ['CustomerID', 'customerID', 'Customer ID', 'Customer No'],
                'ChequeNo': ['ChequeNo', 'chequeNo', 'Cheque No', 'Cheque Number', 'Actual Cheque No'],
                'Remarks': ['Remarks', 'remarks', 'Note', 'Description']
            }
            
            for std_field, possible_names in field_mappings.items():
                value_found = False
                for name in possible_names:
                    if name in data:
                        standardized_data[std_field] = str(data[name]).strip()
                        value_found = True
                        break
                if not value_found:
                    standardized_data[std_field] = "Not Found"
            
            return standardized_data
            
        else:
            # Return empty data if JSON not found
            return {
                "bankName": "Not Found",
                "Date": "Not Found", 
                "TransactionID": "Not Found",
                "Amount": "Not Found",
                "ToAccount": "Not Found",
                "ToAccountNumber": "Not Found",
                "ToBankName": "Not Found",
                "FromAccount": "Not Found",
                "FromAccountNumber": "Not Found",
                "FromBankName": "Not Found",
                "Branch": "Not Found",
                "PaymentMode": "Not Found",
                "CustomerID": "Not Found",
                "ChequeNo": "Not Found",
                "Remarks": "Not Found"
            }
            
    except json.JSONDecodeError:
        return {
            "bankName": "Not Found",
            "Date": "Not Found", 
            "TransactionID": "Not Found",
            "Amount": "Not Found",
            "ToAccount": "Not Found",
            "ToAccountNumber": "Not Found",
            "ToBankName": "Not Found",
            "FromAccount": "Not Found",
            "FromAccountNumber": "Not Found",
            "FromBankName": "Not Found",
            "Branch": "Not Found",
            "PaymentMode": "Not Found",
            "CustomerID": "Not Found",
            "ChequeNo": "Not Found",
            "Remarks": "Not Found"
        }

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

# ---------- MAIN APP ----------
def main():
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
                response = call_openai_api_with_image(uploaded_file)
                
                if response:
                    extracted_data = extract_json_from_response(response)
                    extracted_data['FileName'] = uploaded_file.name
                    extracted_data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.current_result = extracted_data
                    st.session_state.extracted_data.append(extracted_data)
                    st.session_state.dataframe = create_dataframe(st.session_state.extracted_data)
                    
                    st.success(f"Transaction saved! Total: {len(st.session_state.extracted_data)}")
                    st.rerun()
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

# Run the app
if __name__ == "__main__":
    main()