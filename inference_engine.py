"""
Shared Inference Engine for Pakistani Bank Transaction Parser
Used by both streamlit_app.py and batch_processor_service.py
Ensures consistent image processing and model inference across the application
"""

import base64
import io
import os
import re
import json
import tempfile
import uuid
from PIL import Image
import requests
import logging

# Try to import PDF libraries
try:
    import pypdfium2 as pdfium
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logger = logging.getLogger(__name__)


def get_pakistani_bank_prompt():
    """Return optimized prompt for Pakistani bank transaction slips"""
    return """


   ### Task: OCR Extraction. Return JSON.
Instructions:
- Date: DD/MM/YYYY
- Bank Name: Extract the Logo Text at the top.
- CRITICAL: Check labels "From/Debit" vs "To/Credit".
- If a field is empty in the image, strictly return "Not Found".

{
"bankName": "Logo Title",
"Date": "DD/MM/YYYY",
"TransactionID": "Ref No",
"Amount": "Amount + Currency",
"FromAccount": "Sender Name (Label: From/Debit)",
"FromAccountNumber": "Sender Account # (Label: From/Debit). Write 'Not Found' if not present with the tag of from/Source or similar word.",
"FromBankName": "Sender Bank",
"ToAccount": "Receiver Name (Label: To/Credit)",
"ToAccountNumber": "Receiver Account # (Label: To/Credit). Write 'Not Found' if not present with the tag of To/Destination or similar word.",
"ToBankName": "Receiver Bank",
"Branch": "Branch Name",
"PaymentMode": "Cash/Online/Cheque",
"CustomerID": "Consumer ID",
"ChequeNo": "Cheque #",
"Remarks": "Narration/Remarks"
}

"""


def resize_image_to_512p(image):
    """
    Resize image to 512p (512 max dimension) while maintaining aspect ratio
    
    Args:
        image: PIL Image object
    
    Returns:
        Resized PIL Image object
    """
    max_dimension = 1024
    
    # Calculate new dimensions maintaining aspect ratio
    if max(image.size) <= max_dimension:
        return image
    
    if image.size[0] > image.size[1]:
        scale_ratio = max_dimension / image.size[0]
    else:
        scale_ratio = max_dimension / image.size[1]
    
    new_width = int(image.size[0] * scale_ratio)
    new_height = int(image.size[1] * scale_ratio)
    
    # Resize with high-quality resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized


def encode_image_to_base64(image):
    """Encode PIL Image to base64 JPEG"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def convert_pdf_to_images(pdf_file):
    """Convert PDF file to list of PIL Images using pypdfium2"""
    try:
        if not PDF_SUPPORT:
            logger.warning("PyPDFium2 not available - PDF support unavailable")
            return None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            if isinstance(pdf_file, bytes):
                tmp_file.write(pdf_file)
            else:
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
        logger.error(f"Error converting PDF: {str(e)}")
        return None


def call_local_model_with_image(image_file, prompt=None, server_url="http://localhost:8088/v1/chat/completions", timeout=300):
    """
    Call llama.cpp Qwen3-VL model via HTTP with uploaded image/PDF
    
    Args:
        image_file: PIL Image object, file path string, or bytes
        prompt: Custom prompt (uses default if None)
        server_url: llama.cpp server endpoint
        timeout: Request timeout in seconds
    
    Returns:
        Response text from model or None on failure
    """
    try:
        # Health check
        try:
            requests.get("http://localhost:8088/health", timeout=5)
        except Exception as e:
            logger.error(f"llama.cpp server not running: {e}")
            return None
        
        # Use Pakistani bank specific prompt
        effective_prompt = prompt if prompt else get_pakistani_bank_prompt()
        
        # Handle different input types
        if isinstance(image_file, str):
            # File path
            if image_file.lower().endswith('.pdf'):
                if not PDF_SUPPORT:
                    logger.error("PDF processing requires pypdfium2")
                    return None
                pdf_images = convert_pdf_to_images(image_file)
                if not pdf_images or len(pdf_images) == 0:
                    logger.error("No images extracted from PDF")
                    return None
                image = pdf_images[0]
            else:
                image = Image.open(image_file)
        elif isinstance(image_file, bytes):
            # Bytes data
            if image_file[:4] == b'%PDF':  # PDF signature
                if not PDF_SUPPORT:
                    logger.error("PDF processing requires pypdfium2")
                    return None
                pdf_images = convert_pdf_to_images(image_file)
                if not pdf_images or len(pdf_images) == 0:
                    logger.error("No images extracted from PDF")
                    return None
                image = pdf_images[0]
            else:
                image = Image.open(io.BytesIO(image_file))
        elif hasattr(image_file, 'read'):
            # File-like object (from streamlit uploader)
            image_data = image_file.read()
            if image_data[:4] == b'%PDF':  # PDF signature
                if not PDF_SUPPORT:
                    logger.error("PDF processing requires pypdfium2")
                    return None
                pdf_images = convert_pdf_to_images(image_data)
                if not pdf_images or len(pdf_images) == 0:
                    logger.error("No images extracted from PDF")
                    return None
                image = pdf_images[0]
            else:
                image = Image.open(io.BytesIO(image_data))
        else:
            # Assume PIL Image
            image = image_file
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 512p for faster inference
        image = resize_image_to_512p(image)
        
        # Encode image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Add unique ID to prevent caching
        unique_id = str(uuid.uuid4())[:8]
        
        # Prepare llama.cpp API request
        payload = {
            "model": "qwen3-vl-2b-instruct-Q3_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{effective_prompt} [ID: {unique_id}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.7,
            "top_k": 40,
            "cache_prompt": False
        }
        
        logger.debug(f"Sending inference request to {server_url}")
        
        # Call llama.cpp HTTP server
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                output_text = result['choices'][0]['message']['content']
                logger.debug(f"Inference successful, received {len(output_text)} chars")
                return output_text
            else:
                logger.error("No response content from model")
                return None
        else:
            logger.error(f"llama.cpp Error: {response.status_code} - {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        logger.error("TIMEOUT - Server overloaded or image processing taking too long")
        return None
    except Exception as e:
        logger.error(f"Inference Error: {str(e)}", exc_info=True)
        return None


def extract_json_from_response(response_text):
    """
    Extract JSON from API response and standardize field names
    
    Args:
        response_text: Response text from model
    
    Returns:
        Dictionary with standardized transaction fields
    """
    try:
        if not response_text:
            logger.warning("Empty response text")
            return _get_empty_response()
        
        response_text = response_text.strip()
        
        # Find JSON pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            json_str = max(matches, key=len)
            json_str = json_str.replace('\\', '').replace('\n', ' ')
            data = json.loads(json_str)
            
            # Standardize field names
            standardized_data = _standardize_fields(data)
            logger.debug(f"Successfully extracted JSON with {len(standardized_data)} fields")
            return standardized_data
        else:
            logger.warning("No JSON found in response")
            return _get_empty_response()
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return _get_empty_response()
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}", exc_info=True)
        return _get_empty_response()


def _standardize_fields(data):
    """Standardize incoming field names to expected format"""
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


def _get_empty_response():
    """Return empty standardized response"""
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
