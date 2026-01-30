import os
import base64
import time
import requests
from PIL import Image
import io
import json
import uuid

def resize_image_to_512p(image_path):
    """
    Resize image to 512p (512 max dimension) while maintaining aspect ratio
    
    Args:
        image_path: Path to image
    
    Returns:
        PIL Image object and new dimensions
    """
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    max_dimension = 1024
    
    # Calculate new dimensions maintaining aspect ratio
    if original_width > original_height:
        scale_ratio = max_dimension / original_width
    else:
        scale_ratio = max_dimension / original_height
    
    if scale_ratio >= 1:  # Original is already smaller or equal
        return image, original_width, original_height
    
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    # Resize with high-quality resampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, new_width, new_height


def encode_image_to_base64(image):
    """Encode PIL Image to base64 JPEG"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def main():
    # Configuration
    server_url = "http://localhost:8080/v1/chat/completions"
    image_dir = "picture_data"
    
    print("\n" + "="*70)
    print("🚀 512p RESOLUTION INFERENCE TEST")
    print("="*70)
    
    # Health check
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        print(f"✅ Server running!")
    except:
        print("❌ Server not running! Run start_llama_server.bat first")
        return
    
    # Find image
    if not os.path.exists(image_dir):
        print(f"❌ Create '{image_dir}' folder with JPG/PNG images")
        return
    
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("❌ No images in 'picture_data'")
        return
    
    sample_image_path = os.path.join(image_dir,"0c491e04-4440-4891-8332-0788c70c8a99.jpeg")
    print(f"\n📷 Image: {sample_image_path}")
    
    # Load original image to get original dimensions
    original_image = Image.open(sample_image_path).convert('RGB')
    original_width, original_height = original_image.size
    print(f"📐 Original Size: {original_width}x{original_height}")
    
    # Resize to 512p
    print("\n🔄 Resizing to 512p (max dimension)...")
    resized_image, width, height = resize_image_to_512p(sample_image_path)
    print(f"✅ Resized to: {width}x{height}")
    
    # Calculate compression ratio
    original_pixels = original_width * original_height
    new_pixels = width * height
    compression_ratio = original_pixels / new_pixels
    print(f"📊 Compression: {compression_ratio:.2f}x fewer pixels")
    
    # Encode to base64
    print("\n📝 Encoding image to base64...")
    img_base64 = encode_image_to_base64(resized_image)
    print(f"✅ Encoded ({len(img_base64)/1024:.1f} KB)")
    
    # Prepare request
    unique_id = str(uuid.uuid4())[:8]
    prompt="""
       ### Task: OCR Pakistani bank slip to JSON.
Rules:
1. Date: Find Date/Trx/Posted/Value. Convert to DD/MM/YYYY.
2. Separation: "From" (Sender/Debit/Payer) is DISTINCT from "To" (Receiver/Credit/Beneficiary). Never mix them.
3. Accuracy: Extract VISIBLE text only. Use "Not Found" for missing fields. Do not guess bank names.

### JSON Output:
{
"bankName": "Top header/Logo text",
"Date": "DD/MM/YYYY",
"TransactionID": "Ref/Doc/Chq No",
"Amount": "Value with PKR",
"FromAccount": "Sender Name",
"FromAccountNumber": "Sender Account No",
"FromBankName": "Sender Bank",
"ToAccount": "Receiver Name",
"ToAccountNumber": "Receiver Account No",
"ToBankName": "Receiver Bank",
"Branch": "Branch Name/Code",
"PaymentMode": "Online/Cash/Cheque",
"CustomerID": "ID if visible",
"ChequeNo": "Cheque No",
"Remarks": "Notes"
}
    """
    payload = {
        "model": "qwen3-vl-2b-instruct-Q3_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.7,
        "top_k": 40,
        "cache_prompt": False  # Disable KV cache
    }
    
    # Run inference
    print("\n" + "="*70)
    print("⏱️  STARTING INFERENCE...")
    print("="*70)
    print("⚠️  Note: This may take 1-3 minutes for CPU processing")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for CPU image processing
        )
        inference_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ INFERENCE COMPLETE")
        print(f"{'='*70}")
        print(f"⏱️  Total Time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
        print(f"📊 Status Code: {response.status_code}")
        print(f"{'='*70}\n")
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                output_text = result['choices'][0]['message']['content']
                
                print("📄 COMPLETE OUTPUT:")
                print("-" * 70)
                print(output_text)
                print("-" * 70)
                
                print(f"\n✅ SUCCESS! 512p inference completed.")
                
            else:
                print("❌ No response content")
                print(json.dumps(result, indent=2))
        else:
            print("❌ ERROR:")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Server overloaded or model too slow")
    except Exception as e:
        print(f"❌ FAILED: {e}")


if __name__ == "__main__":
    main()
