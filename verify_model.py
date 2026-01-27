import os
import base64
import time
import requests
from PIL import Image
import io
import json

def test_qwen_inference():    
    # Use OpenAI-compatible chat endpoint (better for vision)
    server_url = "http://localhost:8080/v1/chat/completions"
    
    
    # Health check
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        print(f"✅ Server running!")
    except:
        print("❌ Server not running! Run start_llama_server.bat first")
        return
    
    # Find image
    image_dir = "picture_data"
    if not os.path.exists(image_dir):
        print(f"❌ Create '{image_dir}' folder with JPG/PNG images")
        return
    
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("❌ No images in 'picture_data'")
        return
    
    sample_image_path = os.path.join(image_dir, files[21])
    print(f"📷 Image: {sample_image_path}")
    
    # Load and encode image
    image = Image.open(sample_image_path).convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # OpenAI-compatible chat format
    payload = {
        "model": "qwen3-vl-2b-instruct-Q3_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.7,
        "top_k": 40
    }
    
    print(f"⏱️  Starting inference...")
    print(f"⚠️  Note: First inference may take 2-3 minutes (image encoding on CPU)")
    start_time = time.time()
    
    try:
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for CPU image processing
        )
        inference_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"⏱️  Duration: {inference_time:.2f}s")
        print(f"Status: {response.status_code}")
        print('='*60)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                output_text = result['choices'][0]['message']['content']
                print("\n📄 OUTPUT:")
                print("-" * 60)
                print(output_text)
                print("-" * 60)
                print("✅ SUCCESS!")
            else:
                print("❌ No response content")
                print(json.dumps(result, indent=2))
        else:
            print("❌ ERROR:")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Server overloaded or model loading")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_qwen_inference()
