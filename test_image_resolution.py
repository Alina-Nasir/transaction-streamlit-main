import os
import base64
import time
import requests
from PIL import Image
import io
import json
import uuid

def resize_image(image_path, max_width=None, max_height=None):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image_path: Path to image
        max_width: Maximum width (None = no limit)
        max_height: Maximum height (None = no limit)
    
    Returns:
        PIL Image object and new dimensions
    """
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    if max_width is None and max_height is None:
        return image, original_width, original_height
    
    # Calculate new dimensions maintaining aspect ratio
    width_ratio = max_width / original_width if max_width else float('inf')
    height_ratio = max_height / original_height if max_height else float('inf')
    
    # Use the smaller ratio to maintain aspect ratio
    scale_ratio = min(width_ratio, height_ratio)
    
    if scale_ratio >= 1:  # Original is already smaller
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


def test_resolution(server_url, image_base64, resolution_name, image_width, image_height):
    """Test inference with specific image resolution"""
    
    # Add unique prompt to prevent caching across tests
    unique_id = str(uuid.uuid4())[:8]
    
    payload = {
        "model": "qwen3-vl-2b-instruct-Q3_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extract all text from this image in detail. [ID: {unique_id}]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.7,
        "top_k": 40,
        "cache_prompt": False  # Disable KV cache to prevent caching effects
    }
    
    print(f"\n{'='*70}")
    print(f"🔍 Testing: {resolution_name} ({image_width}x{image_height})")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes for CPU image processing
        )
        inference_time = time.time() - start_time
        
        print(f"⏱️  Inference Time: {inference_time:.2f}s")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                output_text = result['choices'][0]['message']['content']
                
                # Show first 200 chars of output
                preview = output_text[:200].replace('\n', ' ')
                print(f"📄 Output Preview: {preview}...")
                print(f"✅ SUCCESS!")
                
                return {
                    "resolution": resolution_name,
                    "dimensions": f"{image_width}x{image_height}",
                    "time": inference_time,
                    "output": output_text
                }
            else:
                print("❌ No response content")
                return None
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text[:200])
            return None
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Server overloaded or model too slow")
        return None
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def main():
    # Configuration
    server_url = "http://localhost:8080/v1/chat/completions"
    image_dir = "picture_data"
    
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
    
    sample_image_path = os.path.join(image_dir, files[21])
    print(f"\n📷 Testing Image: {sample_image_path}")
    
    # Load original image to get original dimensions
    original_image = Image.open(sample_image_path).convert('RGB')
    original_width, original_height = original_image.size
    print(f"📐 Original Size: {original_width}x{original_height}")
    
    # Test different resolutions
    resolutions = [
        ("Original", None, None),
        ("1280p (max dimension)", 1280, 1280),
        ("1024p (max dimension)", 1024, 1024),
        ("768p (max dimension)", 768, 768),
        ("512p (max dimension)", 512, 512),
    ]
    
    results = []
    
    for resolution_name, max_width, max_height in resolutions:
        # Resize image
        resized_image, width, height = resize_image(sample_image_path, max_width, max_height)
        
        # Encode to base64
        img_base64 = encode_image_to_base64(resized_image)
        
        # Test this resolution
        result = test_resolution(server_url, img_base64, resolution_name, width, height)
        if result:
            results.append(result)
        else:
            print(f"⚠️  Skipping due to error")
        
        # Wait between tests to clear any caches
        # This ensures each test measures true latency, not cached results
        print("⏳ Waiting 2 seconds before next test (cache clearing)...")
        time.sleep(2)
    
    # Summary
    if results:
        print(f"\n\n{'='*70}")
        print("📊 SUMMARY - LATENCY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Resolution':<25} {'Dimensions':<15} {'Time (s)':<12} {'vs Original':<15}")
        print("-" * 70)
        
        original_time = results[0]['time']
        
        for result in results:
            speedup = original_time / result['time']
            speedup_str = f"{speedup:.2f}x faster" if speedup > 1 else f"{result['time']/original_time:.2f}x slower"
            print(f"{result['resolution']:<25} {result['dimensions']:<15} {result['time']:<12.2f} {speedup_str:<15}")
        
        print("=" * 70)
        print("\n📝 Recommendations:")
        print("- If 768p shows <10% accuracy loss: good tradeoff for 2-3x speedup")
        print("- If 512p shows similar accuracy: best speed/quality balance")
        print("- If all resolutions show same accuracy: use 512p for best latency")
    
    else:
        print("❌ No successful tests - check server and image")


if __name__ == "__main__":
    main()
