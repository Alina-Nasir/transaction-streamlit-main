import os
import base64
import time
from PIL import Image
from ollama import chat

def test_qwen_inference():
    print("Testing Qwen3-VL via Ollama (http://localhost:11434)...")
    model_name = "qwen3-vl:2b-instruct-q4_K_M"

    try:
        # Find a sample image
        image_dir = "picture_data"
        if os.path.exists(image_dir):
            files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_image_path = os.path.join(image_dir, files[0])
                print(f"Using sample image: {sample_image_path}")
                
                # Load and convert image to base64
                print("Loading image...")
                image = Image.open(sample_image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to base64
                import io
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Create a prompt similar to the app
                prompt = "Extract all text from this image."
                
                # Start timing
                print(f"\nCalling Ollama model {model_name}...")
                print("⏱️  Starting inference timer...")
                start_time = time.time()
                
                # Call Ollama API
                response = chat(
                    model=model_name,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [img_base64]
                        }
                    ]
                )
                
                # End timing
                end_time = time.time()
                inference_duration = end_time - start_time
                
                output_text = response['message']['content']
                
                print("\n" + "="*60)
                print("⏱️  INFERENCE TIME MEASUREMENT")
                print("="*60)
                print(f"Duration: {inference_duration:.2f} seconds ({inference_duration:.3f}s)")
                print(f"Duration: {inference_duration * 1000:.0f} milliseconds")
                print("="*60)
                
                print("\n📄 Generated Output:")
                print("-" * 60)
                print(output_text)
                print("-" * 60)
                print("\n✅ Test PASSED.")
            else:
                print("No images found in 'picture_data'. Skipping inference.")
        else:
            print(f"Directory '{image_dir}' not found. Skipping inference.")

    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        print("Make sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Model is pulled (ollama pull qwen3-vl:2b-instruct-q4_K_M)")
        print("3. Ollama is accessible at http://localhost:11434")

if __name__ == "__main__":
    test_qwen_inference()
