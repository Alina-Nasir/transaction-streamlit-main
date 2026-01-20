
import os

# Set Hugging Face cache directory to D drive
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import tempfile

def test_qwen_inference():
    print("Testing Qwen3-VL-2B-Instruct inference on CPU...")
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    try:
        # Load Model with memory-efficient settings
        print(f"Loading model {model_name}...")
        print("Using memory-efficient loading (this may take a few minutes)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 instead of float32 to save memory
            device_map="cpu",
            low_cpu_mem_usage=True  # Enable memory-efficient loading
        )
        processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully.")

        # Find a sample image
        image_dir = "picture_data"
        if os.path.exists(image_dir):
            files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_image_path = os.path.join(image_dir, files[0])
                print(f"Using sample image: {sample_image_path}")
                
                # Create a prompt similar to the app
                prompt = "Extract all text from this image."
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": sample_image_path,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Inference
                print("Preparing inputs...")
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)

                print("Generating...")
                generated_ids = model.generate(**inputs, max_new_tokens=100)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                print("\nGenerated Output:")
                print(output_text)
                print("\nTest PASSED.")
            else:
                print("No images found in 'All Formats'. Skipping inference.")
        else:
            print(f"Directory '{image_dir}' not found. Skipping inference.")

    except Exception as e:
        print(f"\nTest FAILED with error: {e}")

if __name__ == "__main__":
    test_qwen_inference()
