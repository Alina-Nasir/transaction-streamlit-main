# 🚀 Qwen3-VL Setup Guide with llama.cpp Server

This guide will help you set up Qwen3-VL vision model inference using llama.cpp **server mode** (no compilation needed!) for Pakistani bank transaction parsing.

**✨ NEW: Simplified approach using llama.cpp server - no Python compilation required!**

---

## 📋 Prerequisites

- **Windows 10/11**
- **16GB+ RAM** recommended for smooth inference
- **CPU with AVX2 support** (most modern CPUs)
- **Python 3.8+**

---

## 🔧 Step 1: Download llama.cpp (Pre-built Binary)

### 1.1 Download llama.cpp Release
1. Go to: https://github.com/ggerganov/llama.cpp/releases/latest
2. Download the Windows build (look for `llama-*-bin-win-avx2-x64.zip`)
3. Extract to: `C:\llama-cpu\`

**Result:** You should have `C:\llama-cpu\llama-server.exe`

---

## 📦 Step 2: Install Python Dependencies (No Compilation!)

### 2.1 Create Virtual Environment
```bash
# Navigate to your project directory
cd "D:\Machine Learning\JFF JOB\transaction-streamlit-main"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 2.2 Install Requirements (Simple!)
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements (no compilation needed!)
pip install -r requirements.txt
```

**That's it! No C++ compiler needed! 🎉**

---

## 🎯 Step 3: Verify Model Files

### 3.1 Main Model (Already Present)
✅ You already have: `qwen3-vl-2b-instruct-q2_k.gguf`

### 3.2 Vision Support (Optional but Recommended)
For vision capabilities, download the mmproj file:

**Download:**
- URL: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/main
- File: `qwen3-vl-2b-instruct-mmproj-f16.gguf` (~1.7GB)
- Place in: Project root directory

**Note:** The server will work without this, but vision capabilities will be limited.

---

## 🚀 Step 4: Start llama.cpp Server

### Option A: Use Batch File (Easiest)
```bash
# Double-click or run in terminal:
start_llama_server.bat
```

### Option B: Manual Command
```bash
# Open a new terminal
cd C:\llama-cpu

# Start server
llama-server.exe -m "D:\Machine Learning\JFF JOB\transaction-streamlit-main\qwen3-vl-2b-instruct-q2_k.gguf" --host localhost --port 8080 -c 4096 -t 8 --n-gpu-layers 0
```

**Expected Output:**
```
llama server listening at http://localhost:8080
```

**Keep this terminal open!** The server needs to run in the background.

---

## ✅ Step 5: Test Your Setup

### 5.1 In a NEW terminal:
```bash
cd "D:\Machine Learning\JFF JOB\transaction-streamlit-main"
venv\Scripts\activate
python verify_model.py
```

**Expected Output:**
```
Testing Qwen3-VL via llama.cpp server...
✅ Model found: qwen3-vl-2b-instruct-q2_k.gguf
✅ llama.cpp server is running!
📷 Using sample image: picture_data/sample.jpg
⏱️  Starting inference timer...
============================================================
⏱️  INFERENCE TIME MEASUREMENT
============================================================
Duration: 2.34 seconds
Duration: 2340 milliseconds
============================================================
📄 Generated Output:
------------------------------------------------------------
[Extracted text appears here]
------------------------------------------------------------
✅ Test PASSED.
```

---

## 🚀 Step 6: Run the Streamlit App

```bash
# Make sure server is running in another terminal
# Then in your project terminal:
venv\Scripts\activate
streamlit run streamlit_app.py
```

---

## ⚡ Performance Optimization Tips

### CPU Optimization
- **Use all cores**: The script automatically uses `os.cpu_count()` threads
- **Batch size**: Adjust `n_batch` in verify_model.py (512 is optimal for most CPUs)
- **Context size**: Reduce `n_ctx` if you have limited RAM (4096 → 2048)

### For Faster Inference
1. **Use higher quantization** (if available):
   - Q2_K (smallest, fastest, lowest quality) ← You have this
   - Q4_K_M (balanced)
   - Q5_K_M (larger, slower, better quality)

2. **Enable OpenBLAS** (recommended):
   ```bash
   # Reinstall with OpenBLAS support
   pip uninstall llama-cpp-python -y
   CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
   ```

---

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution:** Ensure `qwen3-vl-2b-instruct-q2_k.gguf` is in the project root directory.

### Issue: "Vision encoder (mmproj) file not found"
**Solution:** Download the mmproj file as described in Step 3.2.

### Issue: "ImportError: llama-cpp-python not installed"
**Solution:** 
```bash
venv\Scripts\activate
pip install llama-cpp-python
```

### Issue: Slow inference (>10 seconds)
**Solution:**
1. Reduce `n_ctx` from 4096 to 2048
2. Install OpenBLAS version (see Performance Optimization)
3. Close other applications to free up RAM

### Issue: "Out of memory"
**Solution:**
1. Close other applications
2. Use a smaller quantization (Q2_K is already the smallest)
3. Reduce `n_ctx` to 2048 or 1024

---

## 📊 Expected Performance

With Qwen3-VL-2B Q2_K on a modern CPU:
- **First inference**: 5-10 seconds (model loading)
- **Subsequent inferences**: 2-5 seconds
- **Memory usage**: ~2-3GB RAM

---

## 🔄 Alternative: Using Ollama (Previous Setup)

If you prefer Ollama instead of llama.cpp:

```bash
# Install Ollama from: https://ollama.com/download
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama serve

# Then use the Ollama version of the code
```

---

## 📚 Additional Resources

- **llama.cpp GitHub**: https://github.com/ggerganov/llama.cpp
- **llama-cpp-python Docs**: https://llama-cpp-python.readthedocs.io/
- **Qwen3-VL Models**: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

## ✅ Quick Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created (`venv`)
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `llama-cpp-python` installed
- [ ] Main model file present (`qwen3-vl-2b-instruct-q2_k.gguf`)
- [ ] Vision encoder downloaded (`qwen3-vl-2b-instruct-mmproj-f16.gguf`)
- [ ] Sample images in `picture_data/` folder
- [ ] `verify_model.py` runs successfully

---

**Need help?** Create an issue in the repository with:
- Your Python version
- Operating system
- Error message (full traceback)
- Output of `pip list`
