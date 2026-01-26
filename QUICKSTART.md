# 🚀 Quick Start Guide

## For New Users Setting Up This Project

### Prerequisites
- Windows 10/11
- Python 3.8-3.11
- 16GB+ RAM

---

## 🔥 Fast Track Setup (3 Steps)

### Step 1: Setup Python Environment
```cmd
cd "D:\Machine Learning\JFF JOB\transaction-streamlit-main"
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies

**Option A: Try Quick Install**
```cmd
pip install -r requirements.txt
```

If it fails with C++ compiler error, try:

**Option B: Install Build Tools First**
1. Download: https://visualstudio.microsoft.com/downloads/
2. Install "Desktop development with C++"
3. Restart terminal and retry:
```cmd
pip install -r requirements.txt
```

### Step 3: Download Vision Encoder
Download `qwen3-vl-2b-instruct-mmproj-f16.gguf` from:
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/main

Place it in the project root directory.

---

## ✅ Test Your Setup

```cmd
venv\Scripts\activate
python verify_model.py
```

**Expected:** Inference completes in 2-5 seconds with extracted text.

---

## 🎯 Run the App

```cmd
venv\Scripts\activate
streamlit run streamlit_app.py
```

---

## ❌ If Setup Fails: Use Ollama Instead

```cmd
# Install Ollama from: https://ollama.com/download
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama serve

# Then modify requirements.txt:
# Change: llama-cpp-python → ollama

# Revert verify_model.py to Ollama version
```

---

## 📚 Full Documentation

- **Complete Setup**: See `SETUP_LLAMACPP.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Troubleshooting**: See `SETUP_LLAMACPP.md` → Troubleshooting section

---

## 🆘 Common Issues

### "C++ compiler not found"
**Fix:** Install Visual Studio Build Tools (see Step 2, Option B above)

### "mmproj file not found"
**Fix:** Download it from HuggingFace (see Step 3 above)

### "Model loads but no vision support"
**Fix:** Ensure mmproj file is in the same folder as the .gguf model

### "Out of memory"
**Fix:** Close other applications, reduce `n_ctx` in verify_model.py

---

**Need Help?** Check the full setup guide in `SETUP_LLAMACPP.md`
