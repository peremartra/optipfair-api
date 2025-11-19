# 🔍 Timeout vs Memory Diagnostic Tools

## Overview

When working with heavy models in HF Spaces, you may encounter issues that could be caused by:
1. **Timeout**: The model takes too long to load (>5 minutes)
2. **Memory**: The system runs out of RAM
3. **Both**: A combination of both issues

This toolkit helps you identify and fix the exact problem.

## 📁 Files Added

### 1. `diagnostic_tool.py`
**Purpose**: Identify if the problem is timeout or memory

**Usage**:
```bash
python hf-spaces/diagnostic_tool.py
```

**What it does**:
- Monitors system memory in real real-time
- Tracks model loading time
- Detects the exact failure point
- Provides specific recommendations

**Output**:
```
🔍 MODEL LOADING DIAGNOSTIC: meta-llama/Llama-3.2-1B
📊 INITIAL SYSTEM STATE:
  - Available memory: 12.50 GB
  - Used memory: 3.45 GB (21.6%)
⏳ Starting model loading (timeout: 300s)...
  [1/2] Loading tokenizer...
  ✓ Tokenizer loaded in 2.31s
  [2/2] Loading model...
  ✓ Model loaded in 45.67s
✅ LOADING SUCCESSFUL in 47.98s

💡 RECOMMENDATIONS
✅ Model loaded successfully.
```

### 2. `config_optimized.py`
**Purpose**: Smart configuration based on model size

**Features**:
- Auto-detects model size category (small/medium/large)
- Provides optimized timeout settings
- Recommends appropriate HF Spaces tier
- Warns about memory issues before loading

**Usage**:
```python
from config_optimized import HFSpacesConfig, get_optimized_request_config

# Get optimal timeout for a model
timeout = HFSpacesConfig.get_timeout_for_model("meta-llama/Llama-3.2-1B")

# Get full request config
config = get_optimized_request_config("meta-llama/Llama-3.2-1B")
response = requests.post(url, json=payload, **config)

# Check if model is recommended for your tier
is_ok = HFSpacesConfig.is_model_recommended("meta-llama/Llama-3.2-1B", tier="free")
```

### 3. `DIAGNOSTIC_README.md`
**Purpose**: Complete guide with solutions

**Contents**:
- How to identify timeout vs memory issues
- Step-by-step solutions for each problem
- Model size comparison table
- Code examples for fixes
- Best practices

### 4. Improved Error Messages in `optipfair_frontend.py`
**What changed**:
- More informative timeout error messages
- Explicit memory error detection
- Actionable recommendations in errors
- All messages in English

**Example**:
```
❌ **Timeout Error:**
The request exceeded 5 minutes (300s).

**Possible causes:**
1. The model is very large and takes long to load
2. The server is processing many requests

**Solutions:**
• Use a smaller model (1B parameters)
• Wait and try again (model may be caching)
• If it persists, run `diagnostic_tool.py` for more information
```

## 🚀 Quick Start Guide

### Step 1: Diagnose the Problem
```bash
cd hf-spaces
python diagnostic_tool.py
```

### Step 2: Read the Output
The tool will tell you:
- ✅ **Success**: Model loads fine
- ❌ **MEMORY_ERROR**: Need more RAM or smaller model
- ⏰ **TIMEOUT_ERROR**: Need more time or faster model

### Step 3: Apply the Solution

#### For TIMEOUT problems:
```python
# Option 1: Increase timeout in optipfair_frontend.py
response = requests.post(
    url,
    json=payload,
    timeout=600  # Change from 300 to 600 seconds
)

# Option 2: Use config_optimized.py
from config_optimized import get_optimized_request_config
config = get_optimized_request_config(model_name)
response = requests.post(url, json=payload, **config)
```

#### For MEMORY problems:
```python
# Option 1: Use smaller model
AVAILABLE_MODELS = [
    "meta-llama/Llama-3.2-1B",  # ✅ Works on free tier
    "oopere/pruned40-llama-3.2-1B",  # ✅ Works on free tier
]

# Option 2: Use quantization (in backend)
from transformers import AutoModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
)

# Option 3: Upgrade HF Spaces tier
# Free: 16GB RAM → PRO: 32GB RAM → Enterprise: 64GB RAM
```

## 📊 Model Recommendations by Tier

### Free Tier (16GB RAM)
✅ **Recommended**:
- meta-llama/Llama-3.2-1B (~4 GB, ~30s load)
- oopere/pruned40-llama-3.2-1B (~4 GB, ~30s load)
- google/gemma-3-1b-pt (~4 GB, ~30s load)
- Qwen/Qwen3-1.7B (~6 GB, ~45s load)

⚠️ **May work with optimization**:
- meta-llama/Llama-3.2-3B (~12 GB, ~90s load)

❌ **Won't work**:
- meta-llama/Llama-3-8B (~32 GB)
- meta-llama/Llama-3-70B (~280 GB)

### PRO Tier (32GB RAM)
✅ **Additional models**:
- meta-llama/Llama-3.2-3B
- meta-llama/Llama-3-8B (with quantization)

### Enterprise Tier (64GB RAM)
✅ **Additional models**:
- meta-llama/Llama-3-8B (full precision)
- Larger models with quantization

## 🎯 Common Scenarios

### Scenario 1: "My model times out after 5 minutes"
**Diagnosis**: TIMEOUT_ERROR

**Solution**:
1. Check if model is too large for your tier
2. Increase timeout to 600s (10 minutes)
3. Consider pre-loading models at startup

### Scenario 2: "Process crashes without clear error"
**Diagnosis**: Likely MEMORY_ERROR (Out-Of-Memory kills the process)

**Solution**:
1. Run `diagnostic_tool.py` to confirm
2. Use smaller model (1B parameters)
3. Use int8 quantization
4. Upgrade to PRO tier

### Scenario 3: "Sometimes works, sometimes doesn't"
**Diagnosis**: Memory pressure or concurrent requests

**Solution**:
1. Implement model caching
2. Add memory monitoring
3. Use smaller default model

## 🛠️ Advanced: Pre-loading Models

To avoid timeout on first request, pre-load models at startup:

```python
# In hf-spaces/app.py
from transformers import AutoModel, AutoTokenizer

MODEL_CACHE = {}

def preload_models():
    """Pre-load common models at startup"""
    models = ["meta-llama/Llama-3.2-1B"]
    
    for model_name in models:
        try:
            print(f"Pre-loading {model_name}...")
            MODEL_CACHE[model_name] = {
                "model": AutoModel.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True
                ),
                "tokenizer": AutoTokenizer.from_pretrained(model_name)
            }
            print(f"✓ {model_name} ready")
        except Exception as e:
            print(f"✗ Could not pre-load {model_name}: {e}")

def main():
    preload_models()  # Load models before starting services
    # ... rest of startup code
```

## 📞 Support

If you still have issues after trying these solutions:

1. Check the full diagnostic output
2. Review HF Spaces logs
3. Verify your HF Spaces tier and limits
4. Consider using a different model architecture

## 📝 Summary

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Timeout** | Request > 5 min | Increase timeout, use cache |
| **Memory** | Process crashes/kills | Smaller model, quantization, upgrade tier |
| **Both** | Slow + crashes | Smaller model + longer timeout |

All tools are designed to help you quickly identify and fix the exact problem without guessing.
