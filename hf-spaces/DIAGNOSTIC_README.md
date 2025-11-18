# 🔍 Diagnostic Guide: Timeout vs Memory

## How to identify the problem?

### 1️⃣ Run the diagnostic tool

In your HF Space, execute:

```bash
python hf-spaces/diagnostic_tool.py
```

This tool will tell you **exactly** if the problem is:
- ❌ **MEMORY_ERROR**: The system ran out of RAM
- ⏰ **TIMEOUT_ERROR**: The operation took too long
- ❓ **OTHER_ERROR**: Another type of problem

### 2️⃣ Interpret the results

#### If you see "MEMORY_ERROR":
```
❌ PROBLEM DETECTED: OUT OF MEMORY
Memory used at failure: 15.8 GB (98.5%)
```

**Cause**: The model is too large for the available memory in HF Spaces.

**Solutions**:
1. **Use smaller models** (1B-1.7B parameters)
2. **Upgrade to HF Spaces PRO** (more RAM available)
3. **Use int8 quantization** (reduces memory usage ~50%)
4. **Load models with `low_cpu_mem_usage=True`**

#### If you see "TIMEOUT_ERROR":
```
⏰ TIMEOUT ERROR after 298.5s
Memory used: 8.2 GB (51.2%)
```

**Cause**: The model takes too long to load, but there is available memory.

**Solutions**:
1. **Increase timeout** from 300s to 600s or 900s
2. **Cache pre-loaded models** at startup
3. **Use faster models**

## 🛠️ Implemented Solutions

### Solution 1: Increase Timeout (Easy)

Edit `hf-spaces/optipfair_frontend.py`:

```python
# Change from:
response = requests.post(url, json=payload, timeout=300)

# To:
response = requests.post(url, json=payload, timeout=600)  # 10 minutes
```

### Solution 2: Use Quantization (For memory issues)

Edit model loading code in the backend:

```python
from transformers import AutoModel, BitsAndBytesConfig

# Configure int8 quantization (reduces memory usage ~50%)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModel.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
```

### Solution 3: Model Cache (For timeout)

Pre-load models at startup in `hf-spaces/app.py`:

```python
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# Global model cache
MODEL_CACHE = {}

def preload_models():
    """Pre-load common models at startup"""
    common_models = [
        "meta-llama/Llama-3.2-1B",
        "oopere/pruned40-llama-3.2-1B",
    ]
    
    logger.info("🔄 Pre-loading common models...")
    for model_name in common_models:
        try:
            logger.info(f"  Loading {model_name}...")
            MODEL_CACHE[model_name] = {
                "model": AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True),
                "tokenizer": AutoTokenizer.from_pretrained(model_name)
            }
            logger.info(f"  ✓ {model_name} loaded")
        except Exception as e:
            logger.warning(f"  ✗ Could not pre-load {model_name}: {e}")
    
    logger.info("✅ Pre-loading complete")

def main():
    # Pre-load models before starting services
    preload_models()
    
    # Rest of the code...
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    # ...
```

### Solution 4: Improved Error Messages

Better error messages are already included to help you identify the problem:

```python
except requests.exceptions.Timeout:
    return (
        None,
        "❌ **Timeout Error:**\nThe model took too long to load (>5min). "
        "This is normal with large models. Options:\n"
        "1. Try with a smaller model\n"
        "2. Wait and try again (model may be caching)\n"
        "3. Contact admin to increase timeout",
        ""
    )

except MemoryError:
    return (
        None,
        "❌ **Memory Error:**\nNot enough RAM for this model. Options:\n"
        "1. Use a smaller model (1B parameters)\n"
        "2. Model requires more memory than available in HF Spaces",
        ""
    )
```

## 📊 Model Size Comparison

| Model | Parameters | RAM Needed* | Load Time** |
|--------|-----------|----------------|----------------|
| Llama-3.2-1B | 1B | ~4 GB | ~30s |
| Llama-3.2-3B | 3B | ~12 GB | ~90s |
| Llama-3-8B | 8B | ~32 GB | ~240s |
| Llama-3-70B | 70B | ~280 GB | ~600s+ |

*Without quantization, FP32
**On typical HF Spaces hardware

## 🎯 Recommended Action Plan

1. **Run the diagnostic**:
   ```bash
   python hf-spaces/diagnostic_tool.py
   ```

2. **Read the results** and follow the specific recommendations

3. **Apply the appropriate solution**:
   - If timeout → Increase timeout or use cache
   - If memory → Use small models or quantization

4. **Test again** with the adjusted configuration

## 📝 Useful Logs in HF Spaces

Check the logs in HF Spaces for messages like:

```
🔍 MODEL LOADING DIAGNOSTIC: meta-llama/Llama-3.2-1B
📊 INITIAL SYSTEM STATE:
  - Available memory: 12.50 GB
  - Used memory: 3.45 GB (21.6%)
⏳ Starting model loading (timeout: 300s)...
  [1/2] Loading tokenizer...
  ✓ Tokenizer loaded in 2.31s
  - Memory used: 3.48 GB (21.8%)
  [2/2] Loading model...
  ✓ Model loaded in 45.67s
✅ LOADING SUCCESSFUL in 47.98s
```

This tells you exactly how much memory and time each step uses.
