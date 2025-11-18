"""
Diagnostic tool to identify timeout vs memory issues in HF Spaces.
Run this script in HF Spaces to get detailed performance information.
"""

import psutil
import time
import sys
import traceback
from datetime import datetime


def get_memory_info():
    """Gets detailed information about system memory usage."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent_used": memory.percent,
        "free_gb": memory.free / (1024**3),
    }


def get_cpu_info():
    """Gets information about CPU usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
    }


def monitor_model_loading(model_name: str, timeout_seconds: int = 300):
    """
    Monitors model loading and detects if it fails due to timeout or memory.
    
    Args:
        model_name: HuggingFace model name to load
        timeout_seconds: Maximum wait time in seconds
        
    Returns:
        dict with diagnostic information
    """
    print(f"\n{'='*60}")
    print(f"🔍 MODEL LOADING DIAGNOSTIC: {model_name}")
    print(f"{'='*60}\n")

    # Initial system state
    print("📊 INITIAL SYSTEM STATE:")
    mem_before = get_memory_info()
    cpu_before = get_cpu_info()
    print(f"  - Available memory: {mem_before['available_gb']:.2f} GB")
    print(f"  - Used memory: {mem_before['used_gb']:.2f} GB ({mem_before['percent_used']:.1f}%)")
    print(f"  - Available CPUs: {cpu_before['cpu_count']} cores")
    print(f"  - CPU usage: {cpu_before['cpu_percent']:.1f}%")
    
    start_time = time.time()
    result = {
        "model_name": model_name,
        "success": False,
        "error_type": None,
        "error_message": None,
        "elapsed_time": 0,
        "memory_before": mem_before,
        "memory_after": None,
        "memory_peak": mem_before["used_gb"],
        "timeout_seconds": timeout_seconds,
    }

    try:
        print(f"\n⏳ Starting model loading (timeout: {timeout_seconds}s)...")
        print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Import transformers here to measure its impact
        from transformers import AutoModel, AutoTokenizer
        
        # Real-time monitoring
        model = None
        tokenizer = None
        last_memory_check = time.time()
        
        print("\n📈 REAL-TIME MONITORING:")
        
        # Load tokenizer first (faster)
        print("  [1/2] Loading tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - tokenizer_start
        print(f"  ✓ Tokenizer loaded in {tokenizer_time:.2f}s")
        
        # Check memory after tokenizer
        mem_after_tokenizer = get_memory_info()
        print(f"  - Memory used: {mem_after_tokenizer['used_gb']:.2f} GB ({mem_after_tokenizer['percent_used']:.1f}%)")
        
        # Load model (can be slow)
        print("\n  [2/2] Loading model...")
        model_start = time.time()
        
        # Load with low memory usage if possible
        model = AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,  # Reduces memory usage during loading
            torch_dtype="auto",
        )
        
        model_time = time.time() - model_start
        total_time = time.time() - start_time
        
        print(f"  ✓ Model loaded in {model_time:.2f}s")
        print(f"\n✅ LOADING SUCCESSFUL in {total_time:.2f}s")
        
        # Final system state
        mem_after = get_memory_info()
        cpu_after = get_cpu_info()
        
        print(f"\n📊 FINAL SYSTEM STATE:")
        print(f"  - Available memory: {mem_after['available_gb']:.2f} GB")
        print(f"  - Used memory: {mem_after['used_gb']:.2f} GB ({mem_after['percent_used']:.1f}%)")
        print(f"  - Memory increase: {mem_after['used_gb'] - mem_before['used_gb']:.2f} GB")
        print(f"  - CPU usage: {cpu_after['cpu_percent']:.1f}%")
        
        result["success"] = True
        result["elapsed_time"] = total_time
        result["memory_after"] = mem_after
        result["tokenizer_time"] = tokenizer_time
        result["model_time"] = model_time
        
    except MemoryError as e:
        elapsed = time.time() - start_time
        mem_current = get_memory_info()
        
        print(f"\n❌ MEMORY ERROR after {elapsed:.2f}s")
        print(f"   Memory used at failure: {mem_current['used_gb']:.2f} GB ({mem_current['percent_used']:.1f}%)")
        
        result["error_type"] = "MEMORY_ERROR"
        result["error_message"] = str(e)
        result["elapsed_time"] = elapsed
        result["memory_after"] = mem_current
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        mem_current = get_memory_info()
        
        print(f"\n⏰ TIMEOUT ERROR after {elapsed:.2f}s")
        print(f"   Memory used: {mem_current['used_gb']:.2f} GB ({mem_current['percent_used']:.1f}%)")
        
        result["error_type"] = "TIMEOUT_ERROR"
        result["error_message"] = str(e)
        result["elapsed_time"] = elapsed
        result["memory_after"] = mem_current
        
    except Exception as e:
        elapsed = time.time() - start_time
        mem_current = get_memory_info()
        
        print(f"\n❌ UNEXPECTED ERROR after {elapsed:.2f}s")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"   Memory used: {mem_current['used_gb']:.2f} GB ({mem_current['percent_used']:.1f}%)")
        
        # Analyze error message to detect memory issues
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["memory", "ram", "oom", "out of memory"]):
            result["error_type"] = "MEMORY_ERROR"
            print("\n🔍 DIAGNOSIS: Error appears to be MEMORY related")
        elif "timeout" in error_msg or elapsed >= timeout_seconds * 0.95:
            result["error_type"] = "TIMEOUT_ERROR"
            print("\n🔍 DIAGNOSIS: Error appears to be TIMEOUT related")
        else:
            result["error_type"] = "OTHER_ERROR"
        
        result["error_message"] = str(e)
        result["elapsed_time"] = elapsed
        result["memory_after"] = mem_current
        result["traceback"] = traceback.format_exc()
        
    return result


def print_recommendations(result: dict):
    """Prints recommendations based on diagnostic results."""
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    if result["success"]:
        print("✅ Model loaded successfully.")
        print(f"   Total time: {result['elapsed_time']:.2f}s")
        
        if result["elapsed_time"] > 240:  # > 4 minutes
            print("\n⚠️  Warning: Load time is very high (>4min)")
            print("   - Consider increasing timeout to 600s (10 minutes)")
            print("   - Or use a smaller model")
            
    elif result["error_type"] == "MEMORY_ERROR":
        print("❌ PROBLEM DETECTED: OUT OF MEMORY\n")
        print("Solutions:")
        print("  1. Use a smaller model (1B or 1.7B parameters)")
        print("  2. Request more memory in HF Spaces (PRO plan)")
        print("  3. Use quantization (int8 or int4) to reduce memory usage:")
        print("     ```python")
        print("     from transformers import BitsAndBytesConfig")
        print("     quantization_config = BitsAndBytesConfig(load_in_8bit=True)")
        print("     model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config)")
        print("     ```")
        
        if result["memory_after"]:
            print(f"\n  Available memory: {result['memory_after']['available_gb']:.2f} GB")
            print(f"  More memory needed for this model")
            
    elif result["error_type"] == "TIMEOUT_ERROR":
        print("❌ PROBLEM DETECTED: TIMEOUT\n")
        print("Solutions:")
        print(f"  1. Increase timeout from {result['timeout_seconds']}s to 600s (10 minutes):")
        print("     ```python")
        print("     response = requests.post(url, json=payload, timeout=600)")
        print("     ```")
        print("  2. Implement async loading with progress updates")
        print("  3. Cache pre-loaded models in HF Spaces")
        print("  4. Use smaller models that load faster")
        
    else:
        print("❌ PROBLEM DETECTED: UNEXPECTED ERROR\n")
        print("Review the full traceback for more details")
        if "traceback" in result:
            print("\n" + result["traceback"])
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Models to test (from smallest to largest)
    test_models = [
        "meta-llama/Llama-3.2-1B",  # Small model
        # "meta-llama/Llama-3.2-3B",  # Medium model (uncomment to test)
        # "meta-llama/Llama-3-8B",     # Large model (uncomment to test)
    ]
    
    # You can change the timeout here
    TIMEOUT = 300  # 5 minutes
    
    results = []
    
    for model_name in test_models:
        result = monitor_model_loading(model_name, timeout_seconds=TIMEOUT)
        results.append(result)
        print_recommendations(result)
        
        # If failed, don't test larger models
        if not result["success"]:
            print("\n⚠️  Stopping tests due to error.")
            print("    Larger models will likely fail as well.")
            break
        
        # Wait a bit between tests
        time.sleep(2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['elapsed_time']:.1f}s"
        print(f"{status} Test {i}: {result['model_name']}")
        print(f"   Time: {time_str} | Error: {result['error_type'] or 'None'}")
        
    print()
