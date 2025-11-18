"""
Optimized configuration for HF Spaces with intelligent handling of large models.
This file contains recommended settings based on available hardware.
"""

import os
from typing import Dict, Any


class HFSpacesConfig:
    """Optimized configuration for different HF Spaces tiers"""
    
    # Timeouts (in seconds)
    TIMEOUT_SMALL_MODEL = 120    # Models <2B parameters
    TIMEOUT_MEDIUM_MODEL = 300   # Models 2-5B parameters
    TIMEOUT_LARGE_MODEL = 600    # Models >5B parameters
    TIMEOUT_PING = 5             # Health checks
    
    # Recommended memory limits (GB) per HF Spaces tier
    MEMORY_LIMITS = {
        "free": 16,      # Free HF Spaces
        "pro": 32,       # HF Spaces PRO
        "enterprise": 64 # HF Spaces Enterprise
    }
    
    # Recommended models per tier
    RECOMMENDED_MODELS = {
        "free": [
            "meta-llama/Llama-3.2-1B",
            "oopere/pruned40-llama-3.2-1B", 
            "oopere/Fair-Llama-3.2-1B",
            "google/gemma-3-1b-pt",
            "Qwen/Qwen3-1.7B",
        ],
        "pro": [
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3-8B",
        ],
        "enterprise": [
            "meta-llama/Llama-3-70B",
        ]
    }
    
    # Model loading configuration
    MODEL_LOAD_CONFIG = {
        "small": {  # <2B params
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "device_map": "auto",
            "timeout": TIMEOUT_SMALL_MODEL,
        },
        "medium": {  # 2-8B params
            "low_cpu_mem_usage": True,
            "torch_dtype": "float16",  # Reduces memory
            "device_map": "auto",
            "timeout": TIMEOUT_MEDIUM_MODEL,
        },
        "large": {  # >8B params
            "low_cpu_mem_usage": True,
            "torch_dtype": "float16",
            "device_map": "auto",
            "load_in_8bit": True,  # int8 quantization
            "timeout": TIMEOUT_LARGE_MODEL,
        }
    }
    
    @classmethod
    def get_model_size_category(cls, model_name: str) -> str:
        """
        Determines the model size category based on the name.
        
        Returns:
            "small", "medium", or "large"
        """
        model_lower = model_name.lower()
        
        # Detect by parameters in the name
        if any(size in model_lower for size in ["1b", "1.7b", "1.5b"]):
            return "small"
        elif any(size in model_lower for size in ["3b", "7b", "8b"]):
            return "medium"
        elif any(size in model_lower for size in ["13b", "30b", "70b"]):
            return "large"
        
        # Default: small (assume the safest case)
        return "small"
    
    @classmethod
    def get_timeout_for_model(cls, model_name: str) -> int:
        """Gets the recommended timeout for a model."""
        size = cls.get_model_size_category(model_name)
        return cls.MODEL_LOAD_CONFIG[size]["timeout"]
    
    @classmethod
    def get_load_config(cls, model_name: str) -> Dict[str, Any]:
        """Gets the optimized loading configuration for a model."""
        size = cls.get_model_size_category(model_name)
        return cls.MODEL_LOAD_CONFIG[size].copy()
    
    @classmethod
    def is_model_recommended(cls, model_name: str, tier: str = "free") -> bool:
        """Verifies if a model is recommended for the current tier."""
        return model_name in cls.RECOMMENDED_MODELS.get(tier, [])
    
    @classmethod
    def get_memory_warning(cls, model_name: str, tier: str = "free") -> str:
        """
        Generates a warning if the model may exceed memory limits.
        
        Returns:
            String with warning, or empty string if no problem
        """
        if cls.is_model_recommended(model_name, tier):
            return ""
        
        size = cls.get_model_size_category(model_name)
        
        if size == "medium" and tier == "free":
            return (
                "⚠️ **Warning**: This model may be too large for free HF Spaces. "
                "Consider upgrading to HF Spaces PRO or using a smaller model."
            )
        elif size == "large" and tier in ["free", "pro"]:
            return (
                "❌ **Error**: This model is too large for your HF Spaces tier. "
                "Use a smaller model or upgrade to Enterprise."
            )
        
        return ""


# Usage example:
def get_optimized_request_config(model_name: str) -> dict:
    """
    Gets optimized configuration for HTTP requests based on the model.
    
    Usage:
        config = get_optimized_request_config("meta-llama/Llama-3.2-1B")
        response = requests.post(url, json=payload, **config)
    """
    return {
        "timeout": HFSpacesConfig.get_timeout_for_model(model_name),
    }


# Default configuration for general use
DEFAULT_CONFIG = {
    "timeout": HFSpacesConfig.TIMEOUT_MEDIUM_MODEL,
    "max_retries": 2,
    "retry_delay": 5,  # seconds between retries
}


if __name__ == "__main__":
    # Usage examples
    print("🔧 Optimized configuration for HF Spaces\n")
    
    test_models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B", 
        "meta-llama/Llama-3-8B",
    ]
    
    for model in test_models:
        print(f"📦 Model: {model}")
        print(f"   Category: {HFSpacesConfig.get_model_size_category(model)}")
        print(f"   Timeout: {HFSpacesConfig.get_timeout_for_model(model)}s")
        print(f"   Recommended (free): {HFSpacesConfig.is_model_recommended(model, 'free')}")
        
        warning = HFSpacesConfig.get_memory_warning(model, "free")
        if warning:
            print(f"   {warning}")
        print()
