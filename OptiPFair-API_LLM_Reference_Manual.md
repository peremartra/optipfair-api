# OptiPFair-API: LLM Reference Manual

This reference manual is designed to help Large Language Models understand and work with the OptiPFair-API, a FastAPI-based service that provides visualization endpoints for analyzing bias in large language models using the OptiPFair toolkit.

## 1. Project Overview

OptiPFair-API is a REST API that provides endpoints to generate visualizations for analyzing potential biases in LLM activations. It exposes three main visualization types:

1. **PCA Visualization** - Principal Component Analysis of activations for specific layers
2. **Mean Difference Visualization** - Bar charts showing mean activation differences across layers
3. **Heatmap Visualization** - Heat maps of activation differences in specific layers

The API wraps functionality from the `optipfair.bias` module, handling model loading, visualization generation, and serving image results.

## 2. Project Structure

```
optipfair-api/
├── main.py                # Application entrypoint (FastAPI app)
├── routers/               # API routes
│   └── visualize.py       # Visualization endpoints
├── schemas/               # Pydantic models for requests
│   └── visualize.py       # Request/response schemas
├── utils/                 # Helper utilities
│   └── visualize_pca.py   # Functions to generate visualizations
└── tests/                 # Test files
```

## 3. API Endpoints

### 3.1 Health Check

```http
GET /ping
```

Returns `{"message": "pong"}` to verify the API is running.

### 3.2 PCA Visualization

```http
POST /visualize/pca
```

Generates a PCA scatter plot comparing activations for two prompts at a specific layer.

### 3.3 Mean Difference Visualization

```http
POST /visualize/mean-diff
```

Generates a bar chart showing mean activation differences across layers for a specific component type.

### 3.4 Heatmap Visualization

```http
POST /visualize/heatmap
```

Generates a heatmap showing activation differences for a specific layer.

## 4. Request Models

### 4.1 PCA Request (`VisualizePCARequest`)

```python
{
    "model_name": str,           # Hugging Face model ID
    "prompt_pair": List[str],    # Exactly 2 prompts to compare
    "layer_key": str,            # Layer identifier (e.g., "attention_output_layer_0")
    "highlight_diff": bool,      # Whether to highlight differing tokens (default: True)
    "figure_format": str,        # Output format: "png", "svg", "pdf" (default: "png")
    "pair_index": int,           # Index for naming output files (default: 0)
    "output_dir": str | None     # Optional directory to save output
}
```

### 4.2 Mean Diff Request (`VisualizeMeanDiffRequest`)

```python
{
    "model_name": str,           # Hugging Face model ID
    "prompt_pair": List[str],    # Exactly 2 prompts to compare
    "layer_type": str,           # Component type to analyze (see section 5.2)
    "figure_format": str,        # Output format (default: "png")
    "output_dir": str | None,    # Optional directory to save output
    "pair_index": int            # Index for naming output files (default: 0)
}
```

### 4.3 Heatmap Request (`VisualizeHeatmapRequest`)

```python
{
    "model_name": str,           # Hugging Face model ID
    "prompt_pair": List[str],    # Exactly 2 prompts to compare
    "layer_key": str,            # Layer identifier (e.g., "attention_output_layer_0")
    "figure_format": str,        # Output format (default: "png")
    "output_dir": str | None     # Optional directory to save output
}
```

## 5. Key Parameters and Values

### 5.1 Layer Keys (for PCA and Heatmap)

Layer keys follow the format: `{component_type}_layer_{layer_number}`

For example:
- `attention_output_layer_0` - First attention output layer
- `mlp_output_layer_7` - MLP output of layer 7

### 5.2 Layer Types (for Mean Diff)

Valid values for `layer_type` in mean-diff visualization:
- `mlp_output` - Output of the MLP block
- `attention_output` - Output of the attention mechanism
- `gate_proj` - Output of the gate projection in GLU
- `up_proj` - Output of the up projection in GLU
- `down_proj` - Output of the down projection in GLU
- `input_norm` - Output of the input normalization

## 6. Core Functionality

The API uses three main functions to generate visualizations:

1. `run_visualize_pca` - Generates PCA plots for specific layers
2. `run_visualize_mean_diff` - Generates bar charts showing mean differences across layers
3. `run_visualize_heatmap` - Generates heatmaps showing activation differences

All functions use `load_model_tokenizer` which caches loaded models for efficiency.

## 7. Example Usage

### 7.1 PCA Visualization

```python
import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_key": "attention_output_layer_7",
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/pca", json=payload)
resp.raise_for_status()

with open("pca_output.png", "wb") as f:
    f.write(resp.content)
```

### 7.2 Mean Difference Visualization

```python
import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_type": "attention_output",  # Just the component type, not full layer key
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/mean-diff", json=payload)
resp.raise_for_status()

with open("mean_diff_output.png", "wb") as f:
    f.write(resp.content)
```

### 7.3 Heatmap Visualization

```python
import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_key": "attention_output_layer_2",
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/heatmap", json=payload)
resp.raise_for_status()

with open("heatmap_output.png", "wb") as f:
    f.write(resp.content)
```

## 8. Important Implementation Notes

1. **Mean-Diff vs Other Visualizations**: The mean-diff endpoint uses `layer_type` parameter, not the full `layer_key` like PCA and heatmap. This is because mean-diff shows patterns across all layers of a type.

2. **File Naming**: Files are saved with predetermined naming conventions:
   - PCA: `pca_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}`
   - Mean-Diff: `mean_diff_{layer_type}_pair{pair_index}.{figure_format}`
   - Heatmap: `heatmap_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}`

3. **Device Selection**: The API automatically selects the appropriate device (CUDA > MPS > CPU) for model loading.

## 9. Common Issues and Solutions

1. **File Not Found Errors**: May occur if there's a mismatch between expected and actual filename patterns. Check that the visualization function is generating the expected filename.

2. **422 Validation Errors**: Occur when request payload doesn't match the expected schema (e.g., wrong parameter names, wrong data types).

3. **500 Server Errors**: Could indicate issues with model loading or the visualization process itself.

## 10. Response

All visualization endpoints return:
- The binary image data with appropriate content-type
- HTTP 200 on success
- HTTP 422 for validation errors
- HTTP 500 for internal errors (with error message in response body)

## 11. Model Support

The API is designed to work with Hugging Face transformer models, particularly:
- LLaMA models (meta-llama/Llama-3.2-1B, etc.)
- Other HuggingFace-compatible decoder-only transformer models