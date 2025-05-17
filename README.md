# OptiPFair-API MVP

A REST API built with FastAPI that exposes the bias visualization capabilities of the OptiPFair library.

---

## 📋 Overview

This microservice provides endpoints to generate and download visualizations of activation patterns in transformer-based LLMs (e.g., LLaMA) using the **OptiPFair** toolkit.

* **Main endpoints:** 
  * `POST /visualize/pca` - PCA visualization of activations
  * `POST /visualize/mean-diff` - Mean activation difference across layers
  * `POST /visualize/heatmap` - Heatmap of activation differences
* **Response:** Binary image (PNG, SVG, or PDF) showing the requested visualization.

---

## ⚙️ Requirements

* Python 3.10 or higher
* Git
* Optional: Mac (Apple Silicon) or NVIDIA GPU for hardware acceleration (MPS/CUDA)
* Internet connection to download models from Hugging Face Hub

---

## 🛠 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your_username/optipfair-api.git
cd optipfair-api

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install "fastapi" "uvicorn" "optipfair[viz]" "pytest" "requests"

# 4. (Optional) Verify PyTorch MPS/CUDA support
python3 - <<'EOF'
import torch
print("MPS built?", torch.backends.mps.is_built(), "Available?", torch.backends.mps.is_available())
print("CUDA available?", torch.cuda.is_available())
EOF
```

---

## 🚀 Running the Server

With the virtual environment activated, start the API:

```bash
uvicorn main:app --reload
```

* Server URL: `http://127.0.0.1:8000`
* Swagger UI:   `http://127.0.0.1:8000/docs`
* Redoc:        `http://127.0.0.1:8000/redoc`

---

## 🔍 Endpoints

### Health Check

```http
GET /ping
```

**Response:**

```json
{ "message": "pong" }
```

---

### 1. PCA Visualization

```http
POST /visualize/pca
```

Generates a PCA scatter plot comparing activations for two prompts.

**Request JSON Schema:**

| Field            | Type        | Description                                                                         |
| ---------------- | ----------- | ----------------------------------------------------------------------------------- |
| `model_name`     | `string`    | Hugging Face model identifier (e.g. `"meta-llama/Llama-3.2-1B"`).                   |
| `prompt_pair`    | `string[2]` | Array of exactly two prompts to compare.                                            |
| `layer_key`      | `string`    | Exact layer name (e.g. `"attention_output_layer_0"`).                               |
| `highlight_diff` | `boolean`   | (Optional) Highlight differing tokens (default: `true`).                            |
| `figure_format`  | `string`    | (Optional) Output format: `png`, `svg`, or `pdf` (default: `png`).                  |
| `pair_index`     | `integer`   | (Optional) Index for naming output file (default: `0`).                             |
| `output_dir`     | `string`    | (Optional) Custom directory to save the image.                                      |

**Example using `curl`:**

```bash
curl -X POST http://127.0.0.1:8000/visualize/pca \
  -H "Content-Type: application/json" \
  -d '{
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_pair": [
          "The white doctor examined the patient. The nurse thought",
          "The Black doctor examined the patient. The nurse thought"
        ],
        "layer_key": "attention_output_layer_0",
        "figure_format": "png"
      }' \
  --output pca.png
```

**Example using Python (`requests`):**

```python
import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_key": "attention_output_layer_0",
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/pca", json=payload)
resp.raise_for_status()

with open("pca_result.png", "wb") as f:
    f.write(resp.content)

print("Saved PCA visualization to pca_result.png")
```

---

### 2. Mean Difference Visualization

```http
POST /visualize/mean-diff
```

Generates a bar chart showing mean activation differences across layers for a specific component type.

**Request JSON Schema:**

| Field           | Type        | Description                                                                  |
| --------------- | ----------- | ---------------------------------------------------------------------------- |
| `model_name`    | `string`    | Hugging Face model identifier (e.g. `"meta-llama/Llama-3.2-1B"`).            |
| `prompt_pair`   | `string[2]` | Array of exactly two prompts to compare.                                     |
| `layer_type`    | `string`    | Component type to analyze (e.g. `"attention_output"`).                       |
| `figure_format` | `string`    | (Optional) Output format (default: `png`).                                   |
| `output_dir`    | `string`    | (Optional) Custom directory to save the image.                               |
| `pair_index`    | `integer`   | (Optional) Index for naming output file (default: `0`).                      |

**Valid layer types for `layer_type`:**
- `mlp_output` - Output of the MLP block
- `attention_output` - Output of the attention mechanism
- `gate_proj` - Output of the gate projection in GLU
- `up_proj` - Output of the up projection in GLU
- `down_proj` - Output of the down projection in GLU
- `input_norm` - Output of the input normalization

**Example using Python (`requests`):**

```python
import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_type": "attention_output",
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/mean-diff", json=payload)
resp.raise_for_status()

with open("mean_diff_result.png", "wb") as f:
    f.write(resp.content)

print("Saved mean difference visualization to mean_diff_result.png")
```

---

### 3. Heatmap Visualization

```http
POST /visualize/heatmap
```

Generates a heatmap showing activation differences for a specific layer.

**Request JSON Schema:**

| Field           | Type        | Description                                                                  |
| --------------- | ----------- | ---------------------------------------------------------------------------- |
| `model_name`    | `string`    | Hugging Face model identifier (e.g. `"meta-llama/Llama-3.2-1B"`).            |
| `prompt_pair`   | `string[2]` | Array of exactly two prompts to compare.                                     |
| `layer_key`     | `string`    | Exact layer name (e.g. `"attention_output_layer_0"`).                        |
| `figure_format` | `string`    | (Optional) Output format (default: `png`).                                   |
| `output_dir`    | `string`    | (Optional) Custom directory to save the image.                               |
| `pair_index`    | `integer`   | (Optional) Index for naming output file (default: `0`).                      |

**Example using Python (`requests`):**

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

with open("heatmap_result.png", "wb") as f:
    f.write(resp.content)

print("Saved heatmap visualization to heatmap_result.png")
```

---

## 📁 Project Structure

```
optipfair-api/           # Repository root
├── main.py              # FastAPI application entrypoint
├── routers/             # API route modules
│   └── visualize.py     # Routes for /visualize/*
├── schemas/             # Pydantic request/response models
│   └── visualize.py     # Request schemas for visualizations
├── utils/               # Internal utility functions
│   └── visualize_pca.py # Wrappers for optipfair visualization functions
└── README.md            # Project documentation
```

---

## ✅ Next Steps

* Improve documentation and usage examples
* Add `/bias-report` endpoint for comprehensive reports
* Add automated tests and CI/CD pipeline
* Dockerize the service and deploy (e.g., on Hugging Face Spaces)

---

## 📄 License

Apache License 2.0 © Pere Martra
