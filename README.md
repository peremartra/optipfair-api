# OptiPFair-API 

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
* **Docker & Docker Compose** (recommended for easy deployment)
* Optional: Mac (Apple Silicon) or NVIDIA GPU for hardware acceleration (MPS/CUDA)
* Internet connection to download models from Hugging Face Hub

---

## 🐳 Docker Deployment (Recommended)

The easiest way to run OptiPFair-API is using Docker Compose, which automatically handles all dependencies and services.

### Quick Start with Docker

```bash
# 1. Clone the repository
git clone https://github.com/your_username/optipfair-api.git
cd optipfair-api

# 2. Start the entire stack
docker-compose up -d

# 3. Access the application
# Frontend (Gradio): http://localhost:7860
# Backend API docs: http://localhost:8000/docs
```

### Docker Commands

```bash
# Start services (detached mode)
docker-compose up -d

# View logs in real-time
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build

# Check service status
docker-compose ps
```

### Docker Architecture

```
┌─────────────────────────────────────┐
│           docker-compose            │
├─────────────────┬───────────────────┤
│   Backend       │    Frontend       │
│   (FastAPI)     │    (Gradio)       │
│   Port: 8000    │    Port: 7860     │
└─────────────────┴───────────────────┘
```

**Features:**
- ✅ **Automatic model caching** - Downloads models once, reuses them
- ✅ **Health monitoring** - Services restart automatically if they fail  
- ✅ **Persistent storage** - Model cache survives container restarts
- ✅ **Production ready** - Secure, non-root containers
- ✅ **Cross-platform** - Works on Mac, Linux, Windows

---

## 🌐 Try Online (HF Spaces)

**🚀 Zero Installation Required!**

You can try OptiPFair-API directly in your browser without any setup:

**[🔗 OptipFair Bias Analyzer on Hugging Face Spaces](https://huggingface.co/spaces/oopere/optipfair-bias-analyzer)**

**Features:**
- ✅ **Instant access** - No installation required
- ✅ **GPU acceleration** - Faster model loading and processing  
- ✅ **Pre-loaded models** - Ready to use immediately
- ✅ **Full functionality** - All three visualization types (PCA, Mean Diff, Heatmap)
- ✅ **Public sharing** - Share results with colleagues

**Perfect for:**
- 🧪 **Quick testing** of bias analysis concepts
- 📚 **Learning and experimentation** 
- 🎯 **Demos and presentations**
- 🔄 **Comparing with local deployment**

---

## 🛠 Manual Installation (Alternative)

If you prefer to run without Docker:

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

### Running Manually

```bash
# Terminal 1: Start FastAPI backend
uvicorn main:app --reload

# Terminal 2: Start Gradio frontend  
python gradio_app.py
```

* Backend: `http://127.0.0.1:8000`
* Frontend: `http://127.0.0.1:7860`

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
curl -X POST http://localhost:8000/visualize/pca \
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

resp = requests.post("http://localhost:8000/visualize/pca", json=payload)
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

resp = requests.post("http://localhost:8000/visualize/mean-diff", json=payload)
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

resp = requests.post("http://localhost:8000/visualize/heatmap", json=payload)
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
├── gradio_app.py        # Gradio frontend application
├── docker-compose.yml   # Docker orchestration configuration
├── Dockerfile.backend   # Backend container definition
├── Dockerfile.frontend  # Frontend container definition
├── requirements-docker.txt # Optimized dependencies for containers
├── routers/             # API route modules
│   └── visualize.py     # Routes for /visualize/*
├── schemas/             # Pydantic request/response models
│   └── visualize.py     # Request schemas for visualizations
├── utils/               # Internal utility functions
│   └── visualize_pca.py # Wrappers for optipfair visualization functions
├── hf-spaces/           # Hugging Face Spaces deployment
└── README.md            # Project documentation
```

---

## 🚀 Deployment Options

### 1. 🌐 Hugging Face Spaces (Try Now!)
**[🔗 OptipFair Bias Analyzer](https://huggingface.co/spaces/oopere/optipfair-bias-analyzer)**

**Pros:** Zero setup, GPU acceleration, instant access, public sharing  
**Cons:** Limited to HF Spaces platform, shared resources

### 2. 🐳 Docker (Recommended for Local/Production)
```bash
docker-compose up -d
```
**Pros:** Easy setup, automatic dependencies, production-ready, full control  
**Cons:** Requires Docker installation

### 3. 📱 Manual Installation
Traditional Python virtual environment setup.

**Pros:** Full control, native performance (MPS on Mac), development flexibility  
**Cons:** Manual dependency management, longer setup time

---

## 📖 Citation

If you use OptipFair-API in your research or projects, please cite both the API and the underlying library:

### OptipFair Library (Core Implementation)
```bibtex
@software{optipfair,
  author = {Pere Martra},
  title = {OptipFair: Structured Pruning and Bias Visualization for Large Language Models},
  url = {https://github.com/peremartra/optipfair},
  version = {0.1.3},
  year = {2024}
}
```

### OptipFair-API (REST Interface)
```bibtex
@software{optipfair_api,
  author = {Pere Martra},
  title = {OptipFair-API: REST API for LLM Bias Analysis and Visualization},
  url = {https://github.com/peremartra/optipfair-api},
  year = {2025}
}
```

---

## 📄 License

Apache License 2.0 © Pere Martra
