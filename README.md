# OptiPFair-API MVP

A minimal REST API built with FastAPI to expose the PCA visualization capabilities of the OptiPFair library.

---

## 📋 Overview

This microservice provides an endpoint to generate and download PCA analysis visualizations of activation patterns from transformer-based LLMs (e.g., LLaMA) using the **OptiPFair** toolkit.

* **Primary endpoint:** `POST /visualize/pca`
* **Response:** Binary image (PNG, SVG, or PDF) showing the PCA scatter plot of activations.

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

### PCA Visualization

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
| `output_dir`     | `string`    | (Optional) Custom directory to save the image; if omitted, uses a temporary folder. |

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

**Response:**

* Returns an image file with `Content-Type: image/png` or `image/svg+xml` or `application/pdf` depending on `figure_format`.

---

## 📁 Project Structure

```
optipfair-api/           # Repository root
├── main.py              # FastAPI application entrypoint
├── routers/             # API route modules
│   └── visualize.py     # /visualize/pca router
├── schemas/             # Pydantic request/response models
│   └── visualize.py     # VisualizePCARequest model
├── utils/               # Internal utility functions
│   └── visualize_pca.py # Wrapper for optipfair.bias.visualize_pca
├── tests/               # Unit and integration tests
│   └── test_visualize.py
├── requirements.txt     # Pinned dependencies
└── README.md            # Project documentation
```

---

## ✅ Next Steps

* Implement additional endpoints: `/mean-diff`, `/heatmap`, `/bias-report`.
* Add automated tests and CI/CD pipeline.
* Dockerize the service and deploy (e.g., on Hugging Face Spaces).

---

## 📄 License

Apache License 2.0 © Pere Martra
