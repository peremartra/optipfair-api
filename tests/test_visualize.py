# tests/test_visualize.py

import os
import shutil
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.mark.parametrize("payload, expected_status", [
    # Caso válido: debería devolver 200 y un PNG no vacío
    ({
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_pair": ["A quick brown fox", "A fast brown fox"],
        "layer_key": "attention_output_layer_0",
        "figure_format": "png"
    }, 200),
    # prompt_pair inválido (1 elemento) -> 422
    ({
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_pair": ["only one prompt"],
        "layer_key": "attention_output_layer_0"
    }, 422),
])
def test_visualize_pca_responses(tmp_path, payload, expected_status):
    # Si es un test válido, forzamos output_dir para controlar limpieza
    if expected_status == 200:
        payload["output_dir"] = str(tmp_path)

    response = client.post("/visualize/pca", json=payload)
    assert response.status_code == expected_status

    if expected_status == 200:
        # Debe devolver imagen PNG
        assert response.headers["content-type"] == "image/png"
        # Guardar y comprobar que el fichero existe y no está vacío
        filepath = tmp_path / f"pca_attention_output_layer_0_pair0.png"
        with open(filepath, "wb") as f:
            f.write(response.content)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

def test_layer_not_found_error():
    # layer_key inexistente -> 500 con mensaje de file not found
    payload = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_pair": ["A","B"],
        "layer_key": "nonexistent_layer"
    }
    response = client.post("/visualize/pca", json=payload)
    assert response.status_code == 500
    assert "Expected image file not found" in response.json()["detail"]
