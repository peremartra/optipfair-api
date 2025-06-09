# tests/test_api_endpoints.py
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_ping_endpoint():
    """Test básico - el endpoint más simple"""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}


def test_pca_endpoint_validation_errors():
    """Test validación sin modelo real"""
    # Payload vacío
    response = client.post("/visualize/pca", json={})
    assert response.status_code == 422

    # prompt_pair con solo 1 elemento
    invalid_payload = {
        "model_name": "test-model",
        "prompt_pair": ["only one prompt"],
        "layer_key": "attention_output_layer_0",
    }
    response = client.post("/visualize/pca", json=invalid_payload)
    assert response.status_code == 422


def test_mean_diff_endpoint_validation():
    """Test validación mean-diff"""
    invalid_payload = {
        "model_name": "",
        "prompt_pair": ["prompt1", "prompt2"],
        "layer_type": "invalid_type",
    }
    response = client.post("/visualize/mean-diff", json=invalid_payload)
    assert response.status_code == 422


def test_heatmap_endpoint_validation():
    """Test validación heatmap"""
    invalid_payload = {
        "model_name": "test",
        "prompt_pair": [],  # Array vacío
        "layer_key": "attention_output_layer_0",
    }
    response = client.post("/visualize/heatmap", json=invalid_payload)
    assert response.status_code == 422
