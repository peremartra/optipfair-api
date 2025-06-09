# tests/test_validation.py
import pytest
from pydantic import ValidationError
from schemas.visualize import (
    VisualizePCARequest,
    VisualizeMeanDiffRequest,
    VisualizeHeatmapRequest,
)


def test_pca_request_valid():
    """Test validación correcta de PCA request"""
    valid_data = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_pair": ["Prompt 1", "Prompt 2"],
        "layer_key": "attention_output_layer_0",
    }
    request = VisualizePCARequest(**valid_data)
    assert request.model_name == "meta-llama/Llama-3.2-1B"
    assert len(request.prompt_pair) == 2


def test_pca_request_invalid_prompts():
    """Test error con prompt_pair inválido"""
    with pytest.raises(ValidationError):
        VisualizePCARequest(
            model_name="test",
            prompt_pair=["only one"],  # Debe ser exactamente 2
            layer_key="layer_0",
        )


def test_mean_diff_request_valid():
    """Test validación mean-diff request"""
    valid_data = {
        "model_name": "test-model",
        "prompt_pair": ["A", "B"],
        "layer_type": "attention_output",
    }
    request = VisualizeMeanDiffRequest(**valid_data)
    assert request.layer_type == "attention_output"
