# schemas/visualize.py
from typing import List, Optional

from pydantic import BaseModel, field_validator


class VisualizePCARequest(BaseModel):
    """
    Schema for the /visualize-pca endpoint.
    """

    model_name: str
    prompt_pair: List[str]
    layer_key: str
    highlight_diff: bool = True
    figure_format: str = "png"
    pair_index: int = 0
    output_dir: Optional[str] = None

    @field_validator("prompt_pair")
    def must_be_two_prompts(cls, v):
        if len(v) != 2:
            raise ValueError("prompt_pair must be a list of exactly two strings")
        return v

    @field_validator("model_name")  # ← AÑADIR ESTE VALIDADOR
    def model_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class VisualizeMeanDiffRequest(BaseModel):
    model_name: str
    prompt_pair: List[str]
    layer_type: str  # Changed from layer_key to layer_type
    figure_format: str = "png"
    output_dir: Optional[str] = None
    pair_index: int = 0

    @field_validator("prompt_pair")
    def must_be_two_prompts(cls, v):
        if len(v) != 2:
            raise ValueError("prompt_pair must be a list of exactly two strings")
        return v

    @field_validator("model_name")  # ← AÑADIR ESTE VALIDADOR
    def model_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class VisualizeHeatmapRequest(BaseModel):
    """
    Schema for the /visualize/heatmap endpoint.
    """

    model_name: str
    prompt_pair: List[str]
    layer_key: str
    figure_format: str = "png"
    output_dir: Optional[str] = None

    @field_validator("prompt_pair")
    def must_be_two_prompts(cls, v):
        if len(v) != 2:
            raise ValueError("prompt_pair must be a list of exactly two strings")
        return v

    @field_validator("model_name")  # ← AÑADIR ESTE VALIDADOR
    def model_name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()
