# schemas/visualize.py
from pydantic import BaseModel, field_validator
from typing import List, Optional

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

