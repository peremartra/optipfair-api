# utils/visualize_pca.py
import os
import tempfile
import logging
from functools import lru_cache
from typing import Tuple, Optional, Union, List

import torch
from optipfair.bias import visualize_pca, visualize_mean_differences, visualize_heatmap
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@lru_cache(maxsize=None)
def load_model_tokenizer(model_name: str):
    """
    Loads the model and tokenizer on the CPU once and caches the result.
    """
    logger.info(f"Loading model and tokenizer for '{model_name}'")
    
    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.to(device)
    
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")

    return model, tokenizer

def run_visualize_pca(
    model_name: str,
    prompt_pair: Tuple[str, str],
    layer_key: str,
    highlight_diff: bool = True,
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
) -> str:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optipfair_pca_")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_tokenizer(model_name)

    visualize_pca(
        model=model,
        tokenizer=tokenizer,
        prompt_pair=prompt_pair,
        layer_key=layer_key,
        highlight_diff=highlight_diff,
        output_dir=output_dir,
        figure_format=figure_format,
        pair_index=pair_index
    )

    layer_parts = layer_key.split("_")
    layer_type = "_".join(layer_parts[:-1])
    layer_num = layer_parts[-1]
    filename = build_visualization_filename(
        vis_type="pca",
        layer_type=layer_type,
        layer_num=layer_num,
        pair_index=pair_index,
        figure_format=figure_format
    )
    filepath = os.path.join(output_dir, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Expected image file not found: {filepath}")

    logger.info(f"PCA image saved at {filepath}")
    return filepath

def run_visualize_mean_diff(
    model_name: str,
    prompt_pair: Tuple[str, str],
    layer_type: str,  # Changed from layer_key to layer_type
    figure_format: str = "png",
    output_dir: Optional[str] = None,
    pair_index: int = 0,
) -> str:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optipfair_mean_diff_")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_tokenizer(model_name)

    visualize_mean_differences(
        model=model,
        tokenizer=tokenizer,
        prompt_pair=prompt_pair,
        layer_type=layer_type,
        layers="all",  # By default, show all layers
        output_dir=output_dir,
        figure_format=figure_format,
        pair_index=pair_index
    )

    filename = build_visualization_filename(
        vis_type="mean_diff",
        layer_type=layer_type,
        pair_index=pair_index,
        figure_format=figure_format
    )
    filepath = os.path.join(output_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Expected image file not found: {filepath}")
    logger.info(f"Mean-diff image saved at {filepath}")
    return filepath

def run_visualize_heatmap(
    model_name: str,
    prompt_pair: Tuple[str, str],
    layer_key: str,
    figure_format: str = "png",
    output_dir: Optional[str] = None,
    pair_index: int = 0,
) -> str:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optipfair_heatmap_")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_tokenizer(model_name)

    visualize_heatmap(
        model=model,
        tokenizer=tokenizer,
        prompt_pair=prompt_pair,
        layer_key=layer_key,
        output_dir=output_dir,
        figure_format=figure_format,
        pair_index=pair_index
    )

    parts = layer_key.split("_")
    layer_type = "_".join(parts[:-1])
    layer_num = parts[-1]
    filename = build_visualization_filename(
        vis_type="heatmap",
        layer_type=layer_type,
        layer_num=layer_num,
        pair_index=pair_index,
        figure_format=figure_format
    )
    filepath = os.path.join(output_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Expected image file not found: {filepath}")
    logger.info(f"Heatmap image saved at {filepath}")
    return filepath

def build_visualization_filename(
    vis_type: str,
    layer_type: str,
    layer_num: str = None,
    layers: Union[str, List[int]] = None,
    pair_index: int = 0,
    figure_format: str = "png"
) -> str:
    """
    Builds the filename for any visualization.
    """
    if vis_type == "mean_diff":
        # The visualize_mean_differences function does not include the layer number in the filename
        return f"mean_diff_{layer_type}_pair{pair_index}.{figure_format}"
    elif vis_type in ("pca", "heatmap"):
        return f"{vis_type}_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}"
    else:
        raise ValueError(f"Unknown visualization type: {vis_type}")

