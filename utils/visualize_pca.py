# utils/visualize_pca.py
import logging
import os
import tempfile
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import matplotlib
import torch
from optipfair.bias import (visualize_heatmap, visualize_mean_differences,
                            visualize_pca)
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")  # Use 'Agg' backend for non-GUI environments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configurar timeouts más largos para Docker
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutos
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Permitir descargas


@lru_cache(maxsize=None)
def load_model_tokenizer(model_name: str):
    """
    Loads the model and tokenizer with extended timeouts for Docker environments
    """
    logger.info(f"Loading model and tokenizer for '{model_name}'")

    # Get HF token from environment for gated models
    hf_token = os.getenv("HF_TOKEN")

    # Device selection: CUDA > CPU (MPS no disponible en Docker)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (MPS not available in Docker containers)")

    try:
        # Cargar modelo con timeouts extendidos
        logger.info(
            f"Downloading/loading model from Hub... (this may take several minutes)"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            resume_download=True,  # Reanudar descargas interrumpidas
            local_files_only=False,  # Permitir descargas
            trust_remote_code=True,  # Para modelos custom
            torch_dtype=torch.float32,  # Usar float32 en CPU
            low_cpu_mem_usage=True,  # Optimizar uso de memoria
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            resume_download=True,
            local_files_only=False,
            trust_remote_code=True,
        )

        model = model.to(device)

        logger.info(f"Model loaded successfully on device: {device}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise e


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
        pair_index=pair_index,
    )

    layer_parts = layer_key.split("_")
    layer_type = "_".join(layer_parts[:-1])
    layer_num = layer_parts[-1]
    filename = build_visualization_filename(
        vis_type="pca",
        layer_type=layer_type,
        layer_num=layer_num,
        pair_index=pair_index,
        figure_format=figure_format,
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
        pair_index=pair_index,
    )

    filename = build_visualization_filename(
        vis_type="mean_diff",
        layer_type=layer_type,
        pair_index=pair_index,
        figure_format=figure_format,
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
        pair_index=pair_index,
    )

    parts = layer_key.split("_")
    layer_type = "_".join(parts[:-1])
    layer_num = parts[-1]
    filename = build_visualization_filename(
        vis_type="heatmap",
        layer_type=layer_type,
        layer_num=layer_num,
        pair_index=pair_index,
        figure_format=figure_format,
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
    figure_format: str = "png",
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
