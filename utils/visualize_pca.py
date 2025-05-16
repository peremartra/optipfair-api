# utils/visualize_pca.py

# utils/visualize_pca.py

import os
import tempfile
import logging
from functools import lru_cache
from typing import Tuple, Optional

import torch
from optipfair.bias import visualize_pca
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@lru_cache(maxsize=None)
def load_model_tokenizer(model_name: str):
    """
    Carga el modelo y el tokenizer en CPU una sola vez y cachea el resultado.
    """
    logger.info(f"Loading model and tokenizer for '{model_name}'")
    
    # Selección de device: MPS (Apple Silicon) > CUDA > CPU
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
    """
    Wrapper que:
      - Prepara el directorio de salida
      - Selecciona device: MPS > CUDA > CPU
      - Carga (y cachea) modelo y tokenizer
      - Mueve el modelo al device elegido
      - Llama a optipfair.bias.visualize_pca
      - Devuelve la ruta al fichero generado
    """
    # Preparar output_dir
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optipfair_pca_")
    os.makedirs(output_dir, exist_ok=True)


    # Cargar modelo y tokenizer cacheados
    model, tokenizer = load_model_tokenizer(model_name)

    # Ejecutar la visualización
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

    # Construir la ruta al fichero según convención interna
    layer_parts = layer_key.split("_")
    layer_type = "_".join(layer_parts[:-1])
    layer_num = layer_parts[-1]
    filename = f"pca_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}"
    filepath = os.path.join(output_dir, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Expected image file not found: {filepath}")

    logger.info(f"PCA image saved at {filepath}")
    return filepath

