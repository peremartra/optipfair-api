# utils/visualize_pca.py

import os
import tempfile
from optipfair.bias import visualize_pca
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional


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
    Wrapper que carga el modelo/tokenizer, llama a optipfair.bias.visualize_pca
    y retorna la ruta al fichero de la imagen generada.

    Args:
        model_name: HuggingFace model identifier.
        prompt_pair: Tuple de dos prompts contrastivos.
        layer_key: Nombre exacto de la capa a visualizar.
        highlight_diff: Marcar diferencias entre tokens.
        output_dir: Carpeta donde guardar la imagen. Si es None, crea un tmpdir.
        figure_format: Formato de la imagen ('png', 'svg', 'pdf').
        pair_index: Índice del par de prompts para diferenciar archivos.

    Returns:
        Ruta completa al fichero generado.
    """
    # Preparar output_dir
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="optipfair_pca_")
    os.makedirs(output_dir, exist_ok=True)

    # Cargar modelo y tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Construir filename según convención interna de visualize_pca
    layer_parts = layer_key.split("_")
    layer_type = "_".join(layer_parts[:-1])
    layer_num = layer_parts[-1]
    filename = f"pca_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}"
    filepath = os.path.join(output_dir, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Expected image file not found: {filepath}")

    return filepath
