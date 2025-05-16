# routers/visualize.py
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from schemas.visualize import VisualizePCARequest
from utils.visualize_pca import run_visualize_pca

router = APIRouter(
    prefix="/visualize",
    tags=["visualization"],
)

@router.post(
    "/pca",
    summary="Genera y devuelve la visualización PCA de activaciones",
    response_class=FileResponse,
)
async def visualize_pca_endpoint(req: VisualizePCARequest):
    """
    Recibe los parámetros, llama al wrapper de optipfair.bias.visualize_pca y
    devuelve la imagen PNG/SVG resultante.
    """
    # 1. Ejecutar la generación de la imagen y obtener la ruta al fichero
    try:
        filepath = run_visualize_pca(
            model_name=req.model_name,
            prompt_pair=tuple(req.prompt_pair),
            layer_key=req.layer_key,
            highlight_diff=req.highlight_diff,
            output_dir=req.output_dir,
            figure_format=req.figure_format,
            pair_index=req.pair_index,
        )
    except Exception as e:
        # Logueamos la traza completa para depurar
        import logging
        logging.exception("❌ Error en visualize_pca_endpoint")
        # Y devolvemos el mensaje al cliente
        raise HTTPException(status_code=500, detail=str(e))
    # 2. Verificar que el fichero exista
    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=500, detail="Image file not found after generation")

    # 3. Devolver el fichero directamente al cliente
    return FileResponse(
        path=filepath,
        media_type=f"image/{req.figure_format}",
        filename=os.path.basename(filepath),
        headers={"Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"'},
    )
