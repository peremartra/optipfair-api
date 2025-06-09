# routers/visualize.py
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from schemas.visualize import (
    VisualizeHeatmapRequest,
    VisualizeMeanDiffRequest,
    VisualizePCARequest,
)
from utils.visualize_pca import (
    run_visualize_heatmap,
    run_visualize_mean_diff,
    run_visualize_pca,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(
    prefix="/visualize",
    tags=["visualization"],
)


@router.post(
    "/pca",
    summary="Generates and returns the PCA visualization of activations",
    response_class=FileResponse,
)
async def visualize_pca_endpoint(req: VisualizePCARequest):
    """
    Receives the parameters, calls the wrapper for optipfair.bias.visualize_pca,
    and returns the resulting PNG/SVG image.
    """
    # 1. Execute the image generation and get the file path
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
        # Log the full trace for debugging
        logger.exception("❌ Error in visualize_pca_endpoint")
        # And return the message to the client
        raise HTTPException(status_code=500, detail=str(e))
    # 2. Verify that the file exists
    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(
            status_code=500, detail="Image file not found after generation"
        )

    # 3. Return the file directly to the client
    return FileResponse(
        path=filepath,
        media_type=f"image/{req.figure_format}",
        filename=os.path.basename(filepath),
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"'
        },
    )


@router.post("/mean-diff", response_class=FileResponse)
async def visualize_mean_diff_endpoint(req: VisualizeMeanDiffRequest):
    """
    Receives the parameters, calls the wrapper for optipfair.bias.visualize_mean_differences,
    and returns the resulting PNG/SVG image.
    """
    try:
        filepath = run_visualize_mean_diff(
            model_name=req.model_name,
            prompt_pair=tuple(req.prompt_pair),
            layer_type=req.layer_type,  # Changed from layer_key to layer_type
            figure_format=req.figure_format,
            output_dir=req.output_dir,
            pair_index=req.pair_index,
        )
    except Exception as e:
        # Log the full trace for debugging
        logger.exception("Error in mean-diff endpoint")
        raise HTTPException(status_code=500, detail=str(e))

    # Verify that the file exists
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=500, detail="Image file not found")

    # Return the file directly to the client
    return FileResponse(
        path=filepath,
        media_type=f"image/{req.figure_format}",
        filename=os.path.basename(filepath),
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"'
        },
    )


@router.post("/heatmap", response_class=FileResponse)
async def visualize_heatmap_endpoint(req: VisualizeHeatmapRequest):
    """
    Receives the parameters, calls the wrapper for optipfair.bias.visualize_heatmap,
    and returns the resulting PNG/SVG image.
    """
    try:
        filepath = run_visualize_heatmap(
            model_name=req.model_name,
            prompt_pair=tuple(req.prompt_pair),
            layer_key=req.layer_key,
            figure_format=req.figure_format,
            output_dir=req.output_dir,
        )
    except Exception as e:
        # Log the full trace for debugging
        logger.exception("Error in heatmap endpoint")
        raise HTTPException(status_code=500, detail=str(e))

    # Verify that the file exists
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=500, detail="Image file not found")

    # Return the file directly to the client
    return FileResponse(
        path=filepath,
        media_type=f"image/{req.figure_format}",
        filename=os.path.basename(filepath),
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"'
        },
    )
