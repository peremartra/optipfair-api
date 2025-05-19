"""Gradio interface for OptiPFair‑API (Phase 1 – basic skeleton)."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from .api_client import OptiPFairAPIClient

# Allow the backend URL to be set via env var for docker‑compose deployments.
API_BASE_URL = os.getenv("OPTIPFAIR_API_URL", "http://localhost:8000")
client = OptiPFairAPIClient(base_url=API_BASE_URL)

# ––––––––––––––––––––––––––––– Core callback ––––––––––––––––––––––––––––––––

def run_analysis(
    model_name: str,
    prompt_a: str,
    prompt_b: str,
    viz_type: str,
    layer_key: str,
    layer_type: str,
    figure_format: str,
):
    """Route the request to the proper API endpoint and return a PIL image."""
    prompt_pair = [prompt_a, prompt_b]
    try:
        if viz_type == "PCA":
            img = client.visualize_pca(model_name, prompt_pair, layer_key, figure_format)
        elif viz_type == "Heatmap":
            img = client.visualize_heatmap(model_name, prompt_pair, layer_key, figure_format)
        else:  # "Mean Diff"
            img = client.visualize_mean_diff(model_name, prompt_pair, layer_type, figure_format)
        return img
    except Exception as exc:  # noqa: BLE001
        # Returning (None, str) maps nicely to (Image, Markdown) outputs later.
        return None, f"❌ Error: {exc}"


# ––––––––––––––––––––––––––––– UI definition ––––––––––––––––––––––––––––––––
with gr.Blocks(title="OptiPFair Bias Analysis", theme="soft") as demo:
    gr.Markdown("""
    # OptiPFair Gradio Interface
    *Phase 1 – quick prototype.*
    """)

    with gr.Row():
        model_name_in = gr.Textbox(
            label="Model ID (HF)",
            value="meta-llama/Llama-3.2-1B",
            placeholder="e.g. meta-llama/Llama-3.2-1B",
        )
        figure_format_in = gr.Radio(["png", "svg"], value="png", label="Figure format")

    with gr.Row():
        prompt_a_in = gr.Textbox(label="Prompt A")
        prompt_b_in = gr.Textbox(label="Prompt B")

    viz_type_in = gr.Radio(["PCA", "Heatmap", "Mean Diff"], value="PCA", label="Visualization type")

    layer_key_in = gr.Textbox(label="Layer key (for PCA/Heatmap)", placeholder="attention_output_layer_7")
    layer_type_in = gr.Dropdown(
        [
            "mlp_output",
            "attention_output",
            "gate_proj",
            "up_proj",
            "down_proj",
            "input_norm",
        ],
        label="Layer type (for Mean Diff)",
        value="mlp_output",
        visible=False,
    )

    # Dynamically show/hide layer inputs when switching viz type
    def _toggle(viz_choice):
        return (
            gr.update(visible=viz_choice in {"PCA", "Heatmap"}),
            gr.update(visible=viz_choice == "Mean Diff"),
        )

    viz_type_in.change(_toggle, inputs=viz_type_in, outputs=[layer_key_in, layer_type_in])

    run_btn = gr.Button("Run analysis 🔍")

    with gr.Row():
        output_image = gr.Image(type="pil", label="Result", interactive=False)
        error_md = gr.Markdown(visible=False)

    # Click → call backend
    run_btn.click(
        run_analysis,
        inputs=[
            model_name_in,
            prompt_a_in,
            prompt_b_in,
            viz_type_in,
            layer_key_in,
            layer_type_in,
            figure_format_in,
        ],
        outputs=[output_image, error_md],
    )

if __name__ == "__main__":  # pragma: no cover
    # Launch Gradio in share mode if running inside HF Spaces.
    share = bool(os.getenv("HF_SPACE"))
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
