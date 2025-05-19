"""Thin client for talking to the OptiPFair‑API backend (FastAPI).

All HTTP details live here so the interface code remains tidy.
"""
from __future__ import annotations

import io
from typing import List

import requests
from PIL import Image


class OptiPFairAPIClient:
    """Wraps calls to the running FastAPI service.

    Parameters
    ----------
    base_url: str
        Root URL where the OptiPFair‑API is reachable, without a trailing slash.
        Default (good for local docker‑compose): ``http://localhost:8000``.
    timeout: int | float
        Seconds before an HTTP request is aborted.
    """

    def __init__(self, base_url: str = "http://localhost:8000", *, timeout: int | float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ––––––––––––––––––––––––––––– Private helpers ––––––––––––––––––––––––––––
    def _post_image(self, endpoint: str, payload: dict) -> Image.Image:
        """POST *payload* to *endpoint* and return result as a PIL image."""
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    # ––––––––––––––––––––––––––––– Public helpers –––––––––––––––––––––––––––––
    def visualize_pca(
        self,
        model_name: str,
        prompt_pair: List[str],
        layer_key: str,
        figure_format: str = "png",
    ) -> Image.Image:
        payload = {
            "model_name": model_name,
            "prompt_pair": prompt_pair,
            "layer_key": layer_key,
            "figure_format": figure_format,
        }
        return self._post_image("/visualize/pca", payload)

    def visualize_mean_diff(
        self,
        model_name: str,
        prompt_pair: List[str],
        layer_type: str,
        figure_format: str = "png",
    ) -> Image.Image:
        payload = {
            "model_name": model_name,
            "prompt_pair": prompt_pair,
            "layer_type": layer_type,
            "figure_format": figure_format,
        }
        return self._post_image("/visualize/mean-diff", payload)

    def visualize_heatmap(
        self,
        model_name: str,
        prompt_pair: List[str],
        layer_key: str,
        figure_format: str = "png",
    ) -> Image.Image:
        payload = {
            "model_name": model_name,
            "prompt_pair": prompt_pair,
            "layer_key": layer_key,
            "figure_format": figure_format,
        }
        return self._post_image("/visualize/heatmap", payload)

