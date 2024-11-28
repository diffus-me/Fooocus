#!/usr/bin/env python3

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from api import numpy_array_to_base64
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from modules.async_worker import AsyncTask

_NSFW_ALLOWED_TIERS = {"basic", "plus", "pro", "api", "ltd s", "appsumo ltd tier 2"}


def _check_nsfw(endpoint: str, image: np.array, prompt: str) -> dict[str, Any]:
    url = f"{endpoint}/api/v3/internal/moderation/content"
    body = {
        "text": prompt,
        "image": {"encoded_image": numpy_array_to_base64(image)},
    }

    response = requests.post(url, json=body)
    response.raise_for_status()

    result = response.json()

    return result


def nsfw_blur(
    image: np.array, prompt: str, async_task: "AsyncTask"
) -> tuple[Image.Image | None, dict[str, Any] | None]:
    assert async_task.metadata is not None

    if async_task.metadata["user-tier"].lower() in _NSFW_ALLOWED_TIERS:
        return None, None

    endpoint = async_task.metadata["x-diffus-api-gateway-endpoint"]

    print("[NSFW] Start detecting NSFW content")

    start_at = time.perf_counter()
    result = _check_nsfw(endpoint, image, prompt)
    ended_at = time.perf_counter()

    print(f"[NSFW] Detecting NSFW has taken: {(ended_at - start_at):.2f} seconds")

    if result["flag"]:
        return Image.fromarray(image).filter(ImageFilter.BoxBlur(10)), result

    return None, result
