from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


HSVBounds = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
]

ThresholdMethod = Callable[
    [np.ndarray, np.ndarray, object],
    np.ndarray,
]


def _valid_bounds(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> HSVBounds:
    lower_array = np.clip(np.asarray(lower, dtype=np.int32), 0, 255)
    upper_array = np.clip(np.asarray(upper, dtype=np.int32), 0, 255)

    upper_array = np.maximum(upper_array, lower_array)

    return (
        tuple(int(value) for value in lower_array),
        tuple(int(value) for value in upper_array),
    )


def apply_hsv_bounds(
    hsv_features: np.ndarray,
    bounds: HSVBounds,
) -> np.ndarray:
    """Return the binary uint8 mask expected by histogram.py."""
    lower, upper = bounds

    mask = cv2.inRange(
        hsv_features,
        np.asarray(lower, dtype=np.uint8),
        np.asarray(upper, dtype=np.uint8),
    )

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def manual_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """Original ALINA thresholding implementation."""
    del image

    bounds = _valid_bounds(
        tuple(config.yellow_lower),
        tuple(config.yellow_upper),
    )

    return apply_hsv_bounds(hsv_features, bounds)


def _extract_json(content: str) -> dict[str, object]:
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content.strip(),
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise ValueError(
                "OpenAI threshold response did not contain JSON."
            )
        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError(
            "OpenAI threshold response must be a JSON object."
        )

    return result


def _parse_bounds(content: str) -> HSVBounds:
    response = _extract_json(content)

    lower = response.get("lower")
    upper = response.get("upper")

    if (
        not isinstance(lower, list)
        or not isinstance(upper, list)
        or len(lower) != 3
        or len(upper) != 3
    ):
        raise ValueError(
            "OpenAI must return lower and upper arrays of length three."
        )

    try:
        lower_values = tuple(
            int(round(float(value)))
            for value in lower
        )
        upper_values = tuple(
            int(round(float(value)))
            for value in upper
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OpenAI returned invalid HSV values."
        ) from exc

    return _valid_bounds(lower_values, upper_values)


def _api_key() -> str:
    key_file = Path(
        os.environ.get(
            "OPENAI_API_KEY_FILE",
            "~/.config/codecompanion/token",
        )
    ).expanduser()

    if key_file.exists():
        key = key_file.read_text(
            encoding="utf-8",
        ).strip()
        if key:
            return key

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    raise ValueError(
        "Set OPENAI_API_KEY or create ~/.config/codecompanion/token."
    )


def openai_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """
    Ask a vision model for HSV bounds, then apply them locally.

    This is intentionally an image-adaptive threshold method. It should
    normally be cached or run only on a small experiment subset.
    """
    from openai import OpenAI

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not success:
        raise RuntimeError(
            "Could not encode image for OpenAI threshold selection."
        )

    image_data = base64.b64encode(
        encoded.tobytes(),
    ).decode("ascii")

    prompt = """
You are selecting HSV thresholds for yellow and white airport taxiway
line markings in the supplied warped ROI.

Exclude sky, windshield, aircraft, grass, shadows, and pavement where
possible. Use OpenCV's normalized 8-bit HSV convention:

H: 0 through 255
S: 0 through 255
V: 0 through 255

Return one broad HSV range that captures both yellow and white taxiway
paint.

Return JSON only:
{
  "lower": [H, S, V],
  "upper": [H, S, V]
}
"""

    client = OpenAI(api_key=_api_key())

    response = client.chat.completions.create(
        model=getattr(
            config,
            "threshold_openai_model",
            "gpt-4.1-mini",
        ),
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{image_data}"
                            ),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "OpenAI returned an empty threshold response."
        )

    bounds = _parse_bounds(content)

    # Apply bounds to normalized HSV features, not the BGR image.
    return apply_hsv_bounds(hsv_features, bounds)


THRESHOLD_METHODS: dict[str, ThresholdMethod] = {
    "manual": manual_threshold,
    "openai": openai_threshold,
}


def available_threshold_methods() -> tuple[str, ...]:
    return tuple(sorted(THRESHOLD_METHODS))


def predict_threshold_mask(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    method_name = str(
        getattr(config, "threshold_method", "manual")
    ).lower()

    try:
        method = THRESHOLD_METHODS[method_name]
    except KeyError as exc:
        available = ", ".join(available_threshold_methods())
        raise ValueError(
            f"Unknown threshold method '{method_name}'. "
            f"Available methods: {available}"
        ) from exc

    mask = method(hsv_features, image, config)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    return np.where(mask > 0, 255, 0).astype(np.uint8)

