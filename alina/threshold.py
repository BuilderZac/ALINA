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

    cached_bounds = getattr(config, "threshold_state", None)

    if (
        not _threshold_is_per_image(config)
        and cached_bounds is not None
    ):
        return apply_hsv_bounds(hsv_features, cached_bounds)


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

    if not _threshold_is_per_image(config):
        config.threshold_state = bounds

    return apply_hsv_bounds(hsv_features, bounds)


def _feature_matrix(
    hsv_features: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """Create deterministic pixel features for ML thresholding."""
    hsv = hsv_features.reshape(-1, 3).astype(np.float32) / 255.0
    bgr = image.reshape(-1, 3).astype(np.float32) / 255.0

    # Include normalized HSV and BGR. HSV is useful for color; BGR
    # helps distinguish yellow and white markings under illumination changes.
    return np.concatenate((hsv, bgr), axis=1)


def _sample_rows(
    features: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if len(features) <= sample_size:
        return features

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=sample_size, replace=False)
    return features[indices]


def _cluster_line_mask(
    labels: np.ndarray,
    centers: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Keep only clusters with plausible yellow or white paint colors."""
    saturation = centers[:, 1]
    value = centers[:, 2]
    hue = centers[:, 0]

    # OpenCV hue was normalized from [0, 179] to [0, 1].
    yellow_cluster = (
        (hue >= 0.08)
        & (hue <= 0.22)
        & (saturation >= 0.35)
        & (value >= 0.35)
    )

    # White paint is bright and nearly unsaturated.
    # Keep this stricter to avoid selecting the aircraft dash.
    white_cluster = (
        (saturation <= 0.22)
        & (value >= 0.62)
    )

    selected = yellow_cluster | white_cluster

    if not np.any(selected):
        return np.zeros(image_shape, dtype=np.uint8)

    selected_labels = np.flatnonzero(selected)
    mask = np.isin(labels, selected_labels).reshape(image_shape)

    # Remove the lowest portion of the warped ROI, where the aircraft
    # dash is most likely to appear.
    height = image_shape[0]
    mask[int(height * 0.88):, :] = False

    mask = (mask.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )


def _threshold_is_per_image(config: object) -> bool:
    return str(
        getattr(config, "threshold_mode", "first"),
    ).lower() == "per-image"


def _get_or_fit_kmeans(
    features: np.ndarray,
    config: object,
) -> object:
    from sklearn.cluster import KMeans

    if (
        not _threshold_is_per_image(config)
        and getattr(config, "threshold_state", None) is not None
    ):
        return config.threshold_state

    sampled = _sample_rows(
        features,
        int(getattr(config, "threshold_sample_size", 25_000)),
        int(getattr(config, "threshold_seed", 42)),
    )

    clusters = int(getattr(config, "threshold_clusters", 4))
    clusters = max(2, min(clusters, len(sampled)))

    model = KMeans(
        n_clusters=clusters,
        random_state=int(getattr(config, "threshold_seed", 42)),
        n_init=10,
    )
    model.fit(sampled)

    if not _threshold_is_per_image(config):
        config.threshold_state = model

    return model


def kmeans_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """Fit K-means once or independently for every frame."""
    del hsv_features

    height, width = image.shape[:2]
    raw_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv = np.empty_like(raw_hsv)
    hsv[..., 0] = raw_hsv[..., 0] / 179.0
    hsv[..., 1] = raw_hsv[..., 1] / 255.0
    hsv[..., 2] = raw_hsv[..., 2] / 255.0

    features = hsv.reshape(-1, 3)
    model = _get_or_fit_kmeans(features, config)
    labels = model.predict(features)

    return _cluster_line_mask(
        labels=labels,
        centers=model.cluster_centers_,
        image_shape=(height, width),
    )


def _get_or_fit_gmm(
    features: np.ndarray,
    config: object,
) -> object:
    from sklearn.mixture import GaussianMixture

    if (
        not _threshold_is_per_image(config)
        and getattr(config, "threshold_state", None) is not None
    ):
        return config.threshold_state

    sampled = _sample_rows(
        features,
        int(getattr(config, "threshold_sample_size", 25_000)),
        int(getattr(config, "threshold_seed", 42)),
    )

    clusters = int(getattr(config, "threshold_clusters", 3))
    clusters = max(2, min(clusters, len(sampled)))

    model = GaussianMixture(
        n_components=clusters,
        covariance_type="full",
        random_state=int(getattr(config, "threshold_seed", 42)),
        reg_covar=1e-5,
    )
    model.fit(sampled)

    if not _threshold_is_per_image(config):
        config.threshold_state = model

    return model


def gmm_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """Fit GMM once or independently for every frame."""
    del hsv_features

    height, width = image.shape[:2]
    raw_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    normalized_hsv = np.empty_like(raw_hsv)
    normalized_hsv[..., 0] = raw_hsv[..., 0] / 179.0
    normalized_hsv[..., 1] = raw_hsv[..., 1] / 255.0
    normalized_hsv[..., 2] = raw_hsv[..., 2] / 255.0

    bgr = image.astype(np.float32) / 255.0

    features = np.concatenate(
        (
            normalized_hsv.reshape(-1, 3),
            bgr.reshape(-1, 3),
        ),
        axis=1,
    )

    model = _get_or_fit_gmm(features, config)
    labels = model.predict(features)

    return _cluster_line_mask(
        labels=labels,
        centers=model.means_[:, :3],
        image_shape=(height, width),
    )


def _load_supervised_model(config: object) -> object:
    model_path = getattr(config, "threshold_model_path", None)

    if model_path is None:
        raise ValueError(
            "Supervised thresholding requires "
            "--threshold-model-path."
        )

    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold model does not exist: {path}")

    return joblib.load(path)


def random_forest_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """Predict line-marking pixels using a trained Random Forest."""
    model = _load_supervised_model(config)
    features = _feature_matrix(hsv_features, image)

    labels = model.predict(features)
    return np.where(labels.astype(bool), 255, 0).astype(np.uint8).reshape(
        hsv_features.shape[:2]
    )


def xgboost_threshold(
    hsv_features: np.ndarray,
    image: np.ndarray,
    config: object,
) -> np.ndarray:
    """Predict line-marking pixels using a trained XGBoost model."""
    model = _load_supervised_model(config)
    features = _feature_matrix(hsv_features, image)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        labels = probabilities >= 0.5
    else:
        labels = model.predict(features)

    return np.where(labels, 255, 0).astype(np.uint8).reshape(
        hsv_features.shape[:2]
    )


THRESHOLD_METHODS: dict[str, ThresholdMethod] = {
    "manual": manual_threshold,
    "kmeans": kmeans_threshold,
    "gmm": gmm_threshold,
    "random_forest": random_forest_threshold,
    "xgboost": xgboost_threshold,
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
