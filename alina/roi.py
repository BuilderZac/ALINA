from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2

ROISelector = Callable[
    [Path, Iterable[Path]],
    list[tuple[int, int]],
]


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    return width, height


def _baseline_roi(
    image_path: Path,
    width_fraction: float,
    top_fraction: float,
    bottom_left: float,
    bottom_right: float,
) -> list[tuple[int, int]]:
    width, height = read_image_size(image_path)

    half_width = width_fraction / 2.0
    center = 0.5

    return [
        (int(width * (center - half_width)), int(height * bottom_left)),
        (int(width * (center - top_fraction)), int(height * 0.56)),
        (int(width * (center + top_fraction)), int(height * 0.56)),
        (int(width * (center + half_width)), int(height * bottom_right)),
    ]


def roi_wide(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    return _baseline_roi(first_image, 0.94, 0.20, 0.97, 0.97)


def roi_medium(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    width, height = read_image_size(first_image)

    return [
        (int(width * 0.08), int(height * 0.95)),
        (int(width * 0.35), int(height * 0.56)),
        (int(width * 0.65), int(height * 0.56)),
        (int(width * 0.92), int(height * 0.95)),
    ]


def roi_medium_modded(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    width, height = read_image_size(first_image)

    return [
        (int(width * 0.16), int(height * 0.70)),
        (int(width * 0.36), int(height * 0.52)),
        (int(width * 0.64), int(height * 0.52)),
        (int(width * 0.84), int(height * 0.70)),
    ]


def roi_narrow(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    width, height = read_image_size(first_image)

    return [
        (int(width * 0.18), int(height * 0.95)),
        (int(width * 0.40), int(height * 0.58)),
        (int(width * 0.60), int(height * 0.58)),
        (int(width * 0.82), int(height * 0.95)),
    ]


def roi_left_biased(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    width, height = read_image_size(first_image)

    return [
        (int(width * 0.02), int(height * 0.96)),
        (int(width * 0.22), int(height * 0.55)),
        (int(width * 0.58), int(height * 0.55)),
        (int(width * 0.78), int(height * 0.96)),
    ]


def roi_right_biased(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    del following_images
    width, height = read_image_size(first_image)

    return [
        (int(width * 0.22), int(height * 0.96)),
        (int(width * 0.42), int(height * 0.55)),
        (int(width * 0.78), int(height * 0.55)),
        (int(width * 0.98), int(height * 0.96)),
    ]


def _extract_json_object(text: str) -> dict[str, object]:
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("ROI model response did not contain JSON.")
        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("ROI model response must be a JSON object.")

    return result


def _validate_roi_points(
    points: object,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("ROI selector must return exactly four points.")

    normalized = []
    for point in points:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("Each ROI point must contain two coordinates.")

        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"ROI point {(x, y)} is outside image bounds.")

        normalized.append((x, y))

    return normalized


def roi_openai(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Select a taxiway ROI using an OpenAI vision model."""
    del following_images

    from openai import OpenAI

    api_key_file = Path(
        "~/.config/codecompanion/token",
    ).expanduser()
    api_key = api_key_file.read_text(
        encoding="utf-8",
    ).strip()

    if not api_key:
        raise ValueError(
            f"OpenAI API key file is empty: {api_key_file}"
        )

    image = cv2.imread(str(first_image))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {first_image}"
        )

    original_height, original_width = image.shape[:2]
    max_size = 1600

    scale = min(
        1.0,
        max_size / max(original_width, original_height),
    )

    if scale < 1.0:
        display_width = round(original_width * scale)
        display_height = round(original_height * scale)

        display_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_image = image
        display_width = original_width
        display_height = original_height

    success, encoded = cv2.imencode(
        ".jpg",
        display_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not success:
        raise RuntimeError(
            "Could not encode image for ROI selection."
        )

    image_data = base64.b64encode(
        encoded.tobytes(),
    ).decode("ascii")

    prompt = f"""
You are selecting an ROI for detecting yellow airport taxiway markings.

Use ONLY the lower taxiway pavement. Do not select the sky, clouds,
aircraft windshield, aircraft dashboard, aircraft body, grass, or the
entire image.

The ROI must be a narrow-to-medium perspective trapezoid:

- bottom-left must be near (0.08W, 0.95H);
- top-left must be near (0.35W, 0.56H);
- top-right must be near (0.65W, 0.56H);
- bottom-right must be near (0.92W, 0.95H);
- both bottom points must have y >= 0.78H;
- both top points must have y >= 0.38H;
- the bottom edge must be wider than the top edge;
- the ROI must remain centered on the taxiway;
- do not use image borders as ROI boundaries.

The displayed image dimensions are exactly:
width={display_width}, height={display_height}

Return exactly four integer points in this order:
bottom-left, top-left, top-right, bottom-right.

Coordinates must refer to the displayed image and must satisfy:
0 <= x < {display_width}
0 <= y < {display_height}

Return JSON only:
{{
  "points": [
    [bottom_left_x, bottom_left_y],
    [top_left_x, top_left_y],
    [top_right_x, top_right_y],
    [bottom_right_x, bottom_right_y]
  ]
}}
"""

    response = OpenAI(
        api_key=api_key,
    ).chat.completions.create(
        model="gpt-4.1-mini",
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
            "OpenAI returned an empty ROI response."
        )

    raw_points = _extract_json_object(content).get("points")

    if not isinstance(raw_points, list) or len(raw_points) != 4:
        raise ValueError(
            "OpenAI must return exactly four ROI points."
        )

    parsed_points: list[tuple[int, int]] = []

    for point in raw_points:
        if (
            not isinstance(point, list | tuple)
            or len(point) != 2
        ):
            raise ValueError(
                "Each OpenAI ROI point must contain two coordinates."
            )

        parsed_points.append(
            (
                int(round(float(point[0]))),
                int(round(float(point[1]))),
            )
        )

    points_fit_display = all(
        0 <= x < display_width
        and 0 <= y < display_height
        for x, y in parsed_points
    )

    points_fit_original = all(
        0 <= x < original_width
        and 0 <= y < original_height
        for x, y in parsed_points
    )

    if points_fit_display:
        display_points = parsed_points
    elif points_fit_original:
        # The model ignored the resized-image dimensions and returned
        # coordinates in the original-image coordinate system.
        if scale == 1.0:
            return parsed_points

        return parsed_points
    else:
        raise ValueError(
            "OpenAI returned ROI points outside both image coordinate "
            f"systems: display={display_width}x{display_height}, "
            f"original={original_width}x{original_height}, "
            f"points={parsed_points}"
        )

    if scale == 1.0:
        return display_points

    return [
        (
            int(round(x / scale)),
            int(round(y / scale)),
        )
        for x, y in display_points
    ]


ROI_METHODS: dict[str, ROISelector] = {
    "openai": roi_openai,
    "wide": roi_wide,
    "medium": roi_medium,
    "medium_modded": roi_medium_modded,
    "narrow": roi_narrow,
    "left": roi_left_biased,
    "right": roi_right_biased,
}


def compute_display_scale(
    image: np.ndarray,
    max_display_dim: int = 1080,
) -> float:
    """Return a scale that fits the image within the display dimension."""
    height, width = image.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension <= max_display_dim:
        return 1.0

    return max_display_dim / largest_dimension


def select_roi(image_path: str) -> np.ndarray:
    """
    Interactively select four ROI points.

    Click in this order:
    bottom-left, top-left, top-right, bottom-right.

    Press Enter after four points have been selected.
    Press R to reset or Escape to cancel.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    scale = compute_display_scale(image)
    display_image = (
        cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale != 1.0
        else image.copy()
    )

    points: list[tuple[int, int]] = []
    window_name = "Select ROI"

    def on_mouse(
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        del flags, param

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(points) >= 4:
            return

        points.append(
            (
                int(round(x / scale)),
                int(round(y / scale)),
            )
        )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frame = display_image.copy()

            display_points = np.asarray(
                [
                    (
                        int(round(x * scale)),
                        int(round(y * scale)),
                    )
                    for x, y in points
                ],
                dtype=np.int32,
            )

            for point in display_points:
                cv2.circle(
                    frame,
                    tuple(point),
                    5,
                    POINT_COLOR,
                    -1,
                )

            if len(display_points) >= 2:
                cv2.polylines(
                    frame,
                    [display_points],
                    False,
                    LINE_COLOR,
                    LINE_THICKNESS,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (10, 13) and len(points) == 4:
                break

            if key == ord("r"):
                points.clear()

            if key == 27:
                raise RuntimeError("ROI selection cancelled.")
    finally:
        cv2.destroyAllWindows()

    return np.asarray([points], dtype=np.int32)


def roi_points_to_quad(
    roi_points: np.ndarray | list[tuple[int, int]],
    dtype=np.int32,
) -> np.ndarray:
    points = np.asarray(roi_points)

    if points.ndim == 3:
        points = points[0]

    if len(points) == 4:
        bottom_left, top_left, top_right, bottom_right = points
    elif len(points) == 6:
        bottom_left = points[0]
        top_left = points[1]
        top_right = points[3]
        bottom_right = points[5]
    else:
        raise ValueError("ROI must contain either four or six points.")

    return np.array(
        [[
            tuple(bottom_left),
            tuple(top_left),
            tuple(top_right),
            tuple(bottom_right),
        ]],
        dtype=dtype,
    )

