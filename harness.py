#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import run_batch

import base64
import json
import re

from openai import OpenAI


# ---------------------------------------------------------------------------
# Dataset and output paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = REPO_ROOT / "data/Raw_Data/vidd_1"

FIRST_IMAGE = IMAGE_DIR / "00001.jpg"

FOLLOWING_IMAGES = sorted(
    path
    for path in IMAGE_DIR.glob("*.jpg")
    if path.name != FIRST_IMAGE.name
)

OUTPUT_ROOT = REPO_ROOT / "outputs/harness_benchmark"

RESULTS_FILE = OUTPUT_ROOT / "benchmark_results.csv"

LABELED_COORDS_DIR = REPO_ROOT / "data/Labeled_Data/vidd_1/textfiles"
LABELED_IMAGES_DIR = REPO_ROOT / "data/Labeled_Data/vidd_1/annotations"


COORDINATE_TOLERANCE_PIXELS = 2

OPENAI_API_KEY_FILE = Path(
    "~/.config/codecompanion/token",
).expanduser()

OPENAI_ROI_MODEL = "gpt-4.1-mini"
OPENAI_ROI_MAX_IMAGE_SIZE = 1600


# ---------------------------------------------------------------------------
# OpenAI Helper methods
# ---------------------------------------------------------------------------


def read_openai_api_key() -> str:
    """Read the OpenAI API token from the configured key file."""
    if not OPENAI_API_KEY_FILE.exists():
        raise FileNotFoundError(
            f"OpenAI API key file not found: {OPENAI_API_KEY_FILE}"
        )

    api_key = OPENAI_API_KEY_FILE.read_text(
        encoding="utf-8",
    ).strip()

    if not api_key:
        raise ValueError(
            f"OpenAI API key file is empty: {OPENAI_API_KEY_FILE}"
        )

    return api_key


def extract_json_object(text: str) -> dict[str, object]:
    """Extract a JSON object from a model response."""
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

        if match is None:
            raise ValueError(
                f"OpenAI response did not contain valid JSON: {text}"
            )

        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("OpenAI ROI response must be a JSON object.")

    return result


def validate_roi_points(
    points: object,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Validate and normalize four ROI points."""
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("OpenAI must return exactly four ROI points.")

    normalized: list[tuple[int, int]] = []

    for point in points:
        if (
            not isinstance(point, list | tuple)
            or len(point) != 2
        ):
            raise ValueError(
                "Each ROI point must contain exactly two coordinates."
            )

        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"ROI point {(x, y)} is outside image bounds "
                f"{width}x{height}."
            )

        normalized.append((x, y))

    return normalized


# ---------------------------------------------------------------------------
# ROI-selection methods
# ---------------------------------------------------------------------------

ROISelector = Callable[
    [Path, Iterable[Path]],
    list[tuple[int, int]],
]


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Return image width and height."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    return width, height


def roi_wide(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Wide trapezoid baseline."""
    del following_images

    width, height = read_image_size(first_image)

    return [
        (int(width * 0.03), int(height * 0.97)),
        (int(width * 0.30), int(height * 0.53)),
        (int(width * 0.70), int(height * 0.53)),
        (int(width * 0.97), int(height * 0.97)),
    ]


def roi_medium(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Medium-width trapezoid baseline."""
    del following_images

    width, height = read_image_size(first_image)

    return [
        (int(width * 0.08), int(height * 0.95)),
        (int(width * 0.35), int(height * 0.56)),
        (int(width * 0.65), int(height * 0.56)),
        (int(width * 0.92), int(height * 0.95)),
    ]


def roi_narrow(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Narrow trapezoid baseline."""
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
    """ROI candidate biased toward markings on the left side."""
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
    """ROI candidate biased toward markings on the right side."""
    del following_images

    width, height = read_image_size(first_image)

    return [
        (int(width * 0.22), int(height * 0.96)),
        (int(width * 0.42), int(height * 0.55)),
        (int(width * 0.78), int(height * 0.55)),
        (int(width * 0.98), int(height * 0.96)),
    ]


def roi_openai(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Select a taxiway ROI using an OpenAI vision model."""
    del following_images

    image = cv2.imread(str(first_image))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {first_image}"
        )

    original_height, original_width = image.shape[:2]

    scale = min(
        1.0,
        OPENAI_ROI_MAX_IMAGE_SIZE / max(original_width, original_height),
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

    success, encoded_image = cv2.imencode(
        ".jpg",
        display_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not success:
        raise RuntimeError("Could not encode image for OpenAI.")

    image_data = base64.b64encode(
        encoded_image.tobytes(),
    ).decode("ascii")

    prompt = f"""
You are selecting a trapezoidal region of interest for ALINA, an
airport taxiway line-marking annotation algorithm.

Inspect the attached aircraft taxiway image and return the ROI covering
the visible taxiway surface where yellow taxiway line markings are
expected. EXCLUDE as much of the aircraft dashboard, aircraft body, aircraft
engine, windshield frame, sky, and irrelevant side scenery as possible.
Unwanted details should be outside of the box created with the coordinates
passed back.

The ROI must be a perspective trapezoid:
- its bottom edge should be near the lower taxiway foreground;
- its top edge should be around the distant taxiway region;
- it should be wide enough to include possible left and right markings;
- it should not be as narrow as a single visible line.

Use the medium baseline as the initial prior. For reference, its
normalized points are approximately:
bottom-left=(0.08W, 0.95H)
top-left=(0.35W, 0.56H)
top-right=(0.65W, 0.56H)
bottom-right=(0.92W, 0.95H)

The displayed image dimensions are width={display_width}, height={display_height}.
Return exactly this JSON format and no Markdown:
{{
  "points": [
    [bottom_left_x, bottom_left_y],
    [top_left_x, top_left_y],
    [top_right_x, top_right_y],
    [bottom_right_x, bottom_right_y]
  ]
}}

Coordinates must be integer pixel coordinates in the displayed image.
MAKE NO MISTAKES.
"""

    client = OpenAI(api_key=read_openai_api_key())

    response = client.chat.completions.create(
        model=OPENAI_ROI_MODEL,
        temperature=0,
        top_p=1,
        seed=42,
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
                                f"data:image/jpeg;base64,{image_data}"
                            ),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("OpenAI returned an empty ROI response.")

    result = extract_json_object(response_text)
    model_points = result.get("points")

    display_points = validate_roi_points(
        points=model_points,
        width=display_width,
        height=display_height,
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


# Add or replace entries with real ML/VLM methods.
ROI_METHODS: dict[str, ROISelector] = {
    "openai": roi_openai,
    "wide_baseline": roi_wide,
    "medium_baseline": roi_medium,
    "narrow_baseline": roi_narrow,
    "left_biased": roi_left_biased,
    "right_biased": roi_right_biased,
}


# ---------------------------------------------------------------------------
# ROI conversion
# ---------------------------------------------------------------------------

def convert_roi_for_alina(
    roi_points: list[tuple[int, int]],
) -> np.ndarray:
    """Convert four public ROI points to ALINA's expected shape (1, 6, 2)."""
    if len(roi_points) != 4:
        raise ValueError(
            "ROI selector must return exactly four points."
        )

    bottom_left, top_left, top_right, bottom_right = [
        (int(x), int(y))
        for x, y in roi_points
    ]

    return np.array(
        [[
            bottom_left,   # index 0: used by ALINA
            top_left,      # index 1: used by ALINA
            top_left,      # index 2: unused
            top_right,     # index 3: used by ALINA
            top_right,     # index 4: unused
            bottom_right,  # index 5: used by ALINA
        ]],
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# Coordinate-file utilities
# ---------------------------------------------------------------------------

def read_coordinate_file(path: Path) -> set[tuple[int, int]]:
    """Read whitespace- or comma-separated x/y coordinate files."""
    coordinates: set[tuple[int, int]] = set()

    if not path.exists():
        return coordinates

    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.replace(",", " ").split()

        if len(fields) < 2:
            continue

        try:
            x = int(float(fields[0]))
            y = int(float(fields[1]))
        except ValueError:
            continue

        coordinates.add((x, y))

    return coordinates


def scale_coordinates(
    coordinates: set[tuple[int, int]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> set[tuple[int, int]]:
    """Scale coordinates from the labeled-image resolution."""
    source_width, source_height = source_size
    target_width, target_height = target_size

    if (source_width, source_height) == (target_width, target_height):
        return coordinates

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    return {
        (round(x * scale_x), round(y * scale_y))
        for x, y in coordinates
    }


def coordinate_metrics(
    predicted: set[tuple[int, int]],
    ground_truth: set[tuple[int, int]],
    tolerance: int = COORDINATE_TOLERANCE_PIXELS,
) -> tuple[float, float, float]:
    """Calculate coordinate precision, recall, and F1."""
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0

    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0

    predicted_array = np.asarray(list(predicted), dtype=np.int32)
    ground_truth_array = np.asarray(list(ground_truth), dtype=np.int32)

    distances_squared = (
        predicted_array[:, None, :] - ground_truth_array[None, :, :]
    ) ** 2
    distances_squared = distances_squared.sum(axis=2)

    tolerance_squared = tolerance * tolerance
    candidate_pairs = np.argwhere(distances_squared <= tolerance_squared)

    # Match closest pairs first, with each point used at most once.
    candidate_pairs = sorted(
        candidate_pairs,
        key=lambda pair: distances_squared[pair[0], pair[1]],
    )

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    true_positives = 0

    for prediction_index, truth_index in candidate_pairs:
        if prediction_index in matched_predictions:
            continue
        if truth_index in matched_ground_truth:
            continue

        matched_predictions.add(int(prediction_index))
        matched_ground_truth.add(int(truth_index))
        true_positives += 1

    precision = true_positives / len(predicted)
    recall = true_positives / len(ground_truth)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_output_directory(
    output_coords_dir: Path,
) -> tuple[float | None, float | None, float | None, int]:
    """Compare predictions against the provided labeled coordinate files."""
    reference_files = {
        path.name: path
        for path in LABELED_COORDS_DIR.glob("*.txt")
    }

    target_size = read_image_size(FIRST_IMAGE)

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for filename, reference_path in reference_files.items():
        prediction_path = output_coords_dir / filename

        if not prediction_path.exists():
            continue

        reference_image = LABELED_IMAGES_DIR / filename.replace(
            ".txt",
            ".jpg",
        )

        if reference_image.exists():
            reference_size = read_image_size(reference_image)
        else:
            reference_size = target_size

        reference = read_coordinate_file(reference_path)
        reference = scale_coordinates(
            coordinates=reference,
            source_size=reference_size,
            target_size=target_size,
        )

        prediction = read_coordinate_file(prediction_path)

        precision, recall, f1 = coordinate_metrics(
            predicted=prediction,
            ground_truth=reference,
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    if not f1_values:
        return None, None, None, 0

    return (
        float(np.mean(precision_values)),
        float(np.mean(recall_values)),
        float(np.mean(f1_values)),
        len(f1_values),
    )


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def run_alina_benchmark(
    method_name: str,
    roi_selector: ROISelector,
    first_image: Path,
    following_images: Iterable[Path],
    roi_mode: int,
) -> dict[str, object]:
    """Run ALINA with either one ROI or one ROI per image."""
    method_output = OUTPUT_ROOT / method_name

    if method_output.exists():
        shutil.rmtree(method_output)

    output_images = method_output / "annotated"
    output_coords = method_output / "coords"
    timing_log = method_output / "timing.log"

    output_images.mkdir(parents=True, exist_ok=True)
    output_coords.mkdir(parents=True, exist_ok=True)
    timing_log.parent.mkdir(parents=True, exist_ok=True)

    image_paths = [
        Path(first_image),
        *[Path(path) for path in following_images],
    ]

    roi_prediction_seconds = 0.0
    selected_rois: list[list[tuple[int, int]]] = []

    with tempfile.TemporaryDirectory(
        prefix=f"alina_harness_{method_name}_",
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)

        if roi_mode == 1:
            temporary_input = temporary_root / "all_frames"
            temporary_input.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                shutil.copy2(
                    image_path,
                    temporary_input / image_path.name,
                )

            roi_start = time.perf_counter()
            roi_points = roi_selector(
                first_image,
                image_paths[1:],
            )
            roi_prediction_seconds += time.perf_counter() - roi_start
            selected_rois.append(roi_points)

            config = PipelineConfig(
                input_dir=temporary_input,
                output_images_dir=output_images,
                output_coords_dir=output_coords,
            )

            print()
            print(f"Running method: {method_name}")
            print("ROI mode: first image reused")
            print(f"ROI points: {roi_points}")

            start_time = time.perf_counter()

            run_batch(
                config=config,
                roi_points=convert_roi_for_alina(roi_points),
                confirm_roi=True,
            )

            elapsed_seconds = time.perf_counter() - start_time

        else:
            print()
            print(f"Running method: {method_name}")
            print("ROI mode: one ROI per image")
            print("Manual ROI confirmation is bypassed.")

            start_time = time.perf_counter()

            for image_path in image_paths:
                temporary_input = temporary_root / image_path.stem
                temporary_input.mkdir(parents=True, exist_ok=True)

                shutil.copy2(
                    image_path,
                    temporary_input / image_path.name,
                )

                roi_start = time.perf_counter()
                roi_points = roi_selector(
                    image_path,
                    image_paths,
                )
                roi_prediction_seconds += time.perf_counter() - roi_start
                selected_rois.append(roi_points)

                config = PipelineConfig(
                    input_dir=temporary_input,
                    output_images_dir=output_images,
                    output_coords_dir=output_coords,
                )

                run_batch(
                    config=config,
                    roi_points=convert_roi_for_alina(roi_points),
                    confirm_roi=False,
                )

            elapsed_seconds = time.perf_counter() - start_time

    image_count = len(image_paths)
    milliseconds_per_image = (
        elapsed_seconds * 1000.0 / image_count
        if image_count
        else 0.0
    )

    precision, recall, f1, evaluated_frames = (
        evaluate_output_directory(output_coords)
    )

    width, height = read_image_size(first_image)

    roi_areas = [
        abs(float(cv2.contourArea(np.asarray(roi, dtype=np.float32))))
        for roi in selected_rois
    ]

    average_roi_area = (
        float(np.mean(roi_areas))
        if roi_areas
        else 0.0
    )

    return {
        "method": method_name,
        "images": image_count,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "roi_prediction_milliseconds": round(
            roi_prediction_seconds * 1000.0,
            3,
        ),
        "milliseconds_per_image": round(
            milliseconds_per_image,
            3,
        ),
        "frames_evaluated": evaluated_frames,
        "precision": (
            round(precision, 5)
            if precision is not None
            else ""
        ),
        "recall": (
            round(recall, 5)
            if recall is not None
            else ""
        ),
        "f1": (
            round(f1, 5)
            if f1 is not None
            else ""
        ),
        "image_width": width,
        "image_height": height,
        "roi_area_pixels": round(average_roi_area, 2),
        "roi_points": selected_rois[0] if roi_mode == 1 else "per-image",
    }


def write_results(results: list[dict[str, object]]) -> None:
    """Write benchmark results to CSV."""
    if not results:
        return

    fieldnames = [
        "method",
        "images",
        "elapsed_seconds",
        "roi_prediction_milliseconds",
        "milliseconds_per_image",
        "frames_evaluated",
        "precision",
        "recall",
        "f1",
        "image_width",
        "image_height",
        "roi_area_pixels",
        "roi_points",
    ]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run every ROI method and save a benchmark table."""
    if not FIRST_IMAGE.exists():
        raise FileNotFoundError(
            f"Missing first image: {FIRST_IMAGE}"
        )

    missing_images = [
        path
        for path in FOLLOWING_IMAGES
        if not path.exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"Missing following images: {missing_images}"
        )

    while True:
        roi_mode_text = input(
            "\nROI selection mode:\n"
            "  1 - Select ROI from first image and reuse it\n"
            "  2 - Select ROI independently for every image\n"
            "Choose 1 or 2: "
        ).strip()

        if roi_mode_text in {"1", "2"}:
            roi_mode = int(roi_mode_text)
            break

        print("Invalid choice. Enter 1 or 2.")

    print(f"First image: {FIRST_IMAGE}")
    print(f"Following images: {len(FOLLOWING_IMAGES)}")
    print(f"Methods: {', '.join(ROI_METHODS)}")

    results: list[dict[str, object]] = []

    for method_name, roi_selector in ROI_METHODS.items():
        result = run_alina_benchmark(
            method_name=method_name,
            roi_selector=roi_selector,
            first_image=FIRST_IMAGE,
            following_images=FOLLOWING_IMAGES,
            roi_mode=roi_mode,
        )

        results.append(result)

        print(
            f"{method_name}: "
            f"ALINA={result['milliseconds_per_image']} ms/image, "
            f"ROI={result['roi_prediction_milliseconds']} ms total, "
            f"precision={result['precision']}, "
            f"recall={result['recall']}, "
            f"F1={result['f1']}"
        )

    write_results(results)

    print()
    print(f"Benchmark results written to: {RESULTS_FILE}")
    print(f"Detailed outputs written to: {OUTPUT_ROOT}")


main()
