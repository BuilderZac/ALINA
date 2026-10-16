#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import run_batch


# ---------------------------------------------------------------------------
# Hard-coded test inputs
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

FIRST_IMAGE = (
    REPO_ROOT
    / "data/Raw_Data/vidd_1/00001.jpg"
)

FOLLOWING_IMAGES = sorted(
    (REPO_ROOT / "data/Raw_Data/vidd_1").glob("*.jpg")
)

OUTPUT_IMAGES = (
    REPO_ROOT
    / "outputs/harness/annotated"
)

OUTPUT_COORDS = (
    REPO_ROOT
    / "outputs/harness/coords"
)

LOG_FILE = (
    REPO_ROOT
    / "outputs/harness/timing.log"
)


# ---------------------------------------------------------------------------
# ML integration point
# ---------------------------------------------------------------------------

def predict_roi(
    first_image: Path,
    following_images: Iterable[Path],
) -> list[tuple[int, int]]:
    """Return four ROI points from an ML method.

    Parameters
    ----------
    first_image:
        The first image in the sequence.

    following_images:
        Images following first_image. A temporal ML method can use these
        images. A single-frame method can ignore them.

    Returns
    -------
    list[tuple[int, int]]
        Four points in this order:

        bottom-left, top-left, top-right, bottom-right
    """
    del following_images

    image = cv2.imread(str(first_image))
    if image is None:
        raise FileNotFoundError(
            f"Could not read first image: {first_image}"
        )

    height, width = image.shape[:2]

    # Replace this block with an ML/VLM model call.
    return [
        (int(width * 0.05), int(height * 0.95)),  # bottom-left
        (int(width * 0.35), int(height * 0.55)),  # top-left
        (int(width * 0.65), int(height * 0.55)),  # top-right
        (int(width * 0.95), int(height * 0.95)),  # bottom-right
    ]


# ---------------------------------------------------------------------------
# ROI conversion
# ---------------------------------------------------------------------------

def convert_roi_for_alina(
    roi_points: list[tuple[int, int]],
) -> np.ndarray:
    """Convert four public ROI points to ALINA's expected array format.

    alina.roi.roi_points_to_quad() expects an array with shape:

        (1, 6, 2)

    It uses indexes 0, 1, 3, and 5 to construct the quadrilateral.
    """
    if len(roi_points) != 4:
        raise ValueError(
            "predict_roi() must return exactly four points."
        )

    bottom_left, top_left, top_right, bottom_right = [
        (int(x), int(y))
        for x, y in roi_points
    ]

    return np.array(
        [[
            bottom_left,   # index 0: used
            top_left,      # index 1: used
            top_left,      # index 2: unused
            top_right,     # index 3: used
            top_right,     # index 4: unused
            bottom_right,  # index 5: used
        ]],
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# ALINA execution
# ---------------------------------------------------------------------------

def run_alina(
    first_image: Path,
    following_images: Iterable[Path],
    roi_points: list[tuple[int, int]],
) -> None:
    """Run ALINA only on the explicitly selected images."""
    image_paths = [
        Path(first_image),
        *[Path(path) for path in following_images],
    ]

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_COORDS.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    alina_roi = convert_roi_for_alina(roi_points)

    # run_batch() accepts an input directory, not an image list. A temporary
    # directory limits this test run to the hard-coded images.
    with tempfile.TemporaryDirectory(
        prefix="alina_harness_"
    ) as temporary_directory:
        temporary_input = Path(temporary_directory)

        for image_path in image_paths:
            destination = temporary_input / image_path.name
            shutil.copy2(image_path, destination)

        config = PipelineConfig(
            input_dir=temporary_input,
            output_images_dir=OUTPUT_IMAGES,
            output_coords_dir=OUTPUT_COORDS,
        )

        # run_batch() displays a confirmation window. Press any key while
        # that window is focused to start processing.
        run_batch(
            config=config,
            roi_points=alina_roi,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Validate inputs, obtain the ROI, and run ALINA."""
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

    roi_points = predict_roi(
        FIRST_IMAGE,
        FOLLOWING_IMAGES,
    )

    print(f"First image: {FIRST_IMAGE}")
    print(f"Following images: {len(FOLLOWING_IMAGES)}")
    print(f"ROI points: {roi_points}")

    run_alina(
        FIRST_IMAGE,
        FOLLOWING_IMAGES,
        roi_points,
    )

    print(f"Annotated images: {OUTPUT_IMAGES}")
    print(f"Coordinate files: {OUTPUT_COORDS}")


if __name__ == "__main__":
    main()
