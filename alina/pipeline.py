from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .color import normalize_color_features
from .config import PipelineConfig
from .histogram import calc_histogram
from .roi import roi_points_to_quad
from .traversal import circular_threshold_pixel_discovery_and_traversal

from pathlib import Path
from typing import Callable, Iterable

from .threshold import predict_threshold_mask


@dataclass
class ImageResult:
    filename: str
    had_lines: bool
    num_pixels: int
    elapsed_seconds: float


def process_image(img: np.ndarray, roi_points: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, bool]:
    final_img = img.copy()
    no_lines_img = img.copy()

    # 3.3 perspective transformation -> bird's eye view of the ROI
    roi_quad = roi_points_to_quad(roi_points, dtype=np.float32)
    dst = np.array(
        [
            config.roi.dst_bottom_left,
            config.roi.dst_top_left,
            config.roi.dst_top_right,
            config.roi.dst_bottom_right,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)

    M = cv2.getPerspectiveTransform(roi_quad, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    # 3.4 color feature normalization
    color_features = normalize_color_features(warped)

    # 3.5 configurable thresholding
    mask = predict_threshold_mask(
        hsv_features=color_features,
        image=warped,
        config=config,
    )

    # Only the destination rectangle contains a valid perspective warp.
    # This prevents pixels outside the ROI from affecting thresholding.
    roi_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(
        roi_mask,
        config.roi.dst_top_left,
        config.roi.dst_bottom_right,
        255,
        thickness=-1,
    )
    mask = cv2.bitwise_and(mask, roi_mask)

    if config.mask_ignore_left_columns > 0:
        mask[:, :config.mask_ignore_left_columns] = 0

    # 3.6 histogram analysis -> vertical projection, peak column
    avg_pixel, peak_value = calc_histogram(mask, min_white_pixels=config.min_white_pixels)

    # 3.7 threshold check -> is this peak an actual line marking or noise
    if not avg_pixel or peak_value <= config.peak_pixel_threshold:
        return no_lines_img, np.empty((0, 2), dtype=np.int32), False

    # 3.8 CIRCLEDAT -> traverse from the peak to collect all connected line pixels
    binary = mask.copy()
    binary[avg_pixel[1]][avg_pixel[0]] = 255
    line_marking_pixels = circular_threshold_pixel_discovery_and_traversal(
        binary, avg_pixel[0], avg_pixel[1], config.circular_threshold
    )

    height, width = binary.shape[:2]
    black_img = np.zeros((height, width), dtype=np.uint8)
    for x, y in line_marking_pixels:
        cv2.circle(black_img, (x, y), 1, 255, -1)

    # 3.9 frame unwarping -> map detected pixels back to the original perspective
    Minv = cv2.getPerspectiveTransform(dst, roi_quad)
    unwarped = cv2.warpPerspective(black_img, Minv, (img.shape[1], img.shape[0]))

    line_pixels_only = np.where(unwarped > 0)
    x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]
    coords = np.column_stack((x_coords, y_coords))

    # 3.9 annotate -> mark line marking pixels in red on the original frame
    final_img[line_pixels_only] = (0, 0, 255)
    return final_img, coords, True


def run_batch(
    config: PipelineConfig,
    roi_points: np.ndarray | None = None,
    confirm_roi: bool = True,
    roi_selector: Callable[
        [Path, Iterable[Path]],
        list[tuple[int, int]],
    ] | None = None,
    roi_per_image: bool = False,
) -> list[ImageResult]:
    """Process all JPG images using a fixed or selected ROI."""
    image_filenames = sorted(
        path
        for path in config.input_dir.iterdir()
        if path.suffix.lower() == ".jpg"
    )
    total = len(image_filenames)

    if not image_filenames:
        return []

    if roi_selector is not None and not roi_per_image:
        selected_points = roi_selector(
            image_filenames[0],
            image_filenames[1:],
        )
        roi_points = np.asarray(
            [selected_points],
            dtype=np.int32,
        )

    if roi_points is None and not roi_per_image:
        raise ValueError(
            "A fixed ROI or roi_selector is required."
        )

    if confirm_roi:
        if roi_points is None:
            raise ValueError(
                "ROI confirmation requires fixed ROI points."
            )

        reference_image = cv2.imread(
            str(image_filenames[0]),
        )

        if reference_image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_filenames[0]}"
            )

        reference_quad = roi_points_to_quad(
            roi_points,
            dtype=np.int32,
        )

        from .roi import compute_display_scale

        scale = compute_display_scale(
            reference_image,
            max_display_dim=1080,
        )

        display_image = (
            cv2.resize(
                reference_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            if scale != 1.0
            else reference_image.copy()
        )

        display_quad = (
            (reference_quad * scale).astype(np.int32)
            if scale != 1.0
            else reference_quad
        )

        cv2.polylines(
            display_image,
            display_quad,
            True,
            (0, 0, 255),
            thickness=2,
        )
        cv2.imshow("Confirm ROI", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    config.output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.output_coords_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Labeling Process Initiated!")

    results: list[ImageResult] = []
    log_file = (
        open(config.log_file, "w", encoding="utf-8")
        if config.log_file
        else None
    )

    try:
        for index, path in enumerate(image_filenames, start=1):
            start = time.perf_counter()
            image = cv2.imread(str(path))

            if image is None:
                continue

            current_roi = roi_points

            if roi_selector is not None and roi_per_image:
                selected_points = roi_selector(
                    path,
                    image_filenames,
                )
                current_roi = np.asarray(
                    [selected_points],
                    dtype=np.int32,
                )

            if current_roi is None:
                raise ValueError(
                    f"No ROI available for {path}."
                )

            annotated, coords, had_lines = process_image(
                image,
                current_roi,
                config,
            )
            elapsed = time.perf_counter() - start

            np.savetxt(
                config.output_coords_dir / f"{path.stem}.txt",
                coords,
                fmt="%6d",
            )
            cv2.imwrite(
                str(config.output_images_dir / path.name),
                annotated,
            )

            status = (
                "Labeled"
                if had_lines
                else "No lines found"
            )
            log_line = (
                f"[{index}/{total}] [{status}] {path.name}: "
                f"{len(coords)} px, elapsed {elapsed:.3f}s "
                f"({datetime.now().strftime('%H:%M:%S')})"
            )

            print(log_line)

            if log_file is not None:
                log_file.write(log_line + "\n")
                log_file.flush()

            results.append(
                ImageResult(
                    filename=path.name,
                    had_lines=had_lines,
                    num_pixels=len(coords),
                    elapsed_seconds=elapsed,
                )
            )
    finally:
        if log_file is not None:
            log_file.close()

    labeled_count = sum(
        result.had_lines
        for result in results
    )
    total_elapsed = sum(
        result.elapsed_seconds
        for result in results
    )
    average_ms = (
        total_elapsed / len(results) * 1000
        if results
        else 0.0
    )
    fps = 1000 / average_ms if average_ms else 0.0

    summary = (
        f"Done: {labeled_count}/{len(results)} labeled, "
        f"avg {average_ms:.1f}ms/frame "
        f"({fps:.2f} fps), "
        f"total {total_elapsed:.1f}s"
    )

    print(summary)

    if config.log_file:
        with open(
            config.log_file,
            "a",
            encoding="utf-8",
        ) as summary_file:
            summary_file.write(summary + "\n")

    return results

