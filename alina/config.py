from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ROIConfig:
    # Perspective-warp target rectangle.
    dst_top_left: tuple[int, int] = (50, 100)
    dst_top_right: tuple[int, int] = (1200, 100)
    dst_bottom_right: tuple[int, int] = (1200, 800)
    dst_bottom_left: tuple[int, int] = (50, 800)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_images_dir: Path
    output_coords_dir: Path
    log_file: Path | None = None

    peak_pixel_threshold: int = 50
    min_white_pixels: int = 200
    circular_threshold: int = 15

    yellow_lower: tuple[int, int, int] = (0, 70, 170)
    yellow_upper: tuple[int, int, int] = (255, 255, 255)

    threshold_method: str = "manual"
    threshold_mode: str = "first"
    threshold_openai_model: str = "gpt-4.1-mini"

    # Runtime state used when threshold_mode == "first".
    threshold_state: object | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # Path to a trained Random Forest or XGBoost model.
    threshold_model_path: Path | None = None

    # Deterministic settings for ML threshold methods.
    threshold_seed: int = 42
    threshold_clusters: int = 3
    threshold_sample_size: int = 25_000

    # Zero out mask columns 0..mask_ignore_left_columns.
    mask_ignore_left_columns: int = 300

    roi: ROIConfig = field(default_factory=ROIConfig)

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_images_dir = Path(self.output_images_dir)
        self.output_coords_dir = Path(self.output_coords_dir)

        if self.threshold_model_path is not None:
            self.threshold_model_path = Path(self.threshold_model_path)

        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_coords_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file is not None:
            self.log_file = Path(self.log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    canny_dirs: list[Path]
    alina_dirs: list[Path]

    def __post_init__(self) -> None:
        self.canny_dirs = [Path(path) for path in self.canny_dirs]
        self.alina_dirs = [Path(path) for path in self.alina_dirs]
