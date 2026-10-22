import argparse
from pathlib import Path

import cv2
import numpy as np

from alina.config import EvaluationConfig, PipelineConfig
from alina.pipeline import process_image, run_batch
from alina.roi import ROI_METHODS, select_roi
from eval.cbem import create_cbem
from eval.evaluate import run_evaluation
from eval.superimpose import overlay_coords

from alina.threshold import available_threshold_methods

def _cmd_label(args: argparse.Namespace) -> None:
    if not args.image and not args.input_dir:
        raise SystemExit(
            "alina label: provide either --input-dir or --image"
        )

    config = PipelineConfig(
        input_dir=Path(args.input_dir or "."),
        output_images_dir=Path(args.output_images_dir),
        output_coords_dir=Path(args.output_coords_dir),
        log_file=(
            Path(args.log_file)
            if args.log_file
            else None
        ),
        peak_pixel_threshold=args.peak_threshold,
        min_white_pixels=args.min_white_pixels,
        circular_threshold=args.circular_threshold,
        yellow_lower=tuple(args.yellow_lower),
        yellow_upper=tuple(args.yellow_upper),
        mask_ignore_left_columns=args.mask_ignore_left_columns,
        threshold_method=args.threshold_method,
        threshold_openai_model=args.threshold_openai_model,
    )

    if args.image:
        image_path = Path(args.image)
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        if args.roi_method == "interactive":
            roi_points = select_roi(str(image_path))
        else:
            selector = ROI_METHODS[args.roi_method]
            roi_points = np.asarray(
                [selector(image_path, [])],
                dtype=np.int32,
            )

        annotated, coords, had_lines = process_image(
            image,
            roi_points,
            config,
        )

        config.output_images_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        config.output_coords_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(config.output_images_dir / image_path.name),
            annotated,
        )
        np.savetxt(
            config.output_coords_dir / f"{image_path.stem}.txt",
            coords,
            fmt="%6d",
        )

        print(
            f"{image_path.name}: "
            f"{'Labeled' if had_lines else 'No lines found'}"
        )
        return

    image_files = sorted(
        Path(args.input_dir).glob("*.jpg"),
    )

    if not image_files:
        raise SystemExit(
            f"No JPG images found in {args.input_dir}"
        )

    if args.roi_method == "interactive":
        roi_points = select_roi(str(image_files[0]))

        run_batch(
            config,
            roi_points=roi_points,
            confirm_roi=True,
        )
        return

    run_batch(
        config,
        roi_selector=ROI_METHODS[args.roi_method],
        roi_per_image=args.roi_mode == "per-image",
        confirm_roi=False,
    )


def _cmd_video_to_frames(args: argparse.Namespace) -> None:
    from alina.io_utils import video_to_frames

    video_to_frames(args.video, args.output_dir)


def _cmd_rotate_frames(args: argparse.Namespace) -> None:
    from alina.io_utils import rotate_frames

    rotate_frames(args.input_dir, args.output_dir, angle=args.angle)


def _cmd_resize_images(args: argparse.Namespace) -> None:
    from alina.io_utils import resize_images

    resize_images(args.input_dir, args.output_dir, width=args.width, height=args.height)


def _cmd_cbem(args: argparse.Namespace) -> None:
    create_cbem(args.image, args.output_dir)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    config = EvaluationConfig(canny_dirs=args.canny_dirs, alina_dirs=args.alina_dirs)
    run_evaluation(config)


def _cmd_superimpose(args: argparse.Namespace) -> None:
    overlay_coords(args.image, args.coords, save_to_disk=args.output_dir is not None, output_dir=args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alina")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("label", help="Detect and label taxiway line markings.")
    p.add_argument("--input-dir", dest="input_dir", default=None)
    p.add_argument("--output-images-dir", dest="output_images_dir", required=True)
    p.add_argument("--output-coords-dir", dest="output_coords_dir", required=True)
    p.add_argument("--log-file", dest="log_file", default=None)
    p.add_argument("--peak-threshold", dest="peak_threshold", type=int, default=50)
    p.add_argument("--min-white-pixels", dest="min_white_pixels", type=int, default=200)
    p.add_argument("--circular-threshold", dest="circular_threshold", type=int, default=15)

    p.add_argument(
        "--threshold-method",
        choices=available_threshold_methods(),
        default="manual",
        help="Color thresholding method.",
    )
    p.add_argument(
        "--threshold-openai-model",
        default="gpt-4.1-mini",
        help="OpenAI model used by the openai threshold method.",
    )

    p.add_argument("--yellow-lower", dest="yellow_lower", type=int, nargs=3, default=[0, 70, 170], metavar=("H", "S", "V"))
    p.add_argument("--yellow-upper", dest="yellow_upper", type=int, nargs=3, default=[255, 255, 255], metavar=("H", "S", "V"))
    p.add_argument("--mask-ignore-left-columns", dest="mask_ignore_left_columns", type=int, default=300)
    p.add_argument("--image", default=None, help="Process a single image instead of the whole --input-dir.")
    p.add_argument("--show", action="store_true", help="Display the result window (single-image mode only).")

    p.add_argument(
        "--roi-method",
        choices=["interactive", *ROI_METHODS.keys()],
        default="interactive",
        help="ROI selection method.",
    )
    p.add_argument(
        "--roi-mode",
        choices=["first", "per-image"],
        default="first",
        help=(
            "Reuse the first ROI or select one ROI "
            "for every image."
        ),
    )

    p.set_defaults(func=_cmd_label)

    p = sub.add_parser("video-to-frames", help="Extract frames from a video file.")
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.set_defaults(func=_cmd_video_to_frames)

    p = sub.add_parser("rotate-frames", help="Rotate a directory of frames.")
    p.add_argument("--input-dir", dest="input_dir", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--angle", type=int, default=180, choices=[90, 180, 270])
    p.set_defaults(func=_cmd_rotate_frames)

    p = sub.add_parser("resize-images", help="Batch-resize a directory of images.")
    p.add_argument("--input-dir", dest="input_dir", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.set_defaults(func=_cmd_resize_images)

    p = sub.add_parser("cbem", help="Interactively create a context-based edge map (CBEM) for one frame.")
    p.add_argument("--image", required=True)
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.set_defaults(func=_cmd_cbem)

    p = sub.add_parser("evaluate", help="Compute precision/recall of ALINA output vs. CBEM ground truth.")
    p.add_argument("--canny-dirs", dest="canny_dirs", nargs="+", required=True)
    p.add_argument("--alina-dirs", dest="alina_dirs", nargs="+", required=True)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("superimpose", help="Overlay detected coordinates on a Canny edge map.")
    p.add_argument("--image", required=True)
    p.add_argument("--coords", required=True)
    p.add_argument("--output-dir", dest="output_dir", default=None, help="If given, saves the overlay images here.")
    p.set_defaults(func=_cmd_superimpose)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
