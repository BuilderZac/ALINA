from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

ROI_METHODS = (
    "openai",
    "wide",
    "medium",
    "medium_modded",
    "narrow",
    "left",
    "right",
)

THRESHOLD_METHODS = (
    "manual",
    "kmeans",
    "gmm",
    "random_forest",
    "xgboost",
    "openai",
)

DEFAULT_VIDEOS = ("vidd_1", "vidd_2", "vidd_3")

INPUT_DIRS = {
    "vidd_1": PROJECT_ROOT / "data/benchmark_resized/vidd_1",
    "vidd_2": PROJECT_ROOT / "data/benchmark_resized/vidd_2",
    "vidd_3": PROJECT_ROOT / "data/benchmark_resized/vidd_3",
}

REFERENCE_DIRS = {
    "vidd_1": PROJECT_ROOT / "data/Labeled_Data/vidd_1/textfiles",
    "vidd_2": PROJECT_ROOT / "data/Labeled_Data/vidd_2/textfiles",
    "vidd_3": PROJECT_ROOT / "data/Labeled_Data/vidd_3/textfiles",
}

OUTPUT_ROOT = PROJECT_ROOT / "outputs/master_tests"

FLOAT_RE = r"([0-9]+(?:\.[0-9]+)?)"

METRIC_PATTERNS = {
    "precision": re.compile(rf"precision[^0-9]*{FLOAT_RE}", re.I),
    "recall": re.compile(rf"recall[^0-9]*{FLOAT_RE}", re.I),
    "f1": re.compile(rf"\bf1(?:[- ]score)?[^0-9]*{FLOAT_RE}", re.I),
}

TIMING_PATTERNS = {
    "average_ms": re.compile(rf"avg\s+{FLOAT_RE}ms/frame", re.I),
    "fps": re.compile(rf"\(({FLOAT_RE})\s*fps\)", re.I),
    "total_seconds": re.compile(rf"total\s+{FLOAT_RE}s", re.I),
    "labeled_frames": re.compile(r"Done:\s*(\d+)/(\d+)\s+labeled", re.I),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_number(pattern: re.Pattern[str], text: str) -> float | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def parse_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for name, pattern in METRIC_PATTERNS.items():
        value = parse_number(pattern, text)
        if value is not None and value > 1:
            value /= 100.0
        metrics[name] = value

    metrics["average_ms"] = parse_number(
        TIMING_PATTERNS["average_ms"],
        text,
    )
    metrics["fps"] = parse_number(TIMING_PATTERNS["fps"], text)
    metrics["total_seconds"] = parse_number(
        TIMING_PATTERNS["total_seconds"],
        text,
    )

    labeled_match = list(
        TIMING_PATTERNS["labeled_frames"].finditer(text),
    )
    if labeled_match:
        match = labeled_match[-1]
        metrics["labeled_frames"] = int(match.group(1))
        metrics["total_frames"] = int(match.group(2))
    else:
        metrics["labeled_frames"] = None
        metrics["total_frames"] = None

    return metrics


def run_command(
    command: list[str],
    log_path: Path,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = now()
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    finished = now()

    output = process.stdout or ""

    # Opening with "w" intentionally replaces the previous method log.
    log_path.write_text(
        "\n".join(
            [
                f"started: {started}",
                f"finished: {finished}",
                f"return_code: {process.returncode}",
                f"command: {' '.join(command)}",
                "",
                output,
            ],
        ),
        encoding="utf-8",
    )

    print(output, end="")
    return process.returncode, output


def validate_inputs(videos: list[str]) -> None:
    for video in videos:
        if video not in INPUT_DIRS:
            raise ValueError(f"Unknown video: {video}")

        for path in (INPUT_DIRS[video], REFERENCE_DIRS[video]):
            if not path.is_dir():
                raise FileNotFoundError(path)


def evaluate(
    method_root: Path,
    videos: list[str],
) -> dict[str, Any]:
    canny_dirs = [str(REFERENCE_DIRS[video]) for video in videos]
    alina_dirs = [
        str(method_root / video / "textfiles")
        for video in videos
    ]

    command = [
        PYTHON,
        "cli.py",
        "evaluate",
        "--canny-dirs",
        *canny_dirs,
        "--alina-dirs",
        *alina_dirs,
    ]

    evaluation_log = method_root / "evaluation.log"
    return_code, output = run_command(command, evaluation_log)

    result = {
        "return_code": return_code,
        "metrics": parse_metrics(output),
        "log": str(evaluation_log.relative_to(PROJECT_ROOT)),
    }

    (method_root / "evaluation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def run_roi_method(
    method: str,
    roi_mode: str,
    videos: list[str],
    output_root: Path,
    threshold_model_path: Path | None,
) -> dict[str, Any]:
    method_root = output_root / "roi" / roi_mode / method
    method_root.mkdir(parents=True, exist_ok=True)

    video_results = []

    for video in videos:
        video_root = method_root / video
        annotation_dir = video_root / "annotations"
        textfile_dir = video_root / "textfiles"
        timing_log = video_root / "timing.log"

        command = [
            PYTHON,
            "cli.py",
            "label",
            "--input-dir",
            str(INPUT_DIRS[video]),
            "--output-images-dir",
            str(annotation_dir),
            "--output-coords-dir",
            str(textfile_dir),
            "--roi-method",
            method,
            "--roi-mode",
            roi_mode,
            "--threshold-method",
            "manual",
            "--log-file",
            str(timing_log),
        ]

        if threshold_model_path is not None:
            command.extend(
                [
                    "--threshold-model-path",
                    str(threshold_model_path),
                ],
            )

        return_code, output = run_command(
            command,
            video_root / "run.log",
        )

        video_results.append(
            {
                "video": video,
                "return_code": return_code,
                "metrics": parse_metrics(output),
                "log": str(
                    (video_root / "run.log").relative_to(PROJECT_ROOT),
                ),
            },
        )

    evaluation = evaluate(method_root, videos)

    result = {
        "created_at": now(),
        "suite": "roi",
        "roi_mode": roi_mode,
        "method": method,
        "videos": video_results,
        "evaluation": evaluation,
    }

    (method_root / "run.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def run_threshold_method(
    method: str,
    threshold_mode: str,
    videos: list[str],
    output_root: Path,
    random_forest_model: Path,
    xgboost_model: Path,
    threshold_seed: int,
) -> dict[str, Any]:
    method_root = (
        output_root
        / "threshold"
        / threshold_mode
        / method
    )
    method_root.mkdir(parents=True, exist_ok=True)

    model_path: Path | None = None
    if method == "random_forest":
        model_path = random_forest_model
    elif method == "xgboost":
        model_path = xgboost_model

    video_results = []

    for video in videos:
        video_root = method_root / video
        annotation_dir = video_root / "annotations"
        textfile_dir = video_root / "textfiles"
        timing_log = video_root / "timing.log"

        command = [
            PYTHON,
            "cli.py",
            "label",
            "--input-dir",
            str(INPUT_DIRS[video]),
            "--output-images-dir",
            str(annotation_dir),
            "--output-coords-dir",
            str(textfile_dir),
            "--roi-method",
            "medium_modded",
            "--roi-mode",
            "first",
            "--threshold-method",
            method,
            "--threshold-mode",
            threshold_mode,
            "--threshold-seed",
            str(threshold_seed),
            "--log-file",
            str(timing_log),
        ]

        if model_path is not None:
            command.extend(
                [
                    "--threshold-model-path",
                    str(model_path),
                ],
            )

        return_code, output = run_command(
            command,
            video_root / "run.log",
        )

        video_results.append(
            {
                "video": video,
                "return_code": return_code,
                "metrics": parse_metrics(output),
                "log": str(
                    (video_root / "run.log").relative_to(PROJECT_ROOT),
                ),
            },
        )

    evaluation = evaluate(method_root, videos)

    result = {
        "created_at": now(),
        "suite": "threshold",
        "roi_method": "medium_modded",
        "threshold_mode": threshold_mode,
        "method": method,
        "threshold_seed": threshold_seed,
        "videos": video_results,
        "evaluation": evaluation,
    }

    (method_root / "run.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def select_methods(
    requested: str | None,
    all_methods: bool,
    available: tuple[str, ...],
) -> list[str]:
    if all_methods:
        return list(available)

    if requested is None:
        raise ValueError("Use --method METHOD or --all.")

    if requested not in available:
        raise ValueError(
            f"Unknown method '{requested}'. "
            f"Available: {', '.join(available)}",
        )

    return [requested]


def command_run(args: argparse.Namespace) -> None:
    videos = args.videos or list(DEFAULT_VIDEOS)
    validate_inputs(videos)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    results = []

    if args.suite in ("roi", "both"):
        methods = select_methods(
            args.method,
            args.all,
            ROI_METHODS,
        )

        for method in methods:
            results.append(
                run_roi_method(
                    method=method,
                    roi_mode=args.roi_mode,
                    videos=videos,
                    output_root=output_root,
                    threshold_model_path=(
                        Path(args.threshold_model_path)
                        if args.threshold_model_path
                        else None
                    ),
                ),
            )

    if args.suite in ("threshold", "both"):
        methods = select_methods(
            args.method,
            args.all,
            THRESHOLD_METHODS,
        )

        for method in methods:
            results.append(
                run_threshold_method(
                    method=method,
                    threshold_mode=args.threshold_mode,
                    videos=videos,
                    output_root=output_root,
                    random_forest_model=Path(
                        args.random_forest_model,
                    ),
                    xgboost_model=Path(args.xgboost_model),
                    threshold_seed=args.threshold_seed,
                ),
            )

    print(f"Completed {len(results)} test group(s).")


def load_run_files(root: Path) -> list[dict[str, Any]]:
    records = []

    for path in sorted(root.rglob("run.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        record["run_file"] = str(path.relative_to(PROJECT_ROOT))
        records.append(record)

    return records


def command_master(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    records = load_run_files(root)

    if not records:
        raise SystemExit(
            f"No completed tests found below {root}.",
        )

    master_path = Path(args.output)
    if not master_path.is_absolute():
        master_path = PROJECT_ROOT / master_path

    master_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": now(),
        "project_root": str(PROJECT_ROOT),
        "tests_root": str(root),
        "test_count": len(records),
        "tests": records,
    }

    master_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved master JSON log: {master_path}")


def choose_from_menu(
    title: str,
    options: tuple[str, ...],
    allow_all: bool = True,
) -> str | None:
    print(f"\n{title}")

    menu_options = list(options)
    if allow_all:
        menu_options.append("all")

    for index, option in enumerate(menu_options, start=1):
        print(f"{index}. {option}")

    while True:
        choice = input("Select an option: ").strip()

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(menu_options):
                selected = menu_options[index]
                return None if selected == "all" else selected

        print("Invalid selection. Try again.")


def choose_roi_mode() -> str:
    print("\nROI execution mode")
    print("1. First-frame ROI reused for the batch")
    print("2. New ROI selected for every image")

    while True:
        choice = input("Select an option: ").strip()

        if choice == "1":
            return "first"
        if choice == "2":
            return "per-image"

        print("Invalid selection. Try again.")


def choose_videos() -> list[str]:
    print("\nVideo selection")
    print("1. All videos")
    print("2. Select videos individually")

    while True:
        choice = input("Select an option: ").strip()

        if choice == "1":
            return list(DEFAULT_VIDEOS)

        if choice == "2":
            selected: list[str] = []

            for video in DEFAULT_VIDEOS:
                answer = input(f"Run {video}? [y/N]: ").strip().lower()
                if answer in {"y", "yes"}:
                    selected.append(video)

            if selected:
                return selected

            print("Select at least one video.")

        print("Invalid selection. Try again.")


def create_run_namespace(
    suite: str,
    method: str | None,
    run_all: bool,
    roi_mode: str,
    threshold_mode: str,
    videos: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        suite=suite,
        method=method,
        all=run_all,
        roi_mode=roi_mode,
        threshold_mode=threshold_mode,
        videos=videos,
        output_root=str(OUTPUT_ROOT),
        random_forest_model=str(
            PROJECT_ROOT / "random_forest.joblib",
        ),
        xgboost_model=str(
            PROJECT_ROOT / "xgboost.joblib",
        ),
        threshold_model_path=None,
        threshold_seed=42,
    )


def choose_threshold_mode() -> str:
    print("\nThreshold execution mode")
    print("1. First image: learn threshold once and reuse it")
    print("2. Every image: learn threshold independently")

    while True:
        choice = input("Select an option: ").strip()

        if choice == "1":
            return "first"

        if choice == "2":
            return "per-image"

        print("Invalid selection. Try again.")


def interactive_roi_test() -> None:
    roi_mode = choose_roi_mode()

    method = choose_from_menu(
        "ROI methods",
        ROI_METHODS,
        allow_all=True,
    )

    videos = choose_videos()

    args = create_run_namespace(
        suite="roi",
        method=method,
        run_all=method is None,
        roi_mode=roi_mode,
        threshold_mode="first",
        videos=videos,
    )

    command_run(args)


def interactive_threshold_test() -> None:
    threshold_mode = choose_threshold_mode()

    method = choose_from_menu(
        "Threshold methods",
        THRESHOLD_METHODS,
        allow_all=True,
    )

    videos = choose_videos()

    args = create_run_namespace(
        suite="threshold",
        method=method,
        run_all=method is None,
        roi_mode="first",
        threshold_mode=threshold_mode,
        videos=videos,
    )

    command_run(args)


def interactive_all_roi_test() -> None:
    roi_mode = choose_roi_mode()
    videos = choose_videos()

    args = create_run_namespace(
        suite="roi",
        method=None,
        run_all=True,
        roi_mode=roi_mode,
        threshold_mode="first",
        videos=videos,
    )

    command_run(args)


def interactive_all_threshold_test() -> None:
    threshold_mode = choose_threshold_mode()
    videos = choose_videos()

    args = create_run_namespace(
        suite="threshold",
        method=None,
        run_all=True,
        roi_mode="first",
        threshold_mode=threshold_mode,
        videos=videos,
    )

    command_run(args)


def interactive_master_log() -> None:
    output_root = PROJECT_ROOT / "outputs/master_tests"
    output_path = output_root / "master.json"

    args = argparse.Namespace(
        output_root=str(output_root),
        output=str(output_path),
    )

    command_master(args)


def interactive_menu() -> None:
    while True:
        print("\n==============================")
        print(" ALINA MASTER TEST MENU")
        print("==============================")
        print("1. Test one ROI method")
        print("2. Test one threshold method")
        print("3. Batch test all ROI methods")
        print("4. Batch test all threshold methods")
        print("5. Create master JSON log")
        print("6. Exit")

        choice = input("\nSelect an option: ").strip()

        try:
            match choice:
                case "1":
                    interactive_roi_test()
                case "2":
                    interactive_threshold_test()
                case "3":
                    interactive_all_roi_test()
                case "4":
                    interactive_all_threshold_test()
                case "5":
                    interactive_master_log()
                case "6":
                    print("Exiting.")
                    return
                case _:
                    print("Invalid selection. Try again.")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"\nTest failed: {exc}")

        input("\nPress Enter to return to the main menu...")


def main() -> None:
    interactive_menu()


if __name__ == "__main__":
    main()

