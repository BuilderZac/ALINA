from __future__ import annotations

import csv
import json
import re
from pathlib import Path


EVALUATION_RE = {
    "precision": re.compile(r"precision[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.I),
    "recall": re.compile(r"recall[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.I),
    "f1": re.compile(r"\bf1(?:[- ]score)?[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.I),
}

TIMING_RE = {
    "average_ms": re.compile(r"avg\s+([0-9]+(?:\.[0-9]+)?)ms/frame", re.I),
    "fps": re.compile(r"\(([0-9]+(?:\.[0-9]+)?)\s*fps\)", re.I),
    "total_seconds": re.compile(r"total\s+([0-9]+(?:\.[0-9]+)?)s", re.I),
    "labeled_frames": re.compile(r"Done:\s*(\d+)/(\d+)\s+labeled", re.I),
}


def number(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def parse_evaluation(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8", errors="replace")

    values: dict[str, float | None] = {
        name: number(pattern, text)
        for name, pattern in EVALUATION_RE.items()
    }

    # Supports evaluation files that use percentages or fractions.
    for name, value in values.items():
        if value is not None and value > 1.0:
            values[name] = value / 100.0

    return values


def parse_timing(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float | int | None] = {
        "average_ms": number(TIMING_RE["average_ms"], text),
        "fps": number(TIMING_RE["fps"], text),
        "total_seconds": number(TIMING_RE["total_seconds"], text),
        "labeled_frames": None,
        "total_frames": None,
    }

    matches = list(TIMING_RE["labeled_frames"].finditer(text))
    if matches:
        values["labeled_frames"] = int(matches[-1].group(1))
        values["total_frames"] = int(matches[-1].group(2))

    return values


def infer_metadata(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts

    # Expected:
    # threshold_benchmark/<mode>/<method>/<video>/...
    mode = parts[0] if len(parts) >= 1 else "unknown"
    method = parts[1] if len(parts) >= 2 else "unknown"
    video = parts[2] if len(parts) >= 3 else "all"

    return {
        "roi_mode": mode,
        "threshold_method": method,
        "video": video,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/threshold_benchmark"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/threshold_benchmark_summary.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/threshold_benchmark_summary.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []

    for evaluation_path in sorted(args.root.glob("*/*/evaluation.txt")):
        metadata = infer_metadata(evaluation_path, args.root)
        metrics = parse_evaluation(evaluation_path)

        method_root = evaluation_path.parent
        timing_files = sorted(method_root.glob("*/timing.log"))

        timing: dict[str, object] = {}
        for timing_file in timing_files:
            video = timing_file.parent.name
            timing_values = parse_timing(timing_file)

            row = {
                **metadata,
                "video": video,
                "evaluation_file": str(evaluation_path),
                "timing_file": str(timing_file),
                **metrics,
                **timing_values,
            }
            rows.append(row)

        if not timing_files:
            rows.append(
                {
                    **metadata,
                    "evaluation_file": str(evaluation_path),
                    "timing_file": "",
                    **metrics,
                }
            )

    if not rows:
        raise SystemExit(
            f"No benchmark files found below {args.root}. "
            "Run testAllThresh.sh first."
        )

    columns = [
        "roi_mode",
        "threshold_method",
        "video",
        "precision",
        "recall",
        "f1",
        "average_ms",
        "fps",
        "labeled_frames",
        "total_frames",
        "total_seconds",
        "evaluation_file",
        "timing_file",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in columns}
            for row in rows
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(rows)} rows to {args.output_csv}")
    print(f"Saved {len(rows)} records to {args.output_json}")


if __name__ == "__main__":
    main()
