from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from alina.color import normalize_color_features
from alina.roi import ROI_METHODS, roi_points_to_quad
from alina.threshold import _feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised ALINA pixel threshold model."
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing training JPG frames.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing CBEM coordinate TXT files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("random_forest.joblib"),
        help="Output model path. Default: random_forest.joblib",
    )
    parser.add_argument(
        "--model",
        choices=("random_forest", "xgboost"),
        default="random_forest",
        help="Classifier to train.",
    )
    parser.add_argument(
        "--roi-method",
        choices=tuple(ROI_METHODS),
        default="medium_modded",
        help="Fixed ROI method used during training and inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help="Number of background pixels per positive pixel.",
    )
    parser.add_argument(
        "--max-samples-per-image",
        type=int,
        default=30_000,
        help="Maximum training pixels sampled from each image.",
    )

    return parser.parse_args()


def load_coordinates(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.float32)

    coordinates = np.loadtxt(path, dtype=np.float32)

    if coordinates.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    coordinates = np.asarray(coordinates).reshape(-1, 2)
    return coordinates


def make_training_mask(
    label_coordinates: np.ndarray,
    perspective_matrix: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(label_coordinates) == 0:
        return mask

    original_points = label_coordinates.reshape(-1, 1, 2)

    warped_points = cv2.perspectiveTransform(
        original_points,
        perspective_matrix,
    ).reshape(-1, 2)

    warped_points = np.rint(warped_points).astype(np.int32)

    valid = (
        (warped_points[:, 0] >= 0)
        & (warped_points[:, 0] < width)
        & (warped_points[:, 1] >= 0)
        & (warped_points[:, 1] < height)
    )

    valid_points = warped_points[valid]
    mask[valid_points[:, 1], valid_points[:, 0]] = 1

    # CBEM stores edge pixels. Dilating them makes the classifier learn
    # the paint region instead of only a one-pixel edge.
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def sample_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    negative_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)

    if len(positive_indices) == 0:
        return (
            np.empty((0, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    max_positive = max(1, int(max_samples / (1.0 + negative_ratio)))
    positive_count = min(len(positive_indices), max_positive)

    positive_indices = rng.choice(
        positive_indices,
        size=positive_count,
        replace=False,
    )

    desired_negative_count = min(
        len(negative_indices),
        int(positive_count * negative_ratio),
    )

    negative_indices = rng.choice(
        negative_indices,
        size=desired_negative_count,
        replace=False,
    )

    selected_indices = np.concatenate(
        (positive_indices, negative_indices)
    )
    rng.shuffle(selected_indices)

    return (
        features[selected_indices].astype(np.float32),
        labels[selected_indices].astype(np.uint8),
    )


def create_model(model_name: str, seed: int) -> object:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()

    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)

    if not args.labels_dir.is_dir():
        raise FileNotFoundError(args.labels_dir)

    image_paths = sorted(args.images_dir.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No JPG images found in {args.images_dir}")

    roi_selector = ROI_METHODS[args.roi_method]
    roi_points = roi_selector(image_paths[0], [])
    roi_quad = roi_points_to_quad(
        roi_points,
        dtype=np.float32,
    )

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise RuntimeError(f"Could not read {image_paths[0]}")

    dst = np.array(
        [
            [50, 800],
            [50, 100],
            [1200, 100],
            [1200, 800],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)

    perspective_matrix = cv2.getPerspectiveTransform(
        roi_quad,
        dst,
    )

    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    processed = 0
    skipped = 0

    for image_index, image_path in enumerate(image_paths):
        label_path = args.labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        coordinates = load_coordinates(label_path)

        if len(coordinates) == 0:
            skipped += 1
            continue

        warped = cv2.warpPerspective(
            image,
            perspective_matrix,
            (image.shape[1], image.shape[0]),
        )

        hsv_features = normalize_color_features(warped)
        features = _feature_matrix(
            hsv_features,
            warped,
        )

        line_mask = make_training_mask(
            coordinates,
            perspective_matrix,
            image.shape,
        )

        flat_labels = line_mask.reshape(-1)
        sampled_features, sampled_labels = sample_pixels(
            features,
            flat_labels,
            max_samples=args.max_samples_per_image,
            negative_ratio=args.negative_ratio,
            seed=args.seed + image_index,
        )

        if len(sampled_labels) == 0:
            skipped += 1
            continue

        feature_batches.append(sampled_features)
        label_batches.append(sampled_labels)
        processed += 1

        print(
            f"[{processed}] {image_path.name}: "
            f"{len(sampled_labels)} samples, "
            f"{int(sampled_labels.sum())} positive"
        )

    if not feature_batches:
        raise RuntimeError(
            "No training samples were created. "
            "Check image/label filenames and ROI configuration."
        )

    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)

    print(f"Training samples: {len(labels)}")
    print(f"Positive samples: {int(labels.sum())}")
    print(f"Negative samples: {int((labels == 0).sum())}")
    print(f"Skipped frames: {skipped}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    model = create_model(args.model, args.seed)

    print(f"Training {args.model}...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, digits=4))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    print(f"Saved model to: {args.output}")


if __name__ == "__main__":
    main()
