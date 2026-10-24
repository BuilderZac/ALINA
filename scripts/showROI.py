from pathlib import Path

import cv2
import numpy as np

from alina.roi import ROI_METHODS, roi_points_to_quad

image_path = Path("data/benchmark_resized/vidd_1/00001.jpg")
output_path = Path("outputs/roi_check.jpg")

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

points = ROI_METHODS["medium_modded"](image_path, [])
quad = roi_points_to_quad(points, dtype=np.int32)

print("ROI points:", points)
print("Image shape:", image.shape)

cv2.polylines(
    image,
    quad,
    isClosed=True,
    color=(0, 0, 255),
    thickness=5,
)

for index, point in enumerate(quad[0]):
    x, y = map(int, point)
    cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
    cv2.putText(
        image,
        str(index),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), image)
print(f"Saved: {output_path}")

