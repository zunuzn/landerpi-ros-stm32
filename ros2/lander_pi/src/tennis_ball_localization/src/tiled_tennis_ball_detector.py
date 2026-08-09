from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .tennis_ball_detector import TennisBallDetector


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def non_max_suppression(
    detections: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Remove duplicate detections caused by overlapping tiles."""
    kept: list[dict] = []
    for detection in sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        if all(
            detection["class_id"] != other["class_id"]
            or box_iou(detection["bbox"], other["bbox"]) < iou_threshold
            for other in kept
        ):
            kept.append(detection)
    return kept


class TiledTennisBallDetector:
    def __init__(
        self,
        model_path: str | Path = "models/tennis_ball_best.pt",
        conf: float = 0.25,
        tile_size: int = 640,
        overlap: int = 160,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.detector = TennisBallDetector(
            model_path=model_path,
            conf=conf,
            min_box_size=3.0,
            device=device,
        )
        self.device = self.detector.device

    def _starts(self, length: int) -> list[int]:
        if length < self.tile_size:
            raise ValueError(
                f"Image dimension {length} is smaller than tile size {self.tile_size}"
            )

        stride = self.tile_size - self.overlap
        starts = list(range(0, length - self.tile_size + 1, stride))
        last_start = length - self.tile_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _tile_origins(self, image: np.ndarray) -> list[tuple[int, int]]:
        image_height, image_width = image.shape[:2]
        x_starts = self._starts(image_width)
        y_starts = self._starts(image_height)
        return [(x, y) for y in y_starts for x in x_starts]

    def detect(self, image: np.ndarray) -> list[dict]:
        if image is None or image.size == 0:
            return []

        image_height, image_width = image.shape[:2]
        all_detections: list[dict] = []

        for offset_x, offset_y in self._tile_origins(image):
            tile = image[
                offset_y : offset_y + self.tile_size,
                offset_x : offset_x + self.tile_size,
            ]
            tile_detections = self.detector.detect(tile)

            for detection in tile_detections:
                x1, y1, x2, y2 = detection["bbox"]
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                x1 = max(0.0, min(float(image_width - 1), x1))
                y1 = max(0.0, min(float(image_height - 1), y1))
                x2 = max(0.0, min(float(image_width - 1), x2))
                y2 = max(0.0, min(float(image_height - 1), y2))

                all_detections.append(
                    {
                        **detection,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )

        return non_max_suppression(all_detections, self.iou_threshold)


def draw_tiled_regions(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: int = 160,
) -> np.ndarray:
    output = image.copy()
    detector = object.__new__(TiledTennisBallDetector)
    detector.tile_size = tile_size
    detector.overlap = overlap

    for offset_x, offset_y in detector._tile_origins(image):
        cv2.rectangle(
            output,
            (offset_x, offset_y),
            (offset_x + tile_size - 1, offset_y + tile_size - 1),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            output,
            f"tile ({offset_x},{offset_y})",
            (offset_x + 8, min(output.shape[0] - 8, offset_y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output
