from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class TennisBallDetection:
    class_id: int
    confidence: float
    bbox: list[float]
    center: list[float]
    width: float
    height: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "width": self.width,
            "height": self.height,
        }


class TennisBallDetector:
    def __init__(
        self,
        model_path: str | Path = "models/tennis_ball_best.pt",
        conf: float = 0.25,
        min_box_size: float = 3.0,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.min_box_size = min_box_size
        self.device = self._select_device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _select_device(device: str) -> str:
        normalized = device.lower()
        if normalized == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has CPU-only PyTorch. "
                "Install a CUDA-enabled PyTorch build first."
            )
        return device

    def detect(self, image: np.ndarray) -> list[dict]:
        if image is None or image.size == 0:
            return []

        image_h, image_w = image.shape[:2]
        results = self.model(image, conf=self.conf, device=self.device, verbose=False)
        detections: list[dict] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                x1 = max(0.0, min(float(image_w - 1), x1))
                y1 = max(0.0, min(float(image_h - 1), y1))
                x2 = max(0.0, min(float(image_w - 1), x2))
                y2 = max(0.0, min(float(image_h - 1), y2))
                width = x2 - x1
                height = y2 - y1

                if width < self.min_box_size or height < self.min_box_size:
                    continue

                detection = TennisBallDetection(
                    class_id=class_id,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    width=width,
                    height=height,
                )
                detections.append(detection.to_dict())

        return detections


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        center_x, center_y = detection["center"]
        confidence = detection["confidence"]

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        center = (int(round(center_x)), int(round(center_y)))

        cv2.rectangle(output, p1, p2, color, 2)
        cv2.circle(output, center, 4, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"tennis_ball {confidence:.2f}",
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


