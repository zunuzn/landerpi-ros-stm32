from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Literal

import numpy as np


SamplePoint = Literal["center", "bottom_center"]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class DepthSample:
    raw_depth: float
    depth_m: float | None
    sample_count: int
    sample_u: float
    sample_v: float


@dataclass(frozen=True)
class CameraPoint:
    x_m: float | None
    y_m: float | None
    z_m: float | None


ESTIMATED_AURORA930_640X400_INTRINSICS = CameraIntrinsics(
    fx=448.62,
    fy=471.17,
    cx=319.50,
    cy=199.50,
)


class TennisBallDepthLocator:
    """Convert 2D ball detections plus aligned depth into camera coordinates."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = ESTIMATED_AURORA930_640X400_INTRINSICS,
        depth_scale: float = 0.001,
        patch_radius: int = 3,
        sample_point: SamplePoint = "center",
    ) -> None:
        if depth_scale <= 0:
            raise ValueError("depth_scale must be greater than zero")
        if patch_radius < 0:
            raise ValueError("patch_radius must not be negative")
        if sample_point not in ("center", "bottom_center"):
            raise ValueError("sample_point must be 'center' or 'bottom_center'")

        self.intrinsics = intrinsics
        self.depth_scale = float(depth_scale)
        self.patch_radius = int(patch_radius)
        self.sample_point = sample_point

    def locate(self, detection: dict, depth: np.ndarray) -> dict:
        """Return the detection enriched with depth and camera XYZ fields.

        ``valid=False`` tracks from the stability layer are passed through but
        are not depth-aligned, because they were not observed by YOLO this frame.
        """
        located = dict(detection)
        located["depth_aligned"] = False

        if detection.get("valid", True) is False:
            self._fill_empty_depth_fields(located)
            return located

        u, v = self._sample_pixel(detection)
        sample = self.sample_depth(depth, u, v, detection["bbox"])
        point = self.pixel_to_camera(u, v, sample.depth_m)

        located.update(
            {
                "sample_point": self.sample_point,
                "sample_x": sample.sample_u,
                "sample_y": sample.sample_v,
                "depth_raw_median": sample.raw_depth,
                "depth_m": sample.depth_m,
                "camera_x_m": point.x_m,
                "camera_y_m": point.y_m,
                "camera_z_m": point.z_m,
                "depth_samples": sample.sample_count,
                "depth_aligned": sample.depth_m is not None,
            }
        )
        return located

    def locate_all(self, detections: list[dict], depth: np.ndarray) -> list[dict]:
        return [self.locate(detection, depth) for detection in detections]

    def sample_depth(
        self,
        depth: np.ndarray,
        u: float,
        v: float,
        bbox: list[float] | tuple[float, float, float, float],
    ) -> DepthSample:
        if depth is None or depth.size == 0 or depth.ndim != 2:
            return DepthSample(0.0, None, 0, u, v)

        h, w = depth.shape[:2]
        center_x = int(round(u))
        center_y = int(round(v))
        patch = self._clip_patch(depth, center_x, center_y, self.patch_radius)
        valid = patch[patch > 0]

        if valid.size == 0:
            x1, y1, x2, y2 = bbox
            bx1 = max(0, int(floor(x1)))
            by1 = max(0, int(floor(y1)))
            bx2 = min(w, int(ceil(x2)))
            by2 = min(h, int(ceil(y2)))
            patch = depth[by1:by2, bx1:bx2]
            valid = patch[patch > 0]

        if valid.size == 0:
            return DepthSample(0.0, None, 0, u, v)

        raw_depth = float(np.median(valid))
        return DepthSample(
            raw_depth=raw_depth,
            depth_m=raw_depth * self.depth_scale,
            sample_count=int(valid.size),
            sample_u=u,
            sample_v=v,
        )

    def pixel_to_camera(self, u: float, v: float, z_m: float | None) -> CameraPoint:
        if self.intrinsics is None or z_m is None or z_m <= 0:
            return CameraPoint(None, None, z_m if z_m and z_m > 0 else None)

        x_m = (u - self.intrinsics.cx) * z_m / self.intrinsics.fx
        y_m = (v - self.intrinsics.cy) * z_m / self.intrinsics.fy
        return CameraPoint(x_m, y_m, z_m)

    def _clip_patch(self, depth: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray:
        h, w = depth.shape[:2]
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(w, center_x + radius + 1)
        y2 = min(h, center_y + radius + 1)
        return depth[y1:y2, x1:x2]

    def _sample_pixel(self, detection: dict) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        if self.sample_point == "bottom_center":
            return (x1 + x2) / 2.0, y2

        if "center" in detection:
            center = detection["center"]
            return float(center[0]), float(center[1])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _fill_empty_depth_fields(result: dict) -> None:
        result.update(
            {
                "sample_point": "",
                "sample_x": None,
                "sample_y": None,
                "depth_raw_median": 0.0,
                "depth_m": None,
                "camera_x_m": None,
                "camera_y_m": None,
                "camera_z_m": None,
                "depth_samples": 0,
            }
        )
