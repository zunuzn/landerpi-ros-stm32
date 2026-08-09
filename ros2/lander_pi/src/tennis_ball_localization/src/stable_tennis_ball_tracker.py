from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np


@dataclass
class _Track:
    track_id: int
    class_id: int
    bbox: list[float]
    center: list[float]
    confidence: float
    age: int = 1
    miss_count: int = 0
    hits: int = 1


class StableTennisBallTracker:
    """Keep a short-lived identity for every detected tennis ball.

    This is a sensing-layer tracker. It does not choose a navigation target.
    Tracks are matched by center-point distance and lightly smoothed with an
    exponential moving average. A track can remain in the output for a few
    missed frames, but ``valid`` tells downstream code whether this frame has
    a real YOLO observation suitable for depth alignment.
    """

    def __init__(
        self,
        max_match_distance: float = 60.0,
        max_missed: int = 5,
        smooth_alpha: float = 0.65,
    ) -> None:
        if max_match_distance <= 0:
            raise ValueError("max_match_distance must be greater than zero")
        if max_missed < 0:
            raise ValueError("max_missed must not be negative")
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in the range (0, 1]")

        self.max_match_distance = float(max_match_distance)
        self.max_missed = int(max_missed)
        self.smooth_alpha = float(smooth_alpha)
        self._next_track_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update(self, detections: list[dict]) -> list[dict]:
        """Update all ball tracks and return stable results for this frame.

        Each returned dictionary keeps pixel coordinates in the same image
        coordinate system as the input detection. ``valid=True`` means that
        YOLO detected the ball in the current frame. ``valid=False`` is a
        short prediction retained only to bridge a temporary missed frame.
        """
        normalized = [self._normalize_detection(detection) for detection in detections]
        normalized = [detection for detection in normalized if detection is not None]

        matched_track_indexes: set[int] = set()
        matched_detection_indexes: set[int] = set()
        valid_track_ids: set[int] = set()

        # Greedy nearest-neighbour matching is enough for small, well-separated
        # tennis balls and keeps this layer easy to replace later.
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(normalized):
                distance = hypot(
                    track.center[0] - detection["center"][0],
                    track.center[1] - detection["center"][1],
                )
                if distance <= self.max_match_distance:
                    candidates.append((distance, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_track_indexes or detection_index in matched_detection_indexes:
                continue

            self._update_matched_track(self._tracks[track_index], normalized[detection_index])
            matched_track_indexes.add(track_index)
            matched_detection_indexes.add(detection_index)
            valid_track_ids.add(self._tracks[track_index].track_id)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indexes:
                track.age += 1
                track.miss_count += 1
                track.confidence *= 0.9

        for detection_index, detection in enumerate(normalized):
            if detection_index not in matched_detection_indexes:
                new_track = self._new_track(detection)
                self._tracks.append(new_track)
                valid_track_ids.add(new_track.track_id)

        self._tracks = [track for track in self._tracks if track.miss_count <= self.max_missed]
        return [
            self._track_to_dict(track, valid=track.track_id in valid_track_ids)
            for track in self._tracks
        ]

    @staticmethod
    def _normalize_detection(detection: dict) -> dict | None:
        try:
            bbox = [float(value) for value in detection["bbox"]]
            center = [float(value) for value in detection["center"]]
            confidence = float(detection["confidence"])
        except (KeyError, TypeError, ValueError):
            return None

        if len(bbox) != 4 or len(center) != 2:
            return None

        return {
            "class_id": int(detection.get("class_id", 0)),
            "bbox": bbox,
            "center": center,
            "confidence": confidence,
            "width": float(detection.get("width", bbox[2] - bbox[0])),
            "height": float(detection.get("height", bbox[3] - bbox[1])),
        }

    def _new_track(self, detection: dict) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            class_id=detection["class_id"],
            bbox=detection["bbox"],
            center=detection["center"],
            confidence=detection["confidence"],
        )
        self._next_track_id += 1
        return track

    def _update_matched_track(self, track: _Track, detection: dict) -> None:
        track.class_id = detection["class_id"]
        alpha = self.smooth_alpha
        track.bbox = self._smooth(track.bbox, detection["bbox"], alpha)
        track.center = self._smooth(track.center, detection["center"], alpha)
        track.confidence = alpha * detection["confidence"] + (1.0 - alpha) * track.confidence
        track.age += 1
        track.miss_count = 0
        track.hits += 1

    @staticmethod
    def _smooth(previous: list[float], current: list[float], alpha: float) -> list[float]:
        return [
            (1.0 - alpha) * old_value + alpha * new_value
            for old_value, new_value in zip(previous, current)
        ]

    @staticmethod
    def _track_to_dict(track: _Track, valid: bool) -> dict:
        x1, y1, x2, y2 = track.bbox
        return {
            "track_id": track.track_id,
            "class_id": track.class_id,
            "confidence": track.confidence,
            "bbox": track.bbox.copy(),
            "center": track.center.copy(),
            "center_x": track.center[0],
            "center_y": track.center[1],
            "width": x2 - x1,
            "height": y2 - y1,
            "valid": valid,
            "age": track.age,
            "hits": track.hits,
            "miss_count": track.miss_count,
        }


def draw_stable_tracks(
    image: np.ndarray,
    tracks: list[dict],
    show_invalid: bool = True,
) -> np.ndarray:
    """Draw stable tracks without changing the image or track dictionaries."""
    output = image.copy()

    for track in tracks:
        if not show_invalid and not track["valid"]:
            continue

        x1, y1, x2, y2 = track["bbox"]
        center_x, center_y = track["center"]
        valid = track["valid"]
        color = (0, 255, 0) if valid else (0, 165, 255)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        center = (int(round(center_x)), int(round(center_y)))

        cv2.rectangle(output, p1, p2, color, 2)
        cv2.circle(output, center, 4, (0, 0, 255), -1)
        state = "observed" if valid else f"predicted ({track['miss_count']})"
        cv2.putText(
            output,
            f"ball #{track['track_id']} {track['confidence']:.2f} {state}",
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
