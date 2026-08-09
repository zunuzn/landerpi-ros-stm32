from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_ball_depth_locator import (
    CameraIntrinsics,
    ESTIMATED_AURORA930_640X400_INTRINSICS,
    TennisBallDepthLocator,
)
from src.tennis_ball_detector import TennisBallDetector


@dataclass(frozen=True)
class SavedFrame:
    index: int
    timestamp: str
    rgb_path: Path
    depth_path: Path


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def read_frame_index(csv_path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), str(row["timestamp"])))
    return rows


def collect_saved_frames(data_dir: Path) -> list[SavedFrame]:
    rgb_rows = read_frame_index(data_dir / "Rgb.csv")
    depth_rows = read_frame_index(data_dir / "Depth.csv")
    depth_by_index = {index: timestamp for index, timestamp in depth_rows}

    frames: list[SavedFrame] = []
    for index, rgb_timestamp in rgb_rows:
        depth_timestamp = depth_by_index.get(index)
        if depth_timestamp is None:
            continue

        rgb_path = data_dir / "Rgb" / f"Rgb_{index}_{rgb_timestamp}.jpg"
        depth_path = data_dir / "Depth" / f"Depth_{index}_{depth_timestamp}.png"
        if rgb_path.exists() and depth_path.exists():
            frames.append(SavedFrame(index, rgb_timestamp, rgb_path, depth_path))

    return frames


def draw_result(image: np.ndarray, detection: dict) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    center_x, center_y = detection["center"]
    sample_x = detection.get("sample_x")
    sample_y = detection.get("sample_y")
    depth_m = detection.get("depth_m")

    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    center = (int(round(center_x)), int(round(center_y)))
    cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
    cv2.circle(image, center, 4, (0, 0, 255), -1)
    if sample_x is not None and sample_y is not None:
        cv2.circle(image, (int(round(sample_x)), int(round(sample_y))), 4, (255, 0, 0), -1)

    depth_text = "no depth" if depth_m is None else f"depth {depth_m:.3f}m"
    label = f"ball {detection['confidence']:.2f} {depth_text}"
    cv2.putText(
        image,
        label,
        (p1[0], max(20, p1[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO + depth lookup on Deptrum Scope saved data.")
    parser.add_argument("--data-dir", required=True, help="Deptrum saved data directory containing Rgb/Depth CSV files.")
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Path to YOLO model.")
    parser.add_argument("--out", default="outputs/deptrum_saved_depth", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda:0, or 0.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth raw value to meters scale.")
    parser.add_argument("--patch-radius", type=int, default=3, help="Depth patch radius in pixels.")
    parser.add_argument(
        "--sample-point",
        default="center",
        choices=["center", "bottom_center"],
        help="Pixel used for depth lookup.",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=ESTIMATED_AURORA930_640X400_INTRINSICS.fx,
        help="Camera focal length fx in pixels.",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=ESTIMATED_AURORA930_640X400_INTRINSICS.fy,
        help="Camera focal length fy in pixels.",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=ESTIMATED_AURORA930_640X400_INTRINSICS.cx,
        help="Camera principal point cx in pixels.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=ESTIMATED_AURORA930_640X400_INTRINSICS.cy,
        help="Camera principal point cy in pixels.",
    )
    parser.add_argument("--save-images", action="store_true", help="Save annotated detection images.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick tests. 0 means all.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_saved_frames(data_dir)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise FileNotFoundError(f"No paired RGB/depth frames found in {data_dir}")

    detector = TennisBallDetector(model_path=args.model, conf=args.conf, device=args.device)
    locator = TennisBallDepthLocator(
        intrinsics=CameraIntrinsics(args.fx, args.fy, args.cx, args.cy),
        depth_scale=args.depth_scale,
        patch_radius=args.patch_radius,
        sample_point=args.sample_point,
    )

    print(f"Inference device: {detector.device}")
    print(f"Paired frames: {len(frames)}")
    print(f"Camera intrinsics: fx={args.fx:.2f}, fy={args.fy:.2f}, cx={args.cx:.2f}, cy={args.cy:.2f}")

    rows: list[dict[str, str]] = []
    for frame in frames:
        rgb = imread_unicode(frame.rgb_path, cv2.IMREAD_COLOR)
        depth = imread_unicode(frame.depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"Skip unreadable frame {frame.index}")
            continue
        if depth.ndim != 2:
            print(f"Skip non-single-channel depth frame {frame.index}: {depth.shape}")
            continue

        detections = detector.detect(rgb)
        located_detections = locator.locate_all(detections, depth)
        annotated = rgb.copy()

        for det_index, detection in enumerate(located_detections, start=1):
            center_x, center_y = detection["center"]
            draw_result(annotated, detection)

            x1, y1, x2, y2 = detection["bbox"]
            rows.append(
                {
                    "frame_index": str(frame.index),
                    "timestamp": frame.timestamp,
                    "detection_index": str(det_index),
                    "confidence": f"{detection['confidence']:.4f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "center_x": f"{center_x:.2f}",
                    "center_y": f"{center_y:.2f}",
                    "sample_point": detection["sample_point"],
                    "sample_x": fmt(detection["sample_x"]),
                    "sample_y": fmt(detection["sample_y"]),
                    "depth_raw_median": f"{detection['depth_raw_median']:.2f}",
                    "depth_m": fmt(detection["depth_m"]),
                    "camera_x_m": fmt(detection["camera_x_m"]),
                    "camera_y_m": fmt(detection["camera_y_m"]),
                    "camera_z_m": fmt(detection["camera_z_m"]),
                    "depth_samples": str(detection["depth_samples"]),
                    "depth_aligned": str(detection["depth_aligned"]),
                    "rgb_path": str(frame.rgb_path),
                    "depth_path": str(frame.depth_path),
                }
            )

        if args.save_images and located_detections:
            out_path = out_dir / f"frame_{frame.index:04d}_{frame.timestamp}_detected.jpg"
            imwrite_unicode(out_path, annotated)

        print(f"frame {frame.index}: {len(located_detections)} detections")

    csv_path = out_dir / "detections_with_depth.csv"
    fieldnames = [
        "frame_index",
        "timestamp",
        "detection_index",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "sample_point",
        "sample_x",
        "sample_y",
        "depth_raw_median",
        "depth_m",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "depth_samples",
        "depth_aligned",
        "rgb_path",
        "depth_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
