from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stable_tennis_ball_tracker import StableTennisBallTracker
from src.tennis_ball_depth_locator import (
    CameraIntrinsics,
    ESTIMATED_AURORA930_640X400_INTRINSICS,
    TennisBallDepthLocator,
)
from src.tennis_ball_detector import TennisBallDetector, draw_detections


@dataclass(frozen=True)
class SavedFrame:
    index: int
    timestamp: str
    rgb_path: Path
    depth_path: Path


FIELDNAMES = [
    "frame_index",
    "timestamp",
    "track_id",
    "valid",
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
    "age",
    "hits",
    "miss_count",
    "rgb_path",
    "depth_path",
]


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
    if not csv_path.exists():
        return []

    rows: list[tuple[int, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "index" not in reader.fieldnames or "timestamp" not in reader.fieldnames:
            return rows
        for row in reader:
            try:
                rows.append((int(row["index"]), str(row["timestamp"])))
            except (KeyError, TypeError, ValueError):
                continue
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


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def located_track_to_row(frame: SavedFrame, track: dict) -> dict[str, str]:
    x1, y1, x2, y2 = track["bbox"]
    center_x, center_y = track["center"]
    return {
        "frame_index": str(frame.index),
        "timestamp": frame.timestamp,
        "track_id": str(track.get("track_id", "")),
        "valid": str(track.get("valid", True)),
        "confidence": f"{track['confidence']:.4f}",
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "x2": f"{x2:.2f}",
        "y2": f"{y2:.2f}",
        "center_x": f"{center_x:.2f}",
        "center_y": f"{center_y:.2f}",
        "sample_point": str(track.get("sample_point", "")),
        "sample_x": fmt(track.get("sample_x")),
        "sample_y": fmt(track.get("sample_y")),
        "depth_raw_median": f"{track.get('depth_raw_median', 0.0):.2f}",
        "depth_m": fmt(track.get("depth_m")),
        "camera_x_m": fmt(track.get("camera_x_m")),
        "camera_y_m": fmt(track.get("camera_y_m")),
        "camera_z_m": fmt(track.get("camera_z_m")),
        "depth_samples": str(track.get("depth_samples", 0)),
        "depth_aligned": str(track.get("depth_aligned", False)),
        "age": str(track.get("age", "")),
        "hits": str(track.get("hits", "")),
        "miss_count": str(track.get("miss_count", "")),
        "rgb_path": str(frame.rgb_path),
        "depth_path": str(frame.depth_path),
    }


def draw_located_tracks(image: np.ndarray, tracks: list[dict], show_invalid: bool = True) -> np.ndarray:
    output = image.copy()
    for track in tracks:
        if not show_invalid and not track.get("valid", True):
            continue

        x1, y1, x2, y2 = track["bbox"]
        center_x, center_y = track["center"]
        valid = track.get("valid", True)
        depth_m = track.get("depth_m")
        camera_x_m = track.get("camera_x_m")
        camera_y_m = track.get("camera_y_m")
        camera_z_m = track.get("camera_z_m")
        sample_x = track.get("sample_x")
        sample_y = track.get("sample_y")

        color = (0, 255, 0) if valid else (0, 165, 255)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        center = (int(round(center_x)), int(round(center_y)))
        cv2.rectangle(output, p1, p2, color, 2)
        cv2.circle(output, center, 4, (0, 0, 255), -1)
        if sample_x is not None and sample_y is not None:
            cv2.circle(output, (int(round(sample_x)), int(round(sample_y))), 4, (255, 0, 0), -1)

        state = "observed" if valid else f"predicted ({track.get('miss_count', 0)})"
        if camera_x_m is None or camera_y_m is None or camera_z_m is None:
            xyz_text = "no xyz"
        else:
            xyz_text = f"xyz {camera_x_m:.2f},{camera_y_m:.2f},{camera_z_m:.2f}m"
        depth_text = "no depth" if depth_m is None else f"z {depth_m:.3f}m"
        label = f"#{track.get('track_id', '?')} {track['confidence']:.2f} {state} {depth_text} {xyz_text}"
        cv2.putText(
            output,
            label,
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def process_frame(
    frame: SavedFrame,
    detector: TennisBallDetector,
    tracker: StableTennisBallTracker,
    locator: TennisBallDepthLocator,
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    rgb = imread_unicode(frame.rgb_path, cv2.IMREAD_COLOR)
    depth = imread_unicode(frame.depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or depth.ndim != 2:
        return None

    raw_detections = detector.detect(rgb)
    stable_tracks = tracker.update(raw_detections)
    located_tracks = locator.locate_all(stable_tracks, depth)
    return rgb, raw_detections, located_tracks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch DeptrumScope saved RGB/depth frames and run YOLO + stability + depth localization."
    )
    parser.add_argument("--data-dir", required=True, help="Deptrum saved scene directory containing Rgb/Depth CSV files.")
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Path to YOLO model.")
    parser.add_argument("--out", default="outputs/deptrum_watch", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda:0, or 0.")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="Seconds between directory scans.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames. 0 means run forever.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth raw value to meters scale.")
    parser.add_argument("--patch-radius", type=int, default=3, help="Depth patch radius in pixels.")
    parser.add_argument(
        "--sample-point",
        default="center",
        choices=["center", "bottom_center"],
        help="Pixel used for depth lookup.",
    )
    parser.add_argument("--fx", type=float, default=ESTIMATED_AURORA930_640X400_INTRINSICS.fx)
    parser.add_argument("--fy", type=float, default=ESTIMATED_AURORA930_640X400_INTRINSICS.fy)
    parser.add_argument("--cx", type=float, default=ESTIMATED_AURORA930_640X400_INTRINSICS.cx)
    parser.add_argument("--cy", type=float, default=ESTIMATED_AURORA930_640X400_INTRINSICS.cy)
    parser.add_argument("--max-match-distance", type=float, default=60.0)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--show-raw", action="store_true", help="Draw raw YOLO detections in blue.")
    parser.add_argument("--show-invalid", action="store_true", help="Draw short-lived predicted tracks.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV display window.")
    parser.add_argument("--save-debug", action="store_true", help="Save annotated frames.")
    parser.add_argument("--debug-dir", default=None, help="Directory for annotated debug frames.")
    args = parser.parse_args()

    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be greater than zero")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug_dir else out_dir / "debug_frames"
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    detector = TennisBallDetector(model_path=args.model, conf=args.conf, device=args.device)
    tracker = StableTennisBallTracker(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        smooth_alpha=args.smooth_alpha,
    )
    locator = TennisBallDepthLocator(
        intrinsics=CameraIntrinsics(args.fx, args.fy, args.cx, args.cy),
        depth_scale=args.depth_scale,
        patch_radius=args.patch_radius,
        sample_point=args.sample_point,
    )

    csv_path = out_dir / "detections_with_depth.csv"
    processed_indexes: set[int] = set()
    processed_count = 0
    prev_time = time.perf_counter()

    print(f"Inference device: {detector.device}")
    print(f"Watching: {data_dir}")
    print(f"CSV output: {csv_path}")
    print(f"Camera intrinsics: fx={args.fx:.2f}, fy={args.fy:.2f}, cx={args.cx:.2f}, cy={args.cy:.2f}")
    if not args.no_display:
        print("Press q in the display window to quit.")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        try:
            while True:
                frames = collect_saved_frames(data_dir)
                new_frames = [frame for frame in frames if frame.index not in processed_indexes]

                if not new_frames:
                    time.sleep(args.poll_interval)
                    continue

                for frame in new_frames:
                    result = process_frame(frame, detector, tracker, locator)
                    if result is None:
                        continue

                    rgb, raw_detections, located_tracks = result
                    processed_indexes.add(frame.index)
                    processed_count += 1

                    now = time.perf_counter()
                    fps = 1.0 / max(now - prev_time, 1e-6)
                    prev_time = now

                    display = draw_located_tracks(rgb, located_tracks, show_invalid=args.show_invalid)
                    if args.show_raw:
                        display = draw_detections(display, raw_detections, color=(255, 0, 0))

                    observed_count = sum(1 for track in located_tracks if track.get("valid", True))
                    aligned_count = sum(1 for track in located_tracks if track.get("depth_aligned", False))
                    cv2.putText(
                        display,
                        f"frame {frame.index}  FPS {fps:.1f}  observed {observed_count}  xyz {aligned_count}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    for track in located_tracks:
                        writer.writerow(located_track_to_row(frame, track))
                    f.flush()

                    if args.save_debug and located_tracks:
                        out_path = debug_dir / f"frame_{frame.index:04d}_{frame.timestamp}_detected.jpg"
                        imwrite_unicode(out_path, display)

                    print(
                        f"frame {frame.index}: raw={len(raw_detections)} "
                        f"tracks={len(located_tracks)} observed={observed_count} xyz={aligned_count}"
                    )

                    if not args.no_display:
                        cv2.imshow("deptrum_tennis_ball_depth", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return 0

                    if args.max_frames > 0 and processed_count >= args.max_frames:
                        return 0

                time.sleep(args.poll_interval)
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
