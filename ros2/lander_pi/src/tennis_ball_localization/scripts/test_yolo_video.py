from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stable_tennis_ball_tracker import StableTennisBallTracker, draw_stable_tracks
from src.tennis_ball_detector import TennisBallDetector, draw_detections
from src.tiled_tennis_ball_detector import TiledTennisBallDetector


DEFAULT_CAMERA_SOURCE = "0"
DEFAULT_SCAN_LIMIT = 5


def parse_video_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def backend_name(backend: int) -> str:
    names = {
        cv2.CAP_ANY: "any",
        cv2.CAP_DSHOW: "dshow",
        cv2.CAP_MSMF: "msmf",
    }
    return names.get(backend, str(backend))


def selected_backends(name: str) -> list[int]:
    normalized = name.lower()
    if normalized == "any":
        return [cv2.CAP_ANY]
    if normalized == "dshow":
        return [cv2.CAP_DSHOW]
    if normalized == "msmf":
        return [cv2.CAP_MSMF]
    if normalized != "auto":
        raise ValueError(f"Unsupported backend: {name}")

    if platform.system().lower() == "windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_capture(source: int | str, backend: str, width: int = 0, height: int = 0) -> tuple[cv2.VideoCapture, str]:
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        return cap, "file"

    for candidate in selected_backends(backend):
        cap = cv2.VideoCapture(source, candidate)
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            return cap, backend_name(candidate)
        cap.release()

    return cv2.VideoCapture(), "none"


def list_cameras(limit: int, backend: str) -> int:
    print(f"Scanning camera indexes 0..{limit - 1}")
    found = 0

    for index in range(limit):
        cap, used_backend = open_capture(index, backend)
        ok = False
        width = 0
        height = 0

        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                found += 1

        cap.release()

        if ok:
            print(f"[OK] source {index}, backend {used_backend}, frame {width}x{height}")
        else:
            print(f"[--] source {index}")

    if found == 0:
        print("No usable camera found. Check camera connection and Windows camera privacy permissions.")

    return 0 if found else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO tennis ball detection on a camera or video.")
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Path to .pt/.onnx model.")
    parser.add_argument(
        "--source",
        default=DEFAULT_CAMERA_SOURCE,
        help="Camera index such as 0/1/2, or a video file path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda:0, or 0.")
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Run overlapping 640x640 tiled inference and merge duplicate boxes.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=160,
        help="Overlap between neighboring tiles in pixels.",
    )
    parser.add_argument(
        "--tile-iou",
        type=float,
        default=0.45,
        help="IoU threshold used to merge duplicate tile detections.",
    )
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dshow", "msmf", "any"],
        help="Camera backend. On Windows, auto tries dshow, then msmf, then any.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan available camera indexes and exit.",
    )
    parser.add_argument("--scan-limit", type=int, default=DEFAULT_SCAN_LIMIT, help="Camera indexes to scan.")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save detected frames to --debug-dir. Default is off for Raspberry Pi deployment.",
    )
    parser.add_argument("--debug-dir", default="outputs/video_debug", help="Directory for saved debug frames.")
    parser.add_argument("--save-every-sec", type=float, default=2.0, help="Minimum seconds between saved frames.")
    parser.add_argument(
        "--no-stability",
        action="store_true",
        help="Show raw YOLO detections only, without frame-to-frame stabilization.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Draw raw YOLO detections together with stable tracks.",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=60.0,
        help="Maximum pixel distance for matching a detection to an existing track.",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=5,
        help="Frames a track remains visible after YOLO temporarily misses it.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="Smoothing factor in (0, 1]. Higher values follow raw detections faster.",
    )
    args = parser.parse_args()

    if args.list_cameras:
        return list_cameras(args.scan_limit, args.backend)

    source = parse_video_source(args.source)
    cap, used_backend = open_capture(source, args.backend, args.width, args.height)

    if not cap.isOpened():
        print(f"Cannot open video source: {args.source}")
        if isinstance(source, int):
            print("Try listing available cameras:")
            print(r"  .\.venv\Scripts\python.exe scripts\test_yolo_video.py --list-cameras")
            print("Then run with the detected index, for example:")
            print(r"  .\.venv\Scripts\python.exe scripts\test_yolo_video.py --source 0 --backend dshow")
        return 1

    if args.tiled:
        detector = TiledTennisBallDetector(
            model_path=args.model,
            conf=args.conf,
            overlap=args.tile_overlap,
            iou_threshold=args.tile_iou,
            device=args.device,
        )
        print("Inference mode: tiled 640x640")
    else:
        detector = TennisBallDetector(model_path=args.model, conf=args.conf, device=args.device)
        print("Inference mode: full image")
    print(f"Inference device: {detector.device}")
    tracker = None
    if not args.no_stability:
        tracker = StableTennisBallTracker(
            max_match_distance=args.max_match_distance,
            max_missed=args.max_missed,
            smooth_alpha=args.smooth_alpha,
        )
    debug_dir = Path(args.debug_dir)
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    prev_time = time.perf_counter()
    last_save_time = 0.0
    frame_index = 0

    print(f"Opened source {args.source} with backend {used_backend}. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received, stopping.")
            break

        detections = detector.detect(frame)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if tracker is None:
            display = draw_detections(frame, detections)
            displayed_count = len(detections)
        else:
            stable_tracks = tracker.update(detections)
            display = draw_stable_tracks(frame, stable_tracks)
            if args.show_raw:
                display = draw_detections(display, detections, color=(255, 0, 0))
            displayed_count = sum(1 for track in stable_tracks if track["valid"])
        cv2.putText(
            display,
            f"FPS {fps:.1f}  observed {displayed_count}  raw {len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.save_debug and detections and now - last_save_time >= args.save_every_sec:
            out_path = debug_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), display)
            last_save_time = now

        cv2.imshow("tennis_ball_detector", display)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

