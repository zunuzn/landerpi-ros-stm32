from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_ball_detector import TennisBallDetector, draw_detections
from src.tiled_tennis_ball_detector import TiledTennisBallDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def detection_to_yolo_line(detection: dict, image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = detection["bbox"]
    center_x, center_y = detection["center"]
    width = x2 - x1
    height = y2 - y1
    return (
        f"{int(detection['class_id'])} "
        f"{center_x / image_width:.6f} "
        f"{center_y / image_height:.6f} "
        f"{width / image_width:.6f} "
        f"{height / image_height:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-label tennis ball images with the current YOLO model."
    )
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Path to .pt/.onnx model.")
    parser.add_argument("--images", default="dataset_source/images", help="Input image file or directory.")
    parser.add_argument("--labels", default="dataset_source/labels", help="Output YOLO label directory.")
    parser.add_argument("--preview-dir", default=None, help="Optional directory for annotated preview images.")
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files. Default is to skip if the label already exists.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty label files for images with no detections.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional extra confidence filter applied after inference.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images source not found: {images_dir}")

    if args.tiled:
        detector = TiledTennisBallDetector(
            model_path=model_path,
            conf=args.conf,
            overlap=args.tile_overlap,
            iou_threshold=args.tile_iou,
            device=args.device,
        )
        print("Inference mode: tiled 640x640")
    else:
        detector = TennisBallDetector(model_path=model_path, conf=args.conf, device=args.device)
        print("Inference mode: full image")

    print(f"Inference device: {detector.device}")
    images = iter_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1

    labels_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_written = 0
    total_detections = 0

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        detections = detector.detect(image)
        if args.min_confidence is not None:
            detections = [det for det in detections if det["confidence"] >= args.min_confidence]

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            print(f"Skip existing label: {label_path}")
            continue

        if detections or args.write_empty:
            lines = [detection_to_yolo_line(det, image.shape[1], image.shape[0]) for det in detections]
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            total_written += 1

        if preview_dir is not None:
            preview_path = preview_dir / f"{image_path.stem}_preview{image_path.suffix}"
            cv2.imwrite(str(preview_path), draw_detections(image, detections))

        total_images += 1
        total_detections += len(detections)
        print(f"{image_path.name}: {len(detections)} detections -> {label_path}")

    print(f"Processed images: {total_images}")
    print(f"Written labels: {total_written}")
    print(f"Total detections: {total_detections}")
    if preview_dir is not None:
        print(f"Preview images: {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
