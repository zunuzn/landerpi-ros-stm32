from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_ball_detector import TennisBallDetector, draw_detections
from src.tiled_tennis_ball_detector import TiledTennisBallDetector


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a YOLO tennis ball detector on images.")
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Path to .pt/.onnx model.")
    parser.add_argument("--source", default="inputs", help="Image file or directory.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
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
    args = parser.parse_args()

    model_path = Path(args.model)
    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

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
    images = iter_images(source)
    if not images:
        print(f"No images found in {source}")
        return 1

    csv_path = out_dir / "detections.csv"
    rows = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        detections = detector.detect(image)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            u, v = detection["center"]
            rows.append(
                {
                    "image": str(image_path),
                    "class_id": detection["class_id"],
                    "confidence": f"{detection['confidence']:.4f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "center_x": f"{u:.2f}",
                    "center_y": f"{v:.2f}",
                }
            )

        out_path = out_dir / f"{image_path.stem}_detected{image_path.suffix}"
        cv2.imwrite(str(out_path), draw_detections(image, detections))
        print(f"{image_path.name}: {len(detections)} detections -> {out_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved detection table: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

