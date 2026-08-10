from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tennis ball YOLO detector.")
    parser.add_argument("--model", default="models/tennis_ball_best.pt", help="Pretrained model path.")
    parser.add_argument("--data", default="tennis_ball_dataset/tennis_ball.yaml", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--project", default="runs", help="Results root directory.")
    parser.add_argument("--name", default="tennis_ball_ft_v1", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in last N epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable pretrained weights and train from scratch.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        pretrained=args.pretrained,
    )

    print(results)
    print(f"Best weights should be under: {args.project}/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
