from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float


def iter_images(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length < tile_size:
        raise ValueError(f"Image dimension {length} is smaller than tile size {tile_size}")
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if values[-1] != last_start:
        values.append(last_start)
    return values


def tile_origins(image_width: int, image_height: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in starts(image_height, tile_size, overlap)
        for x in starts(image_width, tile_size, overlap)
    ]


def read_labels(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []

    boxes: list[YoloBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line}")

        class_id = int(parts[0])
        cx, cy, width, height = [float(value) for value in parts[1:]]
        boxes.append(YoloBox(class_id, cx, cy, width, height))

    return boxes


def yolo_to_xyxy(box: YoloBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    cx = box.cx * image_width
    cy = box.cy * image_height
    width = box.width * image_width
    height = box.height * image_height
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


def clip_box_to_tile(
    box_xyxy: tuple[float, float, float, float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box_xyxy
    original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if original_area <= 0.0:
        return None

    clipped_x1 = max(x1, float(tile_x))
    clipped_y1 = max(y1, float(tile_y))
    clipped_x2 = min(x2, float(tile_x + tile_size))
    clipped_y2 = min(y2, float(tile_y + tile_size))
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    if clipped_area <= 0.0 or clipped_area / original_area < min_visibility:
        return None

    return (
        clipped_x1 - tile_x,
        clipped_y1 - tile_y,
        clipped_x2 - tile_x,
        clipped_y2 - tile_y,
    )


def xyxy_to_yolo(
    class_id: int,
    box_xyxy: tuple[float, float, float, float],
    tile_size: int,
) -> YoloBox:
    x1, y1, x2, y2 = box_xyxy
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    return YoloBox(
        class_id=class_id,
        cx=cx / tile_size,
        cy=cy / tile_size,
        width=width / tile_size,
        height=height / tile_size,
    )


def write_labels(label_path: Path, boxes: list[YoloBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_name(index: int, val_ratio: float) -> str:
    if val_ratio <= 0.0:
        return "train"
    if val_ratio >= 1.0:
        return "val"
    return "val" if random.random() < val_ratio else "train"


def process_image(
    image_path: Path,
    labels_dir: Path,
    output_dir: Path,
    split: str,
    tile_size: int,
    overlap: int,
    min_visibility: float,
    include_empty_tiles: bool,
    empty_tile_ratio: float,
) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Skip unreadable image: {image_path}")
        return 0, 0

    image_height, image_width = image.shape[:2]
    labels = read_labels(labels_dir / f"{image_path.stem}.txt")
    full_boxes = [yolo_to_xyxy(label, image_width, image_height) for label in labels]

    positive_tiles: list[tuple[int, int, str, list[YoloBox]]] = []
    empty_tiles: list[tuple[int, int, str]] = []
    for tile_index, (tile_x, tile_y) in enumerate(tile_origins(image_width, image_height, tile_size, overlap)):
        tile_boxes: list[YoloBox] = []
        for label, full_box in zip(labels, full_boxes):
            clipped = clip_box_to_tile(
                full_box,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_size=tile_size,
                min_visibility=min_visibility,
            )
            if clipped is not None:
                tile_boxes.append(xyxy_to_yolo(label.class_id, clipped, tile_size))

        tile_stem = f"{image_path.stem}_tile_{tile_index:02d}_x{tile_x}_y{tile_y}"
        if tile_boxes:
            positive_tiles.append((tile_x, tile_y, tile_stem, tile_boxes))
        else:
            empty_tiles.append((tile_x, tile_y, tile_stem))

    if include_empty_tiles and empty_tile_ratio > 0.0:
        max_empty_tiles = int(len(positive_tiles) * empty_tile_ratio)
        if len(empty_tiles) > max_empty_tiles:
            empty_tiles = random.sample(empty_tiles, max_empty_tiles)
    else:
        empty_tiles = []

    written_images = 0
    written_boxes = 0

    def save_tile(tile_x: int, tile_y: int, tile_stem: str, tile_boxes: list[YoloBox]) -> None:
        nonlocal written_images, written_boxes
        tile = image[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
        out_image_path = output_dir / "images" / split / f"{tile_stem}.jpg"
        out_label_path = output_dir / "labels" / split / f"{tile_stem}.txt"
        out_image_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(out_image_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {out_image_path}")

        write_labels(out_label_path, tile_boxes)
        written_images += 1
        written_boxes += len(tile_boxes)

    for tile_x, tile_y, tile_stem, tile_boxes in positive_tiles:
        save_tile(tile_x, tile_y, tile_stem, tile_boxes)

    for tile_x, tile_y, tile_stem in empty_tiles:
        save_tile(tile_x, tile_y, tile_stem, [])

    return written_images, written_boxes


def write_dataset_yaml(output_dir: Path, class_name: str) -> None:
    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (output_dir / "tennis_ball.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a tiled 640x640 YOLO dataset from annotated 1280x800 images."
    )
    parser.add_argument("--images", default="dataset_source/images", help="Annotated source images directory.")
    parser.add_argument("--labels", default="dataset_source/labels", help="YOLO labels directory matching source images.")
    parser.add_argument("--out", default="tennis_ball_dataset", help="Output YOLO dataset directory.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=160, help="Tile overlap in pixels.")
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.35,
        help="Minimum fraction of a label box that must remain inside a tile.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Random validation split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/val split.")
    parser.add_argument(
        "--include-empty-tiles",
        action="store_true",
        help="Keep some tiles without objects as negative samples.",
    )
    parser.add_argument(
        "--empty-tile-ratio",
        type=float,
        default=2.0,
        help="Maximum empty tiles per positive tile. Default: 2.0 for roughly 1:2 positive-to-empty.",
    )
    parser.add_argument("--class-name", default="tennis_ball", help="Class name written to dataset YAML.")
    args = parser.parse_args()

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile-size.")
    if not 0.0 <= args.min_visibility <= 1.0:
        raise ValueError("--min-visibility must be between 0 and 1.")
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.empty_tile_ratio < 0.0:
        raise ValueError("--empty-tile-ratio must be zero or greater.")

    random.seed(args.seed)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.out)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = iter_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1

    total_tiles = 0
    total_boxes = 0
    for image_index, image_path in enumerate(images):
        split = split_name(image_index, args.val_ratio)
        written_images, written_boxes = process_image(
            image_path=image_path,
            labels_dir=labels_dir,
            output_dir=output_dir,
            split=split,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_visibility=args.min_visibility,
            include_empty_tiles=args.include_empty_tiles,
            empty_tile_ratio=args.empty_tile_ratio,
        )
        total_tiles += written_images
        total_boxes += written_boxes
        print(f"{image_path.name}: {written_images} tiles, {written_boxes} boxes -> {split}")

    write_dataset_yaml(output_dir, args.class_name)
    print(f"Dataset saved: {output_dir}")
    print(f"Total tiles: {total_tiles}, total boxes: {total_boxes}")
    print(f"YAML: {output_dir / 'tennis_ball.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
