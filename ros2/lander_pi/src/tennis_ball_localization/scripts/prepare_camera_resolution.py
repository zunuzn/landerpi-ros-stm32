from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def center_crop_to_aspect(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Crop the image around its center without changing object proportions."""
    image_height, image_width = image.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = image_width / image_height

    if source_aspect > target_aspect:
        crop_width = int(round(image_height * target_aspect))
        x0 = (image_width - crop_width) // 2
        return image[:, x0 : x0 + crop_width]

    crop_height = int(round(image_width / target_aspect))
    y0 = (image_height - crop_height) // 2
    return image[y0 : y0 + crop_height, :]


def letterbox_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Keep all source pixels and pad unused areas with black."""
    image_height, image_width = image.shape[:2]
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    output = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    x0 = (target_width - resized_width) // 2
    y0 = (target_height - resized_height) // 2
    output[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return output


def resize_image(image: np.ndarray, target_width: int, target_height: int, mode: str) -> np.ndarray:
    if mode == "crop":
        image = center_crop_to_aspect(image, target_width, target_height)
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if mode == "letterbox":
        return letterbox_to_size(image, target_width, target_height)
    if mode == "stretch":
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    raise ValueError(f"Unsupported resize mode: {mode}")


def video_output_path(source: Path, output: Path, width: int, height: int) -> Path:
    if output.suffix.lower() in VIDEO_EXTENSIONS:
        return output
    return output / f"{source.stem}_{width}x{height}.mp4"


def convert_video(
    source: Path,
    output: Path,
    target_width: int,
    target_height: int,
    mode: str,
    codec: str,
    fps_override: float,
) -> Path:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    source_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_fps = fps_override if fps_override > 0 else source_fps
    if output_fps <= 0:
        output_fps = 30.0

    if len(codec) != 4:
        raise ValueError("--video-codec must contain exactly four characters, for example mp4v.")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*codec),
        output_fps,
        (target_width, target_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    print(
        f"{source.name}: {source_width}x{source_height} @{source_fps:.2f}fps -> "
        f"{target_width}x{target_height} @{output_fps:.2f}fps ({mode})"
    )

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            writer.write(resize_image(frame, target_width, target_height, mode))
            frame_index += 1
            if frame_index % 100 == 0:
                total = f"/{frame_count}" if frame_count > 0 else ""
                print(f"Processed {frame_index}{total} frames")
    finally:
        cap.release()
        writer.release()

    print(f"Saved video: {output} ({frame_index} frames, audio is not copied)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert phone photos or a video to a camera-like fixed RGB resolution."
    )
    parser.add_argument(
        "--source",
        default="photo",
        help="Input image file/directory or a video file. Default: photo",
    )
    parser.add_argument(
        "--out",
        default="inputs",
        help="Output directory. Default: inputs",
    )
    parser.add_argument("--width", type=int, default=1280, help="Target image width.")
    parser.add_argument("--height", type=int, default=800, help="Target image height.")
    parser.add_argument(
        "--mode",
        choices=("crop", "letterbox", "stretch"),
        default="crop",
        help="crop preserves shape and fills the frame; letterbox preserves all pixels; stretch distorts objects.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, 0-100.")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output video FPS. 0 keeps the source FPS.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="FourCC used for output video. Default: mp4v.",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if not 0 <= args.quality <= 100:
        raise ValueError("--quality must be between 0 and 100.")
    if args.fps < 0:
        raise ValueError("--fps must be zero or greater.")

    source = Path(args.source)
    output_root = Path(args.out)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        convert_video(
            source=source,
            output=video_output_path(source, output_root, args.width, args.height),
            target_width=args.width,
            target_height=args.height,
            mode=args.mode,
            codec=args.video_codec,
            fps_override=args.fps,
        )
        return 0

    images = iter_images(source)
    if not images:
        print(f"No images found in {source}")
        return 1

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        converted = resize_image(image, args.width, args.height, args.mode)
        relative_path = image_path.name if source.is_file() else image_path.relative_to(source)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if output_path.suffix.lower() in {".jpg", ".jpeg"} else []
        if not cv2.imwrite(str(output_path), converted, params):
            raise RuntimeError(f"Failed to write {output_path}")

        source_height, source_width = image.shape[:2]
        print(
            f"{image_path.name}: {source_width}x{source_height} -> "
            f"{args.width}x{args.height} ({args.mode}) -> {output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
