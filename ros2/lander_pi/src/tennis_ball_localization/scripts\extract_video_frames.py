from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def iter_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def frame_output_name(video_path: Path, frame_index: int) -> str:
    return f"{video_path.stem}_frame_{frame_index:06d}.jpg"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int,
    start_frame: int,
    max_frames: int,
    quality: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skip unreadable video: {video_path}")
        return 0

    source_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(
        f"{video_path.name}: {source_width}x{source_height} "
        f"@{source_fps:.2f}fps, {total_frames} frames"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_index = start_frame
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_index - start_frame) % every_n_frames == 0:
                output_path = output_dir / frame_output_name(video_path, frame_index)
                if not cv2.imwrite(str(output_path), frame, params):
                    raise RuntimeError(f"Failed to write {output_path}")
                saved += 1
                if max_frames > 0 and saved >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()

    print(f"Saved {saved} frames -> {output_dir}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract still frames from 1280x800 videos for YOLO annotation."
    )
    parser.add_argument(
        "--source",
        default="inputs",
        help="Input video file or directory. Default: inputs",
    )
    parser.add_argument(
        "--out",
        default="dataset_source/images",
        help="Output image directory. Default: dataset_source/images",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=30,
        help="Save one frame every N frames. For 30fps video, 30 means one image per second.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to consider.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum saved frames per video. 0 means unlimited.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, 0-100.")
    args = parser.parse_args()

    if args.every <= 0:
        raise ValueError("--every must be positive.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be zero or greater.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be zero or greater.")
    if not 0 <= args.quality <= 100:
        raise ValueError("--quality must be between 0 and 100.")

    source = Path(args.source)
    output_dir = Path(args.out)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    videos = iter_videos(source)
    if not videos:
        print(f"No videos found in {source}")
        return 1

    total_saved = 0
    for video_path in videos:
        video_output_dir = output_dir if source.is_file() else output_dir / video_path.stem
        total_saved += extract_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            every_n_frames=args.every,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            quality=args.quality,
        )

    print(f"Total saved frames: {total_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
