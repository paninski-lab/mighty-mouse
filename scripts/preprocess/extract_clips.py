#!/usr/bin/env python3
"""
Extract a fixed-length clip from every video in a directory.

For each source video, picks the `--clip-length`-second window with the most movement
(measured from raw pixel differences, or from pose predictions if `--preds-dir` is
given), or, with `--from-start`, just the first `--clip-length` seconds starting at
`--skip-start`. Saves as an h264/yuv420p mp4 in `--out-dir`, regardless of the source
codec/container. See mighty_mouse/preprocess/videos.py:make_video_snippet for the core logic.

Usage:
    python scripts/preprocess/extract_clips.py \\
        --video-dir /media/mattw/poseinterface/_raw/_dlc/cazettes-side/videos-avi \\
        --out-dir /media/mattw/poseinterface/_raw/_dlc/cazettes-side/videos_test
"""

import argparse
from pathlib import Path

from mighty_mouse.preprocess.videos import make_video_snippet

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video-dir", type=Path, required=True, help="directory of source videos")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory to save clips to")
    parser.add_argument(
        "--preds-dir", type=Path, default=None,
        help="directory of pose-prediction csvs (matched to videos by stem); if omitted, "
             "movement is measured from raw pixel differences instead",
    )
    parser.add_argument("--clip-length", type=int, default=15, help="clip length in seconds")
    parser.add_argument(
        "--skip-start", type=float, default=60.0,
        help="ignore this many seconds at the start of each video -- when searching for "
             "the highest-motion window (e.g. to skip past camera setup/handling), or, "
             "with --from-start, as the point each clip itself starts from",
    )
    parser.add_argument(
        "--from-start", action="store_true",
        help="skip the motion-energy search and just take the --clip-length-second "
             "window starting at --skip-start",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="override the frame rate used for all time math and for reading each "
             "source video, for videos whose container reports the wrong frame rate "
             "(e.g. a camera's true capture rate rather than the rate it was saved "
             "at). Defaults to each video's own reported frame rate.",
    )
    parser.add_argument(
        "--likelihood-thresh", type=float, default=0.9,
        help="only used with --preds-dir: likelihood threshold for keypoints counted "
             "toward the movement measure",
    )
    parser.add_argument("--crf", type=int, default=23, help="h264 constant rate factor")
    parser.add_argument("--preset", type=str, default="medium", help="ffmpeg x264 preset")
    args = parser.parse_args()

    video_files = sorted(
        p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        raise FileNotFoundError(f"No videos found in {args.video_dir}")

    for video_file in video_files:
        preds_file = None
        if args.preds_dir is not None:
            candidate = args.preds_dir / f"{video_file.stem}.csv"
            if candidate.exists():
                preds_file = candidate
            else:
                print(f"  no predictions found for {video_file.name}, using pixel motion energy")

        dst, clip_start_idx, clip_start_sec = make_video_snippet(
            video_file=video_file,
            out_dir=args.out_dir,
            preds_file=preds_file,
            clip_length=args.clip_length,
            likelihood_thresh=args.likelihood_thresh,
            skip_start=args.skip_start,
            from_start=args.from_start,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
        )
        print(
            f"{video_file.name} -> {dst.name} "
            f"(start: frame {clip_start_idx}, {clip_start_sec:.1f}s)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
