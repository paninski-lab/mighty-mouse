"""Stage 3 of pretraining frame selection: extract the selected frames as images.

For each method (e.g. A1, A2, B) reads <pretrain_dir>/pretrain_datasets/<size>_method<method>/
selection.csv (columns dataset, video_key, frame_index), decodes those frames from their
videos (each video once, even if several methods chose frames from it), drops junk frames
(blank/flat or blown out), downscales so the short side is at most --min_side, and saves
<dataset>/<video_key>/img<frame>.png inside the method's directory. Dropped frames are listed
in dropped_frames.csv there. Existing images are skipped, so reruns resume. Exits with an error
if the junk share is above the limits (checked after everything is written).

Usage:
    python scripts/pretrain/extract_frames.py --methods A1 A2 B [--size 100k] [--workers 8]
"""

import argparse
import sys
import time
import traceback
from multiprocessing import Pool

import cv2
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.pretrain.frames import (
    MAX_JUNK_OVERALL,
    MAX_JUNK_PER_DATASET,
    MIN_SIDE,
    frame_path,
    is_junk,
    junk_counts,
    junk_stats,
    junk_violations,
    plan_extraction,
    read_frames,
    resize_min_side,
    save_frame,
)
from mighty_mouse.pretrain.layout import pretrain_dir, subset_dir, videos_paths_file
from mighty_mouse.pretrain.video_list import list_videos

SELECTION_FILE = "selection.csv"
DROPPED_FILE = "dropped_frames.csv"
DROPPED_COLUMNS = ["dataset", "video_key", "frame_index", "gray_std", "saturated"]


def process_video(task):
    """Decode one video's wanted frames, save the good ones for each method that chose them.

    Returns `(dataset, key, records, error)`; records are `(method, frame_index, gray_std,
    saturated, junk)` rows.
    """
    dataset, key, path, frames, out_dirs, min_side = task
    cv2.setNumThreads(1)
    records = []
    try:
        for idx, frame in read_frames(path, frames):
            frame = resize_min_side(frame, min_side)
            std, saturated = junk_stats(frame)
            junk = is_junk(std, saturated)
            for method in frames[idx]:
                if not junk:
                    save_frame(frame_path(out_dirs[method], dataset, key, idx), frame)
                records.append((method, idx, std, saturated, junk))
    except Exception:
        return dataset, key, records, traceback.format_exc()
    return dataset, key, records, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--methods", nargs="+", required=True, help="e.g. A1 A2 B")
    parser.add_argument("--size", default="100k", help="subset name prefix, <size>_method<m>")
    parser.add_argument(
        "--min_side", type=int, default=MIN_SIDE, help="short side of saved frames"
    )
    parser.add_argument("--workers", type=int, default=8, help="videos decoded in parallel")
    parser.add_argument("--max_junk", type=float, default=MAX_JUNK_OVERALL)
    parser.add_argument("--max_junk_per_dataset", type=float, default=MAX_JUNK_PER_DATASET)
    args = parser.parse_args()

    root = pretrain_dir(load_paths())
    out_dirs = {m: subset_dir(root, args.size, m) for m in args.methods}
    selections, dropped, done = {}, {}, {}
    for m, d in out_dirs.items():
        selections[m] = pd.read_csv(d / SELECTION_FILE)
        prev = d / DROPPED_FILE
        dropped[m] = pd.read_csv(prev) if prev.exists() else pd.DataFrame(columns=DROPPED_COLUMNS)
        done[m] = set(zip(dropped[m]["dataset"], dropped[m]["video_key"],
                          dropped[m]["frame_index"].astype(int), strict=True))
        for dataset, key, idx in zip(selections[m]["dataset"], selections[m]["video_key"],
                                     selections[m]["frame_index"], strict=True):
            if frame_path(d, dataset, key, int(idx)).exists():
                done[m].add((dataset, key, int(idx)))

    plan = plan_extraction(selections, done)
    videos_paths = yaml.safe_load(videos_paths_file(root).read_text())
    videos, _ = list_videos(videos_paths)
    by_key = {(v.dataset, v.key): v for v in videos}
    tasks = []
    for (dataset, key), frames in plan.items():
        v = by_key.get((dataset, key))
        if v is None:
            sys.exit(f"selected video {dataset}/{key} has no path in the videos paths file")
        if v.split == "test":
            sys.exit(f"selected video {dataset}/{key} is a test video")
        tasks.append((dataset, key, v.path, frames, out_dirs, args.min_side))
    n_frames = sum(len(t[3]) for t in tasks)
    print(f"{len(tasks)} videos, {n_frames} frames to extract for methods {args.methods}")

    new_rows, failed = {m: [] for m in args.methods}, []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (dataset, key, records, error) in enumerate(
            pool.imap_unordered(process_video, tasks), 1
        ):
            if error:
                failed.append(f"{dataset}/{key}")
                print(f"FAILED {dataset}/{key}:\n{error}")
            for method, idx, std, saturated, junk in records:
                if junk:
                    new_rows[method].append((dataset, key, idx, std, saturated))
            if i % 20 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)

    exit_messages = []
    for m, d in out_dirs.items():
        new = pd.DataFrame(new_rows[m], columns=DROPPED_COLUMNS)
        parts = [t for t in (dropped[m], new) if len(t)]
        all_dropped = pd.concat(parts, ignore_index=True) if parts else new
        all_dropped.to_csv(d / DROPPED_FILE, index=False)
        counts = junk_counts(selections[m], all_dropped)
        print(f"\nmethod {m}: junk frames dropped")
        print(counts.to_string(float_format=lambda x: f"{x:.4f}"))
        exit_messages += [
            f"method {m}, {msg}"
            for msg in junk_violations(counts, args.max_junk, args.max_junk_per_dataset)
        ]
    if failed:
        exit_messages.append(f"{len(failed)} videos failed: {failed}")
    if exit_messages:
        sys.exit("\n".join(exit_messages))


if __name__ == "__main__":
    main()
