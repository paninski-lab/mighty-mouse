"""Stage 1 of pretraining frame selection: per-video dedup of CLS embeddings.

For every train/unlabeled video with embeddings, keep at most k (default 15000) frames:
videos with at most k frames are kept whole, shorter ones use k-means with one random frame
per cluster, long ones (default: at least 100k frames) a two-level k-means with balanced
sampling. Output: <pretraining>/dedup/<model_type>/<dataset>/<video_key>.npz (frame_idx,
cluster_size, stats). See skills/pretraining/dedup-videos.

Usage:
    python scripts/pretraining/dedup_videos.py [--dataset ibl] [--overwrite]
"""

import argparse
import time
import traceback

import numpy as np
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.pretraining.dedup import (
    DEFAULT_K,
    METHODS,
    SKIP_FIRST_FRAMES,
    TWO_LEVEL_MIN_FRAMES,
    choose_method,
    load_selection,
    output_path,
    save_selection,
    seed_from_key,
    select_frames,
    stored_method,
)
from mighty_mouse.pretraining.embed import MODEL_TYPES
from mighty_mouse.pretraining.embed import output_path as embeddings_path
from mighty_mouse.pretraining.paths import (
    dedup_dir,
    embeddings_dir,
    pretraining_dir,
    videos_paths_file,
)
from mighty_mouse.pretraining.video_list import list_videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model_type", default="dinov3_vits16", choices=list(MODEL_TYPES))
    parser.add_argument("--dataset", default=None, help="only process this dataset")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "unlabeled"],
        help="splits to process (test videos are never used for pretraining)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="max frames kept per video")
    parser.add_argument(
        "--method", default="auto", choices=METHODS,
        help="auto: two_level for videos with at least --two_level_min_frames, else flat",
    )
    parser.add_argument(
        "--two_level_min_frames", type=int, default=TWO_LEVEL_MIN_FRAMES,
        help="with --method auto, the smallest video that uses two_level",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute existing outputs (default: skip)"
    )
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = parser.parse_args()

    root = pretraining_dir(load_paths())
    videos_paths = yaml.safe_load(videos_paths_file(root).read_text())
    videos, null_path = list_videos(videos_paths, args.dataset, tuple(args.splits))
    print(f"{len(videos)} videos in splits {args.splits}; {len(null_path)} skipped (null path)")

    todo, n_existing, n_missing, stale = [], 0, 0, {}
    for v in videos:
        out = output_path(dedup_dir(root), args.model_type, v.dataset, v.key)
        if not embeddings_path(embeddings_dir(root), args.model_type, v.dataset, v.key).exists():
            n_missing += 1
        elif out.exists() and not args.overwrite:
            n_existing += 1
            stats = load_selection(out)[2]
            skip = SKIP_FIRST_FRAMES.get(v.dataset, 0)
            expected = choose_method(
                stats["n_frames"] - skip, args.k, args.method, args.two_level_min_frames
            )
            if (stats["k"], stored_method(stats), stats.get("skip_first", 0)) != (
                args.k, expected, skip
            ):
                stale[v.dataset] = stale.get(v.dataset, 0) + 1
        else:
            todo.append(v)
    print(
        f"{n_missing} without embeddings yet; {n_existing} already done "
        f"(use --overwrite to redo); {len(todo)} to process"
    )
    if stale:
        print(
            f"WARNING: {sum(stale.values())} existing outputs were made with different "
            f"settings (k, method or skipped frames) and were left alone: {stale}; "
            "rerun those with --overwrite"
        )
    if not todo:
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    failed = []
    for i, v in enumerate(todo, 1):
        t0 = time.time()
        try:
            emb = np.load(embeddings_path(embeddings_dir(root), args.model_type, v.dataset, v.key))
            if v.n_frames is not None and len(emb) != v.n_frames:
                print(f"  WARNING: {len(emb)} embeddings, videos paths file says {v.n_frames}")
            frame_idx, cluster_size, stats = select_frames(
                emb, args.k, seed_from_key(v.key), device, args.method,
                args.two_level_min_frames, SKIP_FIRST_FRAMES.get(v.dataset, 0),
            )
        except Exception:
            traceback.print_exc()
            failed.append(f"{v.dataset}/{v.key}")
            continue
        save_selection(output_path(dedup_dir(root), args.model_type, v.dataset, v.key),
                       frame_idx, cluster_size, stats)
        print(
            f"[{i}/{len(todo)}] {v.dataset}/{v.key}: {stats['n_frames']} -> {len(frame_idx)} "
            f"frames ({stats['method']}), {stats['n_empty_clusters']} empty clusters, "
            f"{stats['n_iter']} iters, coverage p95 {stats['coverage_p95']:.4f}, "
            f"{time.time() - t0:.0f}s"
        )

    print(f"done: {len(todo) - len(failed)} written, {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name}")


if __name__ == "__main__":
    main()
