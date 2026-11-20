"""Stage 2 of pretraining frame selection: choose the frames of one pretraining subset.

Splits the subset size over datasets in proportion to (dataset frames)^alpha, capped at what
stage 1 kept (`budgets.py`), then picks each dataset's share from its stage-1 pool with
--method hier (balanced two-level k-means) or fps (farthest-point). Test videos are never used.
Output: <pretrain_dir>/pretrain_datasets/<size>_method<method>/selection.csv (dataset,
video_key, frame_index), which extract_frames.py turns into images.

Usage:
    python scripts/pretrain/select_frames.py --method hier --size 100k [--alpha 0.3]
"""

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.pretrain.budgets import DEFAULT_ALPHA, dataset_budgets
from mighty_mouse.pretrain.embed import MODEL_TYPES
from mighty_mouse.pretrain.layout import (
    dedup_dir,
    embeddings_dir,
    parse_size,
    pretrain_dir,
    subset_dir,
    videos_paths_file,
)
from mighty_mouse.pretrain.pool import load_pool, pool_counts
from mighty_mouse.pretrain.selection import METHODS, select_pool
from mighty_mouse.pretrain.video_list import list_videos

SELECTION_FILE = "selection.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--size", required=True, help="frames in the subset, e.g. 100k or 1m")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="budget exponent")
    parser.add_argument("--model_type", default="dinov3_vits16", choices=list(MODEL_TYPES))
    parser.add_argument(
        "--splits", nargs="+", default=["train", "unlabeled"],
        help="splits to draw from (test videos are never used for pretraining)",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing selection")
    args = parser.parse_args()

    import torch

    root = pretrain_dir(load_paths())
    out_dir = subset_dir(root, args.size, args.method)
    out_file = out_dir / SELECTION_FILE
    if out_file.exists() and not args.overwrite:
        raise SystemExit(f"{out_file} exists; use --overwrite to replace it")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    videos_paths = yaml.safe_load(videos_paths_file(root).read_text())
    videos, _ = list_videos(videos_paths, splits=tuple(args.splits))
    keys: dict[str, list[str]] = {}
    for v in videos:
        keys.setdefault(v.dataset, []).append(v.key)

    raw, pool, missing = {}, {}, {}
    for dataset, dataset_keys in keys.items():
        r, p, m = pool_counts(dedup_dir(root), args.model_type, dataset, dataset_keys)
        if p:
            raw[dataset], pool[dataset] = r, p
        if m:
            missing[dataset] = len(m)
    if missing:
        print(f"WARNING: videos without stage-1 output, left out: {missing}")
    budgets = dataset_budgets(raw, pool, parse_size(args.size), args.alpha)
    print(f"{'dataset':20s}{'frames':>12s}{'pool':>10s}{'budget':>10s}")
    for d in sorted(budgets):
        print(f"{d:20s}{raw[d]:12d}{pool[d]:10d}{budgets[d]:10d}")

    parts = []
    for dataset in sorted(budgets):
        t0 = time.time()
        p = load_pool(
            dedup_dir(root), embeddings_dir(root), args.model_type, dataset, keys[dataset]
        )
        x = torch.from_numpy(p.embeddings).to(device)
        if torch.device(device).type == "cpu":
            x = x.float()  # fp16 matmuls are slow on cpu
        chosen = select_pool(x, budgets[dataset], args.method, args.seed)
        parts.append(pd.DataFrame({
            "dataset": dataset,
            "video_key": np.array(p.video_keys)[p.video[chosen]],
            "frame_index": p.frame_idx[chosen],
        }))
        print(f"{dataset}: {len(chosen)}/{len(p.frame_idx)} frames, {time.time() - t0:.0f}s")
        del x
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

    selection = pd.concat(parts, ignore_index=True)
    selection = selection.sort_values(["dataset", "video_key", "frame_index"], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out_file, index=False)
    print(f"wrote {len(selection)} frames to {out_file}")


if __name__ == "__main__":
    main()
