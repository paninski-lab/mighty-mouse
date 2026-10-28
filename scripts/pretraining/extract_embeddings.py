"""Extract one CLS token per frame, for every video in the pretraining videos paths file.

Outputs: <pretraining_embeddings_dir>/<model_type>/<dataset>/<video_stem>.npy
(float16, shape (n_frames, hidden_size)). See skills/pretraining/extract-embeddings.

Usage:
    python scripts/pretraining/extract_embeddings.py [--dataset ibl] [--overwrite]
"""

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.pretraining.embed import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RESOLUTION,
    MODEL_TYPES,
    embed_video,
    load_cls_embedder,
    output_path,
)
from mighty_mouse.pretraining.video_list import list_videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model_type", default="dinov3_vits16", choices=list(MODEL_TYPES))
    parser.add_argument("--dataset", default=None, help="only process this dataset")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute existing outputs (default: skip)"
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help="square input side length in pixels; multiple of 16",
    )
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = parser.parse_args()

    paths = load_paths()
    videos_paths = yaml.safe_load(Path(paths["pretraining_videos_paths_file"]).read_text())
    embeddings_dir = Path(paths["pretraining_embeddings_dir"])

    videos, null_path = list_videos(videos_paths, args.dataset)
    print(f"{len(videos)} videos with paths; {len(null_path)} skipped (null path)")

    todo = []
    n_existing = 0
    for v in videos:
        if not args.overwrite and output_path(
            embeddings_dir, args.model_type, v.dataset, v.stem
        ).exists():
            n_existing += 1
        else:
            todo.append(v)
    print(f"{n_existing} already done (use --overwrite to redo); {len(todo)} to process")
    if not todo:
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cls_embedder(args.model_type, device)

    failed = []
    for i, v in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {v.dataset}/{v.stem}")
        try:
            emb = embed_video(
                model, v.path, args.resolution, args.batch_size, n_frames=v.n_frames
            )
        except Exception:
            traceback.print_exc()
            failed.append(f"{v.dataset}/{v.stem}")
            continue
        if v.n_frames is not None and len(emb) != v.n_frames:
            print(f"  WARNING: decoded {len(emb)} frames, videos paths file says {v.n_frames}")
        out = output_path(embeddings_dir, args.model_type, v.dataset, v.stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        # write atomically so a killed run never leaves a truncated file that the
        # skip-existing check would trust
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, emb)
        os.replace(tmp, out)

    print(f"done: {len(todo) - len(failed)} written, {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name}")


if __name__ == "__main__":
    main()
