---
name: extract-embeddings
description: Use when the user wants to extract per-frame CLS-token embeddings (e.g. DINOv3 ViT-S) from the pretraining video collection, for all datasets or one — outputs one .npy per video under pretraining_embeddings_dir.
---

# Extracting per-frame CLS embeddings from pretraining videos

Part of the masked-autoencoder backbone pretraining data pipeline. Runs a frozen ViT over
every frame of every video listed in the pretraining videos paths file and saves one CLS
token per frame.

## Two different `paths.yaml` files

- **Repo `paths.yaml`** (machine-specific, gitignored) — gains two keys for this pipeline:
  - `pretraining_videos_paths_file`: path to the videos paths file below
  - `pretraining_embeddings_dir`: root for outputs
- **Videos paths file** (`poseinterface/videos/paths.yaml`, built by `videos/code/`) —
  nested `dataset -> split (train/test/unlabeled) -> video key -> {path, dims, frames}`.
  Entries with `path: null` aren't filled in yet and are skipped (counted in the output).

## Run

```bash
python scripts/pretraining/extract_embeddings.py                    # every dataset
python scripts/pretraining/extract_embeddings.py --dataset ibl      # one dataset
python scripts/pretraining/extract_embeddings.py --overwrite        # recompute existing
```

Flags: `--model_type` (default `dinov3_vits16`), `--batch_size`, `--resolution` (default
224, multiple of 16), `--device`. Logic lives in `mighty_mouse/pretraining/`.

## Output

`<pretraining_embeddings_dir>/<model_type>/<dataset>/<video_key>.npy` — float16, shape
`(n_frames, hidden_size)`, frame-ordered. `video_key` is the video's key in the videos paths
file, not its file stem (e.g. every ibl video is named `_iblrig_leftCamera.raw.mp4`).
The model runs under bf16 autocast on CUDA; the CLS tokens are cast to float16 for storage.

## Gotchas

- **Resume-safe.** Existing outputs are skipped unless `--overwrite`; files are written
  atomically (`.npy.tmp` then rename), so an interrupted run never leaves a truncated
  file that would be trusted on the next run.
- **Resolution isn't in the output path.** Embeddings from different `--resolution`s
  would collide under the same `model_type` directory; use `--overwrite` deliberately.
- Frames are stretched to a square (no crop). A decoded-vs-expected frame count mismatch
  against the videos paths file prints a warning but still writes the file.
- Videos that fail to decode are logged and skipped; the run continues and lists them at the end.
- A key appearing in more than one split of a dataset is an error (they'd share an output file).
- Needs `torch` and `transformers` (imported lazily; not in `pyproject.toml`/CI).
