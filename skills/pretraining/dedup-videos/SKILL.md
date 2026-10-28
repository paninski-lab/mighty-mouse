---
name: dedup-videos
description: Use when the user wants to run stage 1 of pretraining frame selection — per-video dedup of the CLS embeddings (k-means, one frame kept per cluster), for all videos or one dataset, including after ingesting new videos. Outputs one .npz per video under <pretraining>/dedup.
---

# Stage 1: per-video dedup of pretraining frames

Part of the masked-autoencoder backbone pretraining data pipeline; needs the embeddings from
[`extract-embeddings`](../extract-embeddings/SKILL.md). The goal is ~1M diverse frames from the
whole video corpus; stage 1 only removes redundancy *within each video*, so it never depends on
any other video and can be rerun per new video. Stage 2 (dataset budgets, cross-video dedup) is
a separate step.

## What it does

For each train/unlabeled video (test videos are never used) it L2-normalizes the frame
embeddings, runs spherical k-means (cosine distance, k-means++ init) with `k = 15000`, and keeps
one random frame from each cluster. A video with at most `k` frames is kept whole (no k-means).
Rapid-motion stretches are not thinned beforehand, so every frame competes for a cluster.

## Run

```bash
python scripts/pretraining/dedup_videos.py                    # every dataset
python scripts/pretraining/dedup_videos.py --dataset ibl      # one dataset
python scripts/pretraining/dedup_videos.py --overwrite        # recompute existing
```

Flags: `--model_type` (default `dinov3_vits16`), `--splits` (default `train unlabeled`), `--k`,
`--device`. Videos without embeddings yet are skipped (counted in the output), so it can run
alongside extraction. Logic: `mighty_mouse/pretraining/{kmeans,dedup}.py`. Needs a GPU for large
videos (a ~1M-frame video takes ~20s and ~5 GiB; CPU works but is much slower).

## Output

`<pretraining>/dedup/<model_type>/<dataset>/<video_key>.npz`:

- `frame_idx` — int32, sorted indices of the kept frames (into the video's embeddings `.npy`)
- `cluster_size` — int32, how many frames each kept frame's cluster contained (its weight)
- scalars: `n_frames`, `k`, `seed`, `n_iter`, `n_empty_clusters`, and `coverage_mean/p95/max`
  (cosine distance from every frame to its nearest kept frame — for comparing selection recipes)

## Gotchas

- **Resume-safe, but not param-aware.** Existing outputs are skipped unless `--overwrite`
  (written atomically). If an existing file was made with a different `k` than requested, the
  run only prints a warning — pass `--overwrite` to redo it.
- The random choice within a cluster is seeded from the video key, so results don't depend on
  run order or on what else is processed.
- Embeddings are stored in float16; normalization happens in float32 chunks after loading.
- Needs `torch` (imported lazily; not in `pyproject.toml`/CI, so the k-means tests skip
  without it).
