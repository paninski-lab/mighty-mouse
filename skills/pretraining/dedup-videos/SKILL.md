---
name: dedup-videos
description: Use when the user wants to run stage 1 of pretraining frame selection — per-video dedup of the CLS embeddings (at most 15000 frames kept per video, k-means / two-level k-means), for all videos or one dataset, including after ingesting new videos. Outputs one .npz per video under <pretraining>/dedup.
---

# Stage 1: per-video dedup of pretraining frames

Part of the masked-autoencoder backbone pretraining data pipeline; needs the embeddings from
[`extract-embeddings`](../extract-embeddings/SKILL.md). The goal is ~1M diverse frames from the
whole video corpus; stage 1 only removes redundancy *within each video*, so it never depends on
any other video and can be rerun per new video. Stage 2 (dataset budgets, cross-video dedup) is
a separate step.

## What it does

For each train/unlabeled video (test videos are never used) it L2-normalizes the frame
embeddings and keeps at most `k = 15000` frames. The method depends on the number of usable
frames `n`:

| `n` | method | how |
|---|---|---|
| `<= k` | `whole` | every frame is kept, no k-means |
| `k < n < 100k` | `flat` | spherical k-means (cosine, k-means++ init) with `k` clusters, one random frame per cluster |
| `>= 100k` | `two_level` | 3k fine clusters grouped into 1500 broad groups; the same number of fine clusters is drawn from every group (small groups give all they have), then one random frame per chosen cluster |

Two-level under-weights dense stretches (a long still pose) more than flat does, so rare poses
and fast motion keep more frames; in tests on 20 long videos it gave better tail coverage and
rare-state share at ~2.6x the run time and ~3% worse mean coverage. It was worse on 51-72k-frame
videos (cheese-3d), hence the size threshold (`--two_level_min_frames`; the 100k default is
provisional, untested between 72k and 356k frames). Rapid-motion stretches are not thinned
beforehand, so every frame competes for a cluster.

`SKIP_FIRST_FRAMES` (`mighty_mouse/pretraining/dedup.py`) lists leading frames to ignore per
dataset: the aind-vbn videos start with an encoder-info placeholder frame, which k-means would
otherwise keep as a one-frame cluster. Skipped frames are never selected and aren't counted in
the stats. Check frame 0 of any new dataset's videos before trusting it.

## Run

```bash
python scripts/pretraining/dedup_videos.py                    # every dataset
python scripts/pretraining/dedup_videos.py --dataset ibl      # one dataset
python scripts/pretraining/dedup_videos.py --overwrite        # recompute existing
```

Flags: `--model_type` (default `dinov3_vits16`), `--splits` (default `train unlabeled`), `--k`,
`--method {auto,flat,two_level}`, `--two_level_min_frames`, `--device`. Videos without embeddings
yet are skipped (counted in the output), so it can run alongside extraction. Logic:
`mighty_mouse/pretraining/{kmeans,hierarchical,dedup}.py`. Needs a GPU for long videos
(~25-45 s and up to ~12 GiB for a 0.5-1M-frame two-level video; CPU is much slower).

## Output

`<pretraining>/dedup/<model_type>/<dataset>/<video_key>.npz`:

- `frame_idx` — int32, sorted indices of the kept frames (into the video's embeddings `.npy`)
- `cluster_size` — int32, size of the cluster the kept frame was picked from (1 for `whole`).
  For `two_level` it's the level-1 cluster, and sizes do **not** sum to the frame count
  because unchosen clusters are dropped.
- scalars: `n_frames` (including skipped), `k`, `seed`, `skip_first`, `method`, `k1`, `k2`,
  `n_iter`, `n_iter_level2`, `n_empty_clusters`, and `coverage_mean/p95/max` (cosine distance
  from every usable frame to its nearest kept frame — for comparing selection recipes; only
  comparable within a video/dataset, since scales differ a lot between datasets)

## Gotchas

- **Resume-safe, but not parameter-aware.** Existing outputs are skipped unless `--overwrite`
  (written atomically). If existing outputs were made with a different `k`, method, or skipped
  frames than the current run would use, the run prints one aggregated warning with counts per
  dataset and leaves them alone — pass `--overwrite` to redo them. Files from before the
  `method` field existed count as flat.
- The random choices are seeded from the video key, so results don't depend on run order or on
  what else is processed.
- Embeddings are stored in float16; normalization happens in float32 chunks after loading.
- Two-level keeps a one-frame outlier cluster only with probability ~1/3, unlike flat, where
  outliers (including junk frames) are always kept.
- Needs `torch` (imported lazily; not in `pyproject.toml`/CI, so the k-means tests skip
  without it).
