# Selecting ~1M pretraining frames from the video corpus

Design notes for the frame-selection part of the masked-autoencoder backbone pretraining data
pipeline: why stages 1 and 2 work the way they do, the evidence behind their choices, and how
the first 100k-frame comparison sets were built. How to *run* each step lives in the skills
([`extract-embeddings`](../skills/pretrain/extract-embeddings/SKILL.md),
[`dedup-videos`](../skills/pretrain/dedup-videos/SKILL.md),
[`select-frames`](../skills/pretrain/select-frames/SKILL.md),
[`extract-frames`](../skills/pretrain/extract-frames/SKILL.md)); this note doesn't repeat them.
Numbers below are a snapshot (Oct-Nov 2026) of the corpus and experiments, not current facts.

## Goal and constraints

Pick about 1M frames (an exact count is not needed) from a corpus of ~94M frames of
head-fixed-mouse video, to pretrain a ViT backbone with masked autoencoding. What matters:

- **Diversity over redundancy.** Fewer frames from dense regions of embedding space (a long
  still pose), while **outliers and rare poses are retained**. Rapid movement must not be lost.
- **Only train and unlabeled videos.** Test videos are never used (they're embedded too, but
  every later step filters them out).
- **Datasets are balanced, not proportional.** ibl alone is ~62% of the frames. It should be
  oversampled relative to the others but not at its raw share. Duplicate views of one
  recording (aind body/face, kondo/kondo-front) count as separate datasets.
- **Incremental.** New videos (many `path: null` entries in the videos paths file are still
  unfilled) must be ingestible without redoing other videos' work, which is why stage 1 is
  strictly per-video.

## Pipeline

| Step | What | Status |
|---|---|---|
| 0. Embed | frozen DINOv3 ViT-S/16 CLS token per frame, float16 (`extract_embeddings.py`) | done |
| 1. Per-video dedup | keep at most 15k representative frames per video (`dedup_videos.py`) | done |
| 2. Dataset budgets + per-dataset selection | split the target over datasets, then choose each share from its stage-1 pool (`select_frames.py`, methods `hier` and `fps`) | done |
| 3. Frame extraction | decode the chosen frames, drop junk, save images (`extract_frames.py`) | built; run for the 100k comparison sets |

Embeddings are CLS-only (storage) at 224 px. All of it lives under `pretrain_dir`.

## Stage 1: per-video dedup

For each video: L2-normalize the embeddings, then keep at most `k = 15000` frames by video
length `n` (thresholds in `mighty_mouse/pretrain/dedup.py`; the table in the skill is the
reference):

- `n <= k`: the whole video.
- `k < n < 100k`: spherical k-means with `k` clusters, one random frame per cluster.
- `n >= 100k`: **two-level** k-means with balanced sampling (`hierarchical.py`). 45k fine
  clusters are grouped into 1500 groups on their centroids; an equal number of fine clusters is
  drawn from each group (small groups give everything), and one random frame is taken per
  chosen cluster.

Why the 15k cap: a fixed per-video number would have made hantman, with ~170 short videos, a
third of the pool, and the cap leaves stage 2 free to budget per dataset. The stage-1 pool is
at most ~6M frames before stage 2.

### Evidence behind the choices

- **Literature.** DINOv2 and DINOv3 curate with (hierarchical) k-means; Vo et al. 2024 find
  hierarchical k-means with balanced sampling beats flat k-means, and that a random frame from a
  cluster is as good as the closest to its centroid and far better than the farthest one. SemDeDup
  and D4 motivate removing near-duplicates before clustering, and show that which duplicate is
  kept hardly matters.
- **Random member per cluster.** Cluster members in contact sheets were near-duplicates of the
  kept frame, so there's no reason to prefer a centroid.
- **Flat vs two-level.** Compared on coverage (cosine distance from every frame to its nearest
  kept frame; comparable only within a video), tail coverage (p99.9), the share of the densest
  state, rare-state share (10 rarest of 50 coarse clusters) and the keep rate of fast-motion
  frames. Two-level wins on all of them for long videos at a ~3% cost in mean coverage and ~2.6x
  the run time. Flat keeps every isolated outlier as a singleton cluster, two-level keeps one
  ~1/3 of the time.
- **Rapid motion** is kept at 1.1-1.9x its proportional rate by both methods, so nothing is
  thinned beforehand. Dropping the slowest 10-30% of frames by CLS motion energy first gave small
  gains and cost coverage on slow stretches, so it was left out (it may return as a weight in
  stage 2).
- **Junk frames.** The aind-vbn videos start with an encoder-info placeholder frame, which
  `SKIP_FIRST_FRAMES` drops; scara and costa-peterka skip their first minute and 35 s
  (`SKIP_FIRST_SECONDS`, converted per video at its frame rate). Frame 0 of ibl, cheese-3d and
  cheese-2d is real.

### The 100k threshold

It was set provisionally and then tested by (a) truncating the longest ibl, aind-body and facemap
videos to 50k-350k frames and (b) running both methods on 28 real videos of 100-250k frames
(kondo, kondo-front, steinmetz, scara, cheese-3d, ibl), two seeds each. Ratios below are
two-level / flat, so under 1.00 is better for coverage:

- From 100k frames up, two-level has better p99.9 coverage on every dataset tested (0.86-1.00),
  under-weights the densest state more, and gives rare states 8-30% more share. The mean-coverage
  cost is 1-7% for most datasets, 9-14% for scara, 13-17% for cheese-3d.
- At 50k frames it's about even on aind and facemap and clearly worse on ibl (+16% mean).
- cheese-3d at 105-123k is the marginal case: a large mean cost for a 0-5% tail gain. There was
  no strong enough reason for an exception (12 of its 78 videos). Use
  `--two_level_min_frames` on that dataset to keep them flat if wanted.

## Stage 2: dataset budgets and per-dataset selection

`select_frames.py --method {hier,fps} --size 100k` pools each dataset's stage-1 frames (never
test videos) and writes `selection.csv`.

1. **Budgets** (`budgets.py`). The target is split over datasets in proportion to
   `frames^alpha` with `alpha = 0.3` by default, using each dataset's raw frame count, and a
   dataset never gets more than its stage-1 pool (the surplus is shared among the others). Alpha
   oversamples big datasets (ibl) relative to small ones without following raw proportions.
   Budgets are rounded to sum to exactly the target.
2. **Selection inside each dataset** (`selection.py`), from the pool of unit-length embeddings:
   - `hier`: balanced two-level k-means (`hierarchical.py`; level-1 clusters 3x the budget,
     groups = clusters / 30). It under-weights dense stretches and keeps rare poses, and
     behaves like stage 1's long-video method.
   - `fps`: greedy farthest-point sampling (`farthest_point.py`). Dense regions are thinned
     first and isolated frames are kept, so it is the most aggressive about rare poses, at a
     cost in typical-frame coverage and a tendency to pick up junk.
3. **Output.** `<pretrain_dir>/pretrain_datasets/<size>_method<method>/selection.csv` (dataset,
   video key, frame index), which stage 3 turns into images. The k-means chunk size scales with
   the number of clusters (`kmeans.chunk_for`), since the fixed stage-1 chunk runs out of memory
   at the cluster counts used here.

### What the comparison found

Compared on 16 datasets at a 1M-frame budget, using a random sample of ~100k frames per dataset
from the *full* embeddings: coverage (cosine distance to the nearest selected frame, mean and
p99.9), the share taken from the densest state, the share from the rarest states, the keep rate
of fast-motion frames and the fraction of the 1% most isolated frames kept. Medians relative to
flat k-means (the stage-1 recipe) at equal budgets:

| | mean coverage | p99.9 coverage | densest-state share | rare-state share | isolated kept |
|---|---|---|---|---|---|
| flat k-means | 1.00 | 1.00 | 1.00 | 1.00 | 0.36 |
| `hier` | 1.07 | 0.97 | 0.89 | 1.11 | 0.50 |
| `fps` | 1.29 | 0.84 | 0.72 | 1.38 | 1.00 |

`fps` does best on every tail metric at ~29% worse mean coverage, and over-picks dark frames
(ibl, hantman), a few blown-out or blank ones and cable-crossing frames (aind); the extraction
junk screen below is the only guard. Outlier sheets of its most isolated frames were otherwise
mostly real rare poses.

A third method, one global two-level k-means over all datasets pooled (method B, k1 = 1.5x the
target, 10k groups), was dropped. It selected like `hier` within a dataset (all metrics within
a few percent) but took about 2 h instead of minutes, and its allocation follows the pool size
and cluster scatter instead of a setting: ibl and warren got 28% each of the 1M frames, while
facemap, cheese-2d, hantman-mv, costa-peterka and deepfacemouse got 12-35% of their alpha
budget. Warren's low-resolution faces make that share doubtful for a face-pose project.

### Decisions made

- About 1M frames is fine; an exact count is not required.
- Alpha budgets over cluster-derived allocation; both `hier` and `fps` are kept as options.
- Junk screening happens at extraction (stage 3), not in stage 2.

### Open

- The choice of alpha and of `hier` vs `fps` for the 1M set. The first 100k pretraining runs
  (hier, fps and B sets) showed no obvious differences between the methods.
- Re-check the stage-1 100k threshold if a new dataset's video lengths land in an untested range.

## 100k comparison sets and stage 3

The first pretraining comparison used ~100k-frame sets, one per method, each selected natively
at 100k (not subsampled from a 1M selection). They were made with prototype code of the same
method as `select_frames.py` (per-dataset budgets agree within 0.5%; the frames are a different
random draw, about 45% shared for `hier`), and their directories on disk use the prototype names:
`100k_methodA1` is `hier`, `100k_methodA2` is `fps`, and `100k_methodB` is the dropped global
k-means. New sets are named `<size>_method<hier|fps>`.

Junk dropped at extraction in those sets: `hier` 104 frames (0.10%), `fps` 290 (0.29%; hantman
1.9%, ibl 1.1%), B 115 (0.11%).

**Stage 3.** `extract_frames.py` writes `<dataset>/<video_key>/img<frame:08d>.png` (the layout
the beast repo reads), with the short side downscaled to 256 (smaller frames kept as they are;
the 1M set can use a larger side later). Junk frames are dropped, not backfilled: gray std < 10
(blank/black/flat) or more than half the pixels brighter than 245 (blown out); the run errors
if over 0.5% of a method's frames, or 2% of a dataset's, are junk. The rule was calibrated on
contact sheets of A2 and random pool frames; tinted and dark-but-visible frames (whole real
sessions in ibl, scara, costa-peterka) are deliberately kept.
