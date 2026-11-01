# Selecting ~1M pretraining frames from the video corpus

Design notes for the frame-selection part of the masked-autoencoder backbone pretraining data
pipeline: why stage 1 works the way it does, the evidence behind its choices, and the plan for
stage 2. How to *run* each step lives in the skills
([`extract-embeddings`](../skills/pretrain/extract-embeddings/SKILL.md),
[`dedup-videos`](../skills/pretrain/dedup-videos/SKILL.md)); this note doesn't repeat them.
Numbers below are a snapshot (Oct 2026) of the corpus and experiments, not current facts.

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
| 2. Dataset budgets + cross-video selection | pool stage-1 frames, choose ~1M | planned, below |
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

## Stage 2: dataset budgets and cross-video selection (plan)

**Input:** the stage-1 selections, pooled per dataset (~1.9M frames for ibl, which fits on the
GPU). **Output of this phase:** comparison tables only; repo code comes after a winner is
chosen. Everything is prototyped in the scratchpad first.

1. **Pool loader.** Concatenate each dataset's selected frames' normalized embeddings, with video
   key, frame index and stage-1 `cluster_size`.
2. **Budget table.** `frames^alpha` per dataset for alpha in {0.2, 0.3, 0.4, 0.5}, using raw
   frame counts and the dataset's pool size as a cap with the surplus redistributed (facemap and
   hantman-mv are tight). Choose alpha by looking at the table next to raw proportions.
3. **Method A: alpha budgets, then selection inside each dataset.**
   - A1: balanced hierarchical k-means (reuse `hierarchical.py`, cap = budget).
   - A2: greedy farthest-point / epsilon-net (minimum-distance rule: dense regions are thinned
     first, isolated frames are kept). Exact greedy is feasible on the GPU for ibl.
   - A3: flat k-means, one random member per cluster (the stage-1 recipe, as the baseline).
4. **Method B: global hierarchical k-means (Vo et al.).** Two levels over the whole pool
   (k1 around 1.5M, about 10k groups), balanced sampling to 1M. Dataset budgets then come from
   the cluster structure instead of alpha. This is the expensive arm (an estimated 20-30 min on
   the GPU), so try a 10% subsample first.
5. **Evaluation.** On a random sample of ~100k frames per dataset from the *full* embeddings (not
   only the stage-1 pool): nearest-selected distance (mean, p95, p99.9), densest-state and
   rare-state shares, fast-motion rate, the fraction of the 1% most isolated frames retained,
   and kept share versus raw share per dataset (to compare B's implied allocation with alpha).
6. **Outlier inspection.** Contact sheets of the most isolated kept frames per dataset for the
   finalists. Embeddings can't separate junk (blank, corrupted, placeholder frames) from a
   genuinely rare pose, and the farthest-point selector in particular favours isolated frames, so
   this is a manual check. Whether it becomes an automatic filter is decided after seeing the
   sheets.
7. **Small pretraining tests.** Build ~100k-frame subsets from the winning approach or
   approaches (needs the frame-extraction step) and pretrain on them.

### Decisions already made

- About 1M frames is fine; an exact count is not required (budget rounding and pool caps make
  exact numbers awkward).
- Run the global method B in the comparison, with a subsample first.
- Junk screening stays manual for now.

### Open

- The choice of alpha, and A1 vs A2 vs A3 vs B, from the stage-2 comparison above.
- The output format of the final selection (something like a per-dataset table of video key and
  frame index under `pretrain_dir`) and the frame-extraction step.
- Re-check the stage-1 100k threshold if a new dataset's video lengths land in an untested range.

## 100k comparison sets and stage 3

The first pretraining comparison uses ~100k-frame sets, one per method, each selected natively
at 100k (not subsampled from a 1M selection). Method names are used in directory names
(`<pretrain_dir>/pretrain_datasets/100k_method<m>/`):

- **A1**: per-dataset budgets `frames^0.3` (water-filled, capped at the stage-1 pool), then
  balanced hierarchical k-means inside each dataset.
- **A2**: the same budgets, then greedy farthest-point selection inside each dataset.
- **B**: one global two-level k-means over the pooled stage-1 frames of all datasets (k1 = 1.5x
  the target, 1000 groups), balanced sampling to 100k. Dataset budgets come from the cluster
  structure.

Findings behind the choice of these three (1M-frame budgets, 16 datasets; scratch experiments,
medians relative to flat k-means at equal budgets): A2 has the best tail coverage (-16% p99.9),
fewest densest-state frames (-27%), most rare-state frames (+22%) and keeps all of the 1% most
isolated frames, at +21% worse mean coverage; A1 sits in between (+4% mean, -3% p99.9). B
selects like A1 but its allocation follows the pool (ibl 28%, warren 28%) and starves small
datasets (facemap, cheese-2d, hantman-mv, costa-peterka, deepfacemouse at 12-35% of their A
budget); it also costs ~2 h versus minutes. A2 over-picks dark frames (ibl, hantman) and some
blown-out or blank ones, hence the junk screen below.

**Stage 3.** `extract_frames.py` writes `<dataset>/<video_key>/img<frame:08d>.png` (the layout
the beast repo reads), with the short side downscaled to 256 (smaller frames kept as they are;
the 1M set can use a larger side later). Junk frames are dropped, not backfilled: gray std < 10
(blank/black/flat) or more than half the pixels brighter than 245 (blown out); the run errors
if over 0.5% of a method's frames, or 2% of a dataset's, are junk. The rule was calibrated on
contact sheets of A2 and random pool frames; tinted and dark-but-visible frames (whole real
sessions in ibl, scara, costa-peterka) are deliberately kept.
