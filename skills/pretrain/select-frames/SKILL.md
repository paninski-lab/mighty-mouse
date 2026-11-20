---
name: select-frames
description: Use when the user wants to build a frame selection for MAE backbone pretraining (stage 2) — choose the frames of a subset like 100k or 1m from the stage-1 per-video selections, splitting the size over datasets by frames^alpha and picking each share by balanced k-means (hier) or farthest-point (fps). Outputs a selection.csv that extract-frames turns into images.
---

# Stage 2: choose the frames of a pretraining subset

Part of the pretraining data pipeline; needs stage-1 outputs from
[`dedup-videos`](../dedup-videos/SKILL.md) and the embeddings from
[`extract-embeddings`](../extract-embeddings/SKILL.md). Its output feeds
[`extract-frames`](../extract-frames/SKILL.md). Why these methods, and what the comparison
found: [`docs/pretrain_frame_selection.md`](../../../docs/pretrain_frame_selection.md).

## What it does

`python scripts/pretrain/select_frames.py --method hier --size 100k [--alpha 0.3]`

1. Reads every train/unlabeled video's stage-1 output (test videos are never used). Videos
   without one yet are left out and reported.
2. Splits `--size` over datasets in proportion to `frames^alpha` (default 0.3), each dataset
   capped at its stage-1 pool with the surplus shared among the rest. Prints the table.
3. Picks each dataset's share from its pool, on the GPU if there is one:
   - `hier`: balanced two-level k-means; under-weights dense stretches, keeps rare poses.
   - `fps`: farthest-point sampling; the strongest on rare and isolated frames, but weaker
     typical-frame coverage, and it picks up more junk (dark, blank and blown-out frames), which
     the extraction screen then drops.
4. Writes `<pretrain_dir>/pretrain_datasets/<size>_method<method>/selection.csv`. It refuses to
   overwrite an existing one without `--overwrite`.

`--size` takes forms like `100k` or `1m`. Both methods are seeded (`--seed`), so reruns on the
same stage-1 outputs give the same selection.

## Notes

- Budgets depend on every dataset's stage-1 output, so adding videos changes the split; rerun
  the whole thing for a new set rather than patching one.
- Run time: minutes for `hier` and `fps` at 100k; the largest dataset (ibl) dominates and
  grows with the size.
- Next step: `python scripts/pretrain/extract_frames.py --methods <method> --size <size>`.
