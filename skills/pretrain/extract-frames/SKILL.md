---
name: extract-frames
description: Use when the user wants to turn a frame selection for MAE backbone pretraining (a selection.csv of dataset / video_key / frame_index under <pretrain_dir>/pretrain_datasets/<size>_method<m>/) into image files — decodes the frames, drops junk (blank or blown-out) frames, downscales, and saves PNGs in the layout the beast repo reads.
---

# Stage 3: extract selected frames as pretraining images

Part of the pretraining data pipeline; the selections come from stage 2 (see
[`docs/pretrain_frame_selection.md`](../../../docs/pretrain_frame_selection.md)). Methods are
named in that doc (A1, A2, B, ...).

## Layout

```
<pretrain_dir>/pretrain_datasets/<size>_method<m>/
  selection.csv          dataset, video_key, frame_index (input)
  dropped_frames.csv     junk frames that were not saved, with their stats (output)
  <dataset>/<video_key>/img<8-digit frame index>.png
```

beast finds images with `rglob('*.png')` and treats a frame's parent folder as its video, so
video keys must be unique across datasets; the script refuses to run otherwise.

## What it does

`scripts/pretrain/extract_frames.py --methods A1 A2 B [--size 100k] [--workers 8]`

- Reads every needed video once (even if several methods chose frames from it). Frames less
  than 300 apart are decoded forward, farther ones are seeked to (much cheaper on long videos;
  seeked frames were bit-identical to sequentially decoded ones on all 595 videos, so frame
  indices match the embeddings'). If a seek isn't honored, that video is read sequentially.
- Downscales so the short side is `--min_side` (default 256); smaller frames are left alone,
  never upscaled.
- Drops junk frames: gray std below 10 (blank/black/flat), or more than half the pixels brighter
  than 245 (blown out). Tinted and dark-but-visible frames are kept on purpose (they are real
  sessions). The rule was calibrated on contact sheets of A2 and pool samples; the thresholds
  live in `mighty_mouse/pretrain/frames.py`.
- Exits with an error if more than 0.5% of a method's frames are junk overall, or 2% in any one
  dataset (after writing everything, so the dropped list can be inspected).
- Skips frames already saved or already dropped, so reruns resume.

Dropped frames are not backfilled, so a set ends slightly under its nominal size.
