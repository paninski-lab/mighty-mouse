# aind-vbn-body dataset changelog

See [`README.md`](README.md) for the conversion pipeline.

## Changelog

### 2026-10-19 (MW) (version 1)
- Pseudo-labeled 15 keypoints in both CSVs with the `head-fixed_v9/nonlinear_head` model
  (confidence >= 0.9; cells below threshold left empty; existing labels in these columns were
  replaced): `eye_{front,back,top,bottom}_l`, `whisker_pad_l_{top,side}`,
  `ear_{base,bottom,tip,top}_l`, `nose_{top,tip,bottom}` and `paw_wrist_{l,r}`.
- Added columns `eye_front_l`, `eye_back_l`, `ear_base_l`, `ear_top_l`, `nose_top` and
  `nose_bottom`. The old `ear_base_l` column was renamed `ear_bottom_l` and now holds the
  model's `ear_bottom`; `ear_base_l` is now a new column for the model's `ear_base`. Columns
  are reordered into labeling order (eyes, pad, ears, nose, jaw, tongue, paws).
- Dropped the unused `nostril_l` / `nostril_r` columns.
- Manual relabeling in the labeling app. Eyes: all four eye predictions were blanked and
  relabeled by hand in both splits (488 of 489 train rows, all 74 test rows). Nose:
  predictions were blanked for 250 train rows (from
  `1070961372_544835_20201216.behavior/img1813.png` to the end of the file) and all test rows,
  and the nose labels were then redone by hand. The other pseudo-labeled columns (pad, ears,
  wrists) may have had hand corrections since; the edits aren't itemized here.

### 2026-09-29 (MW) (version 0)
- Initial conversion from the `.behavior` projects in `_raw/_dlc/aind-vbn` via
  `convert_aind_vbn.py --view body`: 563 frames (489 train / 74 test, 16 sessions, 16 mice,
  2 test mice), plus 15 s clips of each session video.
- Cropped each frame to face + upper trunk (box from `nose_tip`/eye keypoints, see
  [`README.md`](README.md)) and resized to 256x256. The 15 s video clips are cropped the same
  way, using each session's median box over its labeled frames.
- Dropped `*_lh`, `*_rh` and `tail_*` keypoints (40 -> 25).
- Drafted `configs/datasets/aind-vbn-body.yaml` (8 keypoints mapped, 17 excluded, all
  sessions `left`). Stage 2 not started.
