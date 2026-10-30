# ibl dataset changelog

See [`README.md`](README.md) for the pseudo-labeling pipeline.

## Changelog

### 2026-10-24 (MW) (version 2)
- Added four ear keypoints (`ear_base`, `ear_tip`, `ear_top`, `ear_bottom`) to the schema.
  All ear labels were manually reviewed and adjusted by hand in the Lightning Pose app
  (not itemized per frame). Not every frame has every ear point: `ear_tip` and `ear_top`
  are labeled on fewer frames (~3.9k / ~2.5k of 7608 train) than `ear_base` / `ear_bottom`
  (~6.9k / ~6.8k).
- A small number of coordinates in existing keypoints also differ from v1 (about 202 train
  and 10 test); these edits were not itemized.

### 2026-06-18 (MW) (version 1)
- Added `nose_tip`, `pupil_center_r`, `tongue_end_r`, and `tongue_end_l` as new keypoints
  to the `ibl-paw` schema (`paw_l`, `paw_r`), filled in with pseudo-labels from the
  [iblvideo](https://github.com/int-brain-lab/iblvideo) Lightning Pose pipeline run on
  per-session videos built from each labeled frame plus 2 context frames either side.
  `nose_tip` and tongue predictions with likelihood < 0.9 were set to NaN, and
  `pupil_center_r` is the median of the 4 predicted pupil points. See
  [`README.md`](README.md) for the full pipeline.
- In July 2026 all ~10k frames (7608 train / 1446 test) were manually reviewed and
  corrected in the Lightning Pose app, so these are now full human annotations rather
  than pseudo-labels. The review also made small adjustments to `paw_l`/`paw_r` (about
  18 train and 8 test coordinates relative to v0).

### 2026-06-15 (MW) (version 0)
- Imported the `ibl-paw` dataset from `_raw/_dlc/ibl-paw` as received: paw-only labels
  (`paw_l`, `paw_r`; scorer `lightning_tracker`) for 7609 train and 1446 test rows. The
  train CSV contains one duplicated row (`8d316998-..._right_flipped/img00305878.png`).
