# kaufman dataset changelog

Calibrated multi-view reaching-task recordings, already in standard DLC layout — no
custom conversion script was needed, so this folder exists only to track
keypoint-level changes to the source labels over time. See
[`README.md`](README.md) for the source format and
[`configs/datasets/kaufman.yaml`](../../../configs/datasets/kaufman.yaml) for the
current keypoint mapping.

## Changelog

### 2026-09-29 (MW)
- Narrowed `POST_PROCESS["kaufman"]` in `scripts/convert_dataset.py` to force
  `visible=0` only on the `_left` forepaw keypoints (`wrist_left`, `d[1-4]_tip_left`).
  The `_left` face keypoints added in version 1 (`eye_*`, `ear_*`, `pad_*`) now keep
  the default `visible=1`, since that side of the face is genuinely occluded from this
  camera. No label CSV changes.

### 2026-09-28 (MW) (version 1)
- Removed the `Nose1`, `Nose2`, `Tng1`, `Tng2` columns (previously excluded in
  `configs/datasets/kaufman.yaml`) to declutter labeling; their original labels are
  preserved in `versions/*_version0.csv`.
- Added 13 new keypoint columns: `eye_back`, `eye_bottom`, `eye_front`, `eye_top`,
  `pad_top`, `pad_side`, `ear_base`, `ear_bottom`, `ear_tip`, `ear_top` (mapped to
  `*_{side}`), and `nose_tip`, `nose_top`, `nose_bottom`; added the matching entries
  to `configs/datasets/kaufman.yaml`.
- Initial pseudo-labels for `nose_tip`, `nose_top`, `nose_bottom`, `pad_top`,
  `pad_side` from LA's model trained on cheese-2d/cheese-3d/kondo (likelihood >= 0.7),
  via `scripts/transfer_pseudo_labels.py --predictions_csvs`.
- Iterative bootstrapping for `eye_*`, `ear_*`, `pad_top`, `pad_side`, `nose_tip`,
  `nose_bottom` (not `nose_top`): trained a standalone LP model on the leading
  hand-corrected rows of `CollectedData.csv` (504, 730, 811, 1050, 1222, 1610 rows),
  overwrote the remaining rows and all of `CollectedData_test.csv` with its
  predictions (likelihood >= 0.7), then hand-corrected further — until all of
  `CollectedData.csv` was manually corrected. A final model trained on the full train
  CSV pseudo-labeled `CollectedData_test.csv`, which was then also manually corrected.
  Configs/models: `poseinterface/configs/kaufman.yaml`, `poseinterface/results/kaufman/`.

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of kaufman's label CSVs, as received — no manual
  edits have been made to this dataset's labels yet.
