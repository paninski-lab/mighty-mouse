# facemap dataset changelog

Single-camera orofacial mouse recordings from the public
[Facemap dataset](https://doi.org/10.25378/janelia.23712957) (Syeda et al. 2024),
already in standard DLC layout — no custom conversion script was needed, so this
folder exists only to track keypoint-level changes to the source labels over
time. See
[`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
for the current keypoint mapping.

## Changelog

### 2026-09-22 (MW) (version 2)
- Added manual labels for the `pupil` keypoint to `CollectedData.csv` and `CollectedData_test.csv`.
  Added `pupil: pupil_center_{side}` to the `keypoints` mapping in
  [`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml).
  `pupil_center_left`/`pupil_center_right` already existed in `configs/keypoints.yaml`/
  `model.yaml` (populated via `ibl`), so no vocab changes were needed. `convert_dataset.py` still
  needs to be re-run to pick up the new labels.

### 2026-09-17 (MW) (version 1)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints
  (all rows empty — ears are never visible in this view) to
  `CollectedData.csv` + `CollectedData_test.csv`, `project.yaml`, and the
  `keypoints` mapping in
  [`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
  (`ear_*: ear_*_{side}`). All four canonical names already existed in
  `configs/keypoints.yaml`/`model.yaml`, so no vocab changes were needed.
- This snapshot also includes undocumented manual adjustments to the `nose_bottom`,
  `nose_tip`, and `nose_top` labels relative to v0 (roughly 150–280 train frames per
  keypoint, 15–29 test frames; a handful of other keypoints changed on a few frames).

### 2026-06-10 (MW) (version 0)
- Imported the public Facemap dataset (2,400 InD + 100 OOD labeled frames, 31 columns =
  15 keypoints, scorer `All`) as received, with no label changes. Snapshot taken from
  the original label CSVs preserved in `_raw/facemap/.old/` (`CollectedData_v0.csv` and
  `CollectedData_test_v0.csv`, dated 2024-05-30).
