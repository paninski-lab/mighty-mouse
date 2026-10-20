# cavelli dataset changelog

Left- and right-camera side views of head-fixed mice, converted from a DLC project by
[`convert_cavelli.py`](convert_cavelli.py). See [`README.md`](README.md) for the source
format and [`configs/datasets/cavelli.yaml`](../../../configs/datasets/cavelli.yaml) for
the current keypoint mapping.

## Changelog

### 2026-10-20 (MW) (version 1)
- Manually adjusted all six keypoints in `CollectedData.csv` and `CollectedData_test.csv`
  to bring TR's labels in line with the ibl labeling conventions. Nearly every labeled
  cell moved (median shift roughly 2-10 px per keypoint). Label count changes, train:
  `wrist_left` +17, `wrist_right` +13 (−1), `pupil_center_left` −9,
  `pupil_center_right` −56. Test: `wrist_left` +4, `nose_bottom` +1 (−3).

### 2026-10-20 (TR) (version 0)
- First versioned snapshot of cavelli's label CSVs: TR's original annotations, as
  received, converted from the contributed DLC project (`_raw/_dlc/cavelli`) by
  `convert_cavelli.py`. The 38 keypoints with no labels and the `visible` columns were
  dropped, leaving `pupil_center_left`, `pupil_center_right`, `nose_tip`, `nose_bottom`,
  `wrist_left`, `wrist_right`. Left and right cameras are merged; subject-level split
  (M3, M6 test: 47 of 265 rows).
