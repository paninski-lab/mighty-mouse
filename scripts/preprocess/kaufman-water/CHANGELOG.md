# kaufman-water dataset changelog

Earlier (2020-2022) calibrated multi-view reaching-task recordings from the same lab as
`kaufman`, already in standard DLC layout. See [`README.md`](README.md) for the
conversion and [`configs/datasets/kaufman-water.yaml`](../../../configs/datasets/kaufman-water.yaml)
for the current keypoint mapping. This file tracks keypoint-level changes to the labels.

## Changelog

### 2026-10-24 (MW) (version 0)
- First versioned snapshot of kaufman-water's label CSVs, as received from
  `_raw/_dlc/kaufman-water` — no manual edits have been made to this dataset's labels.
  All 27 source keypoints are kept; per-session CSVs were concatenated into
  `CollectedData.csv` / `CollectedData_test.csv` with a subject-level train/test split.
