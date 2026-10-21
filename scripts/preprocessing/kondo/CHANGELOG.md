# kondo dataset changelog

Already in standard DLC layout — no custom conversion script was needed, so this
folder exists only to track changes to the source data over time. See
[`configs/datasets/kondo.yaml`](../../../configs/datasets/kondo.yaml) for the
current keypoint mapping.

## Changelog

### 2026-09-26 (MW) (version 2)
- Manually adjusted the `earroot`/`earlateral`/`eartip` labels to be consistent with
  the ear landmark definitions used in `cheese-2d`.
- This snapshot also includes undocumented manual adjustments to other keypoints relative
  to v1 (`noseroot`, `lowerjaw`, `nosetip`, `eyelateral`, `eyemedial`, `nosebottom`,
  and a few frames of `lickport`/`tonguetip`); the date of these edits is unknown.

### 2026-09-22 (MW)
- Ran `scripts/preprocessing/extract_clips.py` (see
  [`cazettes-side/CHANGELOG.md`](../cazettes-side/CHANGELOG.md) for where that script
  came from) on the four raw face-camera recordings in
  `_raw/_dlc/MouseViewsDLC/dlc-projects/mouse-forelimb-v1/videos/`
  (`sub-VG1-GC#125_ses-2024-04-19-task-day2`,
  `sub-VG1-GC#127_ses-2024-04-16-task-day1`,
  `sub-VG1-GC#129_ses-2024-03-07-task-day1`,
  `sub-VG1-GC#130_ses-2024-04-16-task-day1`), using the script's defaults
  (`--skip-start 60`, `--clip-length 15`), to produce short high-motion clips in
  `_raw/kondo/videos_test/` for review as labeling candidates. h264/yuv420p/mp4,
  15s each. This is raw-video review, not a label change —
  `CollectedData.csv`/`CollectedData_test.csv` are untouched.

### 2026-08-17 (MW) (version 1)
- Manually adjusted the `rightpawcenter` and `leftpawcenter` labels to align them
  more closely with the IBL paw labels.
- v1 is a reconstruction: v0 with only the `rightpawcenter`/`leftpawcenter` columns taken
  from the current labels (made 2026-10-20; no snapshot was kept at the time).

### 2026-08-14 (MW) (version 0)
- Initial data (`CollectedData.csv`/`CollectedData_test.csv`, `labeled-data/`,
  `project.yaml`) extracted into `_raw/kondo/` from
  `_raw/_dlc/MouseViewsDLC/dlc-projects/mouse-face-v1`.
- v0 is the `iter1`–`iter6` rows of that project's per-iteration `CollectedData_KS.csv`
  files, with `iter6` as the test set (439 train / 120 test frames, 14 keypoints); the
  `iter7_*` rows were not used. Regenerated from the DLC source on 2026-10-20.
