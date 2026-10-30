# kaufman-water dataset conversion

Stage 2 complete — kaufman-water is registered in the combined corpus (`ALL_DATASETS`).
See [`configs/datasets/kaufman-water.yaml`](../../../configs/datasets/kaufman-water.yaml)
for the keypoint mapping.

## Source format

`_raw/_dlc/kaufman-water/` is a standard DLC layout with one `CollectedData_Hank.csv`
(+ `.h5`) per `labeled-data/<timestamp>-cam[12]/` dir: 24 dirs (12 sessions x cam1/cam2),
7 mice, same 27-keypoint schema as `kaufman`'s version 0. Sessions map to mice via
`labeled-sessions-mouse-date-time.txt` (`<mouse>/<timestamp>` per line).

## Decisions

- **Separate dataset from `kaufman`** (different dates, mice and setup).
- **Single-view**: cam1 and cam2 are kept as independent rows; `project.yaml` has
  `view_names: []`.
- **Keypoints / laterality**: same as `kaufman` — `RFPf1-4` → `d4..d1_tip_right`, every
  session `right`, the other keypoints excluded. `POST_PROCESS["kaufman-water"]` reuses
  `_post_process_kaufman` (forces `_left` forepaw columns to `visible=0`).
- **Split**: subject-level (mouse), pooled across cams, target the midpoint of 10-15%;
  seed search lands on `cfaVG1` + `cfaVG4` as test (see script output for current counts).
- **Videos**: sample videos are sorted into `videos/` and `videos_test/` by subject. Only
  some sessions have one, and none belong to the test subjects, so `videos_test/` is empty.

## Running

```bash
python scripts/preprocess/kaufman-water/convert_kaufman_water.py [--dry-run]
```

Concatenates the per-session CSVs into `_raw/kaufman-water/CollectedData{,_test}.csv`,
copies only the referenced images (not the per-session CSV/`.h5` files), writes
`project.yaml`, and sorts the videos.
