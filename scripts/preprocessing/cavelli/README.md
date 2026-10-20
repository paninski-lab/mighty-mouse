# cavelli preprocessing

Builds `_raw/cavelli` from the DLC project at `_raw/_dlc/cavelli`.

> **Status:** `cavelli` is in `ALL_DATASETS` (`mighty_mouse/datasets.py`); `convert_dataset.py`
> and `build_dataset.py` haven't been run for it yet. The keypoint mapping is in
> [`configs/datasets/cavelli.yaml`](../../../configs/datasets/cavelli.yaml).

## Source

One `CollectedData_cavelli.csv` (x/y/visible per keypoint) plus `frames_metadata.csv`
(per-frame animal, session, camera, and the reason the frame was picked). Frames come from
a left and a right camera per session (different resolutions), with six animals (M1–M6).
The CSV is a labeling template for a much larger skeleton; only the keypoints below have
any labels.

## What `convert_cavelli.py` does

- Keeps only keypoints with at least one label (see `project.yaml`), drops the `visible`
  columns (raw datasets here are x/y only), and merges both cameras into one project.
- Subject-level (animal) train/test split via `mighty_mouse.subject_split`, targeting
  10–20% test; the seed was chosen so the greedy split lands in range. Read the script
  output or the CSVs for the resulting sizes.
- Reproduces version 0 (TR's labels as received) only: `_raw/cavelli/` holds the
  hand-edited version 1+ labels, so the script refuses to overwrite existing CSVs — point
  `--dst` elsewhere to regenerate. See `CHANGELOG.md`.
- Copies only referenced images, writes `project.yaml` (`view_names: []`) and an empty `videos/`.

All six keypoints already exist in `configs/keypoints.yaml`, so no vocabulary changes
were needed.
