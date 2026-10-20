# cavelli preprocessing

Builds `_raw/cavelli` from the DLC project at `_raw/_dlc/cavelli`.

> **Status: stage 1 only.** `_raw/cavelli/` is a usable standalone LP project, not yet in
> the combined corpus. See [`skills/preprocess-new-dataset/SKILL.md`](../../../skills/preprocess-new-dataset/SKILL.md) for what
> stage 2 would involve; don't start it unless asked.

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
- Copies only referenced images, writes `project.yaml` (`view_names: []`) and an empty `videos/`.

`configs/datasets/cavelli.yaml` is a draft; all six keypoints already exist in
`configs/keypoints.yaml`, so stage 2 would need no vocabulary changes.
