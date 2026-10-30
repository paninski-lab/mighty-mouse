# cheese-2d dataset changelog

Single-view mouse orofacial recordings, already in standard DLC layout — no custom
conversion script was needed, so this folder exists only to track keypoint-level
changes to the source labels over time. See
[`configs/datasets/cheese-2d.yaml`](../../../configs/datasets/cheese-2d.yaml) for the
current keypoint mapping.

## Changelog

### 2026-10-17 (MW) (version 2)
- Manual label review and correction pass (MW) over `CollectedData.csv` and
  `CollectedData_test.csv`, checking per-view consistency (eye/ear/pad/lip sides match the camera
  view, no left/right swaps, train vs. test labeling conventions). No rows added or removed.
  Relative to version 1, `CollectedData.csv` has 66 frames with edited labels (214 moved
  coordinates, 96 newly labeled, 68 labels removed) and `CollectedData_test.csv` has labels
  changed in 8 frames (32 moved coordinates, 28 newly labeled, 250 labels removed).
- Renamed 126 digit-only test frames (e.g. `000.png`) in the `20250518_B1_ephys-record_awake_002`
  sessions to `img000.png` etc. (images on disk and index entries in `CollectedData_test.csv`).
  The labeling app only accepts `labeled-data/<dir>/<letters><digits>.png|jpg` index paths, so
  `CollectedData_test.csv` was previously hidden from its label-file dropdown. The version 0/1
  test snapshots still use the old digit-only names.

### 2026-09-26 (MW) (version 1)
- Filled `pad(top)(left)`, `pad(side)(left)`, `pad(top)(right)`, `pad(side)(right)` (whisker
  pad keypoints) using a single-view heatmap model trained on `cheese-3d`
  (`poseinterface/results/cheese-3d/2026-09-26_15-45-24`, vits_dinov2 backbone), via
  `scripts/transfer_pseudo_labels.py`. Only cells with no existing label and prediction
  confidence >= 0.7 were filled; no new rows were added and no other keypoints were touched
  by this step.
- Followed by a manual review and correction pass (MW) over the whisker-pad labels (both
  the auto-filled ones and pre-existing ones), plus some additional ear-label touch-ups.

### 2026-09-26 (MW) (version 0)
- First versioned snapshot of cheese-2d's label CSVs. This predates the versioning
  scheme, so v0 is the best-available current state rather than a pristine
  collaborator copy — it already includes the undocumented pre-versioning edits
  logged above (the 2026-09-24 LA keypoint infill, 2026-09-25 `POST_PROCESS`
  override removal, and 2026-09-26 manual ear/eye/paw label check).

### 2026-09-26 (MW)
- Manually checked all ear, eye, and paw labels (a mix of manual labels and the
  pseudo-labels from the 2026-09-25 infill), fixed obvious mislabels, and infilled
  remaining missing labels. Adjusted every ear base label to be more consistent.

### 2026-09-25 (MW)
- Removed the `POST_PROCESS["cheese-2d"]` override (`_post_process_cheese2d`) from
  `scripts/convert_dataset.py`. It existed to force `visible=0` on facial keypoints
  that were structurally absent (wrong head orientation for that session) but still
  came through as `visible=1` (annotation gap) from the default per-split processing.
  Now that the (LA) infill below has filled in those gaps directly in the source
  labels, every keypoint that's actually visible in a frame is labeled, so the
  override's correction is no longer needed.

### 2026-09-24 (LA)
- Infilled missing keypoint labels across the source data — including but not limited
  to `wrist_left`/`wrist_right`, added here for the first time — using a model trained
  on `cheese-3d`. Added the `wrist_left`/`wrist_right` mapping entries to
  `configs/datasets/cheese-2d.yaml`.
