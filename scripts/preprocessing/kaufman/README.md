# kaufman dataset conversion

Stage 2 complete — kaufman is in the combined corpus. See
[`configs/datasets/kaufman.yaml`](../../../configs/datasets/kaufman.yaml) for the
current keypoint mapping.

## Source format

`_raw/kaufman/` arrived already in standard DLC layout (`labeled-data/<session>/`,
`CollectedData_Hank.csv`, `config_multiview_384.yaml`, `calibration/`) — a calibrated
**multi-view** Lightning Pose project: 104 `labeled-data/<timestamp>-cam[12]/` dirs (52
session timestamps x cam1/cam2), each with its own per-session
`CollectedData_Hank.csv`, sharing one `calibration.toml`/`calibrations.csv` and a
`model_type: heatmap_multiview_transformer` config.

27 keypoints, scorer `Hank`, identical schema across all 104 sessions (verified).

## Decisions

- **Video encoding**: the 24 sample videos in `videos/` are already h264/yuv420p/mp4 —
  no re-encoding needed.
- **Single-view, by explicit request**: despite the calibrated multi-view source,
  this is being built as a **single-view** dataset — cam1 and cam2 are kept as
  independent rows/sessions, not merged into a multi-view sample. `project.yaml`
  (hand-written, keypoint names copied from `config_multiview_384.yaml`) has
  `view_names: []`.
- **Keypoints**: stage 1 carries through all 27 source keypoint names as-is (laterality
  is already baked into the source names, e.g. `LFPm` vs `RHPm`). Stage 2 keeps only
  the 4 right-forepaw digit tips (`RFPf1`-`RFPf4` → `d4_tip`-`d1_tip`, note the reversed
  digit order — see `configs/datasets/kaufman.yaml`), all already-canonical, lateralized
  to `_right` since only the right forepaw's fingers were ever labeled (the left forepaw
  only has a few coarse keypoints, `LFPm`/`LFPl`/`LFPp`); the other 23 keypoints are
  excluded.
- **Train/test split**: grouped by **subject**, using the mouse-to-session mapping the
  user supplied at `_raw/_dlc/kaufman/labeled-sessions-mouse-date-time.txt` (one
  `<mouse>/<timestamp>` per line — all 52 labeled-data timestamps resolve to one of 6
  mice, b8sSM5 through b8sSM10). By explicit user decision, **b8sSM7 and b8sSM10 are
  held out entirely for test**, and b8sSM5/b8sSM6/b8sSM8/b8sSM9 are train — not a
  random/percentage split. Splitting by subject also keeps a session's cam1 and cam2
  views (the same trial, viewed twice) on the same side of the split, since treating
  them as independent would leak near-duplicate frames across train/test.

  Result: 35 train / 17 test timestamps, 1808 / 732 frames (28.8% test) — well above the
  usual 10-15% target, accepted because the user wanted these two specific subjects
  held out for test regardless of the resulting fraction.
- **`videos_test/`**: `videos/` and `videos_test/` were reorganized by hand to match the
  subject split — all b8sSM7/b8sSM10 sample videos moved into
  `_raw/kaufman/videos_test/`, the rest remain in `videos/`. `convert_kaufman.py` only
  builds the `CollectedData*.csv` files; it doesn't move videos.

## Running

```bash
python scripts/preprocessing/kaufman/convert_kaufman.py
```

Concatenates all 104 per-session CSVs and writes `_raw/kaufman/CollectedData.csv` /
`CollectedData_test.csv`. No image copying needed — everything already lives at its
final `_raw/kaufman/` path. Spot-checked by overlaying keypoints from the first train
row on its source image; all 27 land in anatomically correct positions (nose/tongue at
the mouth, spout keypoints on the water spout, forepaw digit keypoints clustered on the
visible paw, hindpaw and left-paw points in the right spots).

## Stage 2

`configs/datasets/kaufman.yaml` maps the 4 right-forepaw digit tips to their canonical
names and excludes everything else; all already existed in
`configs/keypoints.yaml`/`configs/model.yaml`, so no vocab changes were needed. All 104
sessions are declared `right`.

The left forepaw is visible, but its fingers were never labeled (it only has a few
coarse keypoints, `LFPm`/`LFPl`/`LFPp`, which are excluded). The default per-split
output would mark every `d[1-4]_tip_left` `visible=1` ("in dataset, unlabeled") rather
than `visible=0` ("not part of this dataset") — training on that would teach the model
to predict a suppressed heatmap for fingers that are there but were simply never
labeled. A
`POST_PROCESS["kaufman"]` entry in `mighty_mouse/convert.py` forces the `_left`
forepaw columns (`d[1-4]_tip_left`, plus `wrist_left` if it's ever mapped) to
`visible=0` after the standard split processing (same pattern as `hantman-mv`). The
lateralized face keypoints added in version 1 (`eye_*`, `ear_*`, `pad_*`) are *not*
overridden: the left side of the face is genuinely hidden behind the head from this
camera, so their default `visible=1` (occluded) is the right signal.

`kaufman` was added to `ALL_DATASETS` in `mighty_mouse/datasets.py`.

```bash
python scripts/convert_dataset.py --dataset kaufman
python scripts/build_dataset.py --tag <tag> --datasets kaufman ...  # not yet run
```
