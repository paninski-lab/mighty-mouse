# murphy-1 preprocessing

Builds `_raw/murphy-1` (one merged project) from the three per-camera DeepLabCut
(TensorFlow) projects at `_raw/_dlc/murphy-1/{cama,camb,camc}`.

Stage 2 complete — murphy-1 is registered in the combined corpus (`ALL_DATASETS`).
See [`configs/datasets/murphy-1.yaml`](../../../configs/datasets/murphy-1.yaml) for the
keypoint mapping.

## Source

Three DLC projects (one per camera), each with the same 29 keypoints and scorer `tf`, and one
`labeled-data/<session>/CollectedData_tf.csv` per session folder (1200x800 frames). No
train/test split ships with it.

## Design decisions

- **Cameras merged** into one project (`view_names: []`). Sessions are renamed
  `<source folder>_<cam>` since the same folder name appears in several cameras.
- **Keypoints kept:** digit tips only (`r{1-4}_tip_A`, `l{1-4}_tip_A`). Dropped: `*_mid_A`,
  `*_joint_A`, `r_mid_A`/`l_mid_A`, `snout_A`, `mouth_tip_A`, `spout_tip_A` (an object, not a
  body part). Output names are the source names.
- **Laterality:** `r`/`l` assumed to be the animal's right/left, in all cameras (not verified).
  The config maps them to `d{1-4}_tip_{right,left}`. A paw with 0 tips
  labeled is treated as occluded (default `visible=1`, no custom `POST_PROCESS`).
- **Folders skipped:** session folders without a `CollectedData_tf.csv`, and `*.cache` files.
  Outlier folders (`*outlier*`, `*_out`) are kept.
- **Session name = folder name**, not the CSV's own session column (which disagrees for
  some folders, e.g. `test/`).
- **Duplicates removed:** byte-identical images within a camera (`batch3_*`/`b2*` copies,
  `FS_Trial31` vs `FS_L2_..._Trial31`, ...). The best-named folder's copy is kept.
- **Digit-count filter:** frames are dropped unless each paw has 0, 3 or 4 digit tips labeled
  (`ALLOWED_DIGIT_COUNTS` in the script), applied before the split so no manual-labeling effort
  goes into frames that would need digit fixes. Images of dropped frames are not copied.
- **Split:** subject-level, pooled across cameras, target 12.5% (first seed in 10-15%). The
  subject is the animal ID from the folder name (`FG`, `FB`, `EZ`, `FU`, `FS`, `FJ`, `FV1`,
  `G8S`, `U`, `G`, `AZ`), not the first `_` token, since many names have none (`FBL20324t101`).
  `sf_*` sessions are treated as `FS` (probable typo) so they can't split from it. Names with
  no recognizable ID (`unknow_Trial23`, `outliers`, `paw_camA`, `test`, ...) each count as a
  separate individual (per camera). Run the script to see the current counts and test
  subjects.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `convert_murphy_1.py` | `pose` | Merges cameras, filters keypoints, dedupes, splits, copies referenced images, writes `project.yaml` |

See [`CHANGELOG.md`](CHANGELOG.md) for label history.
