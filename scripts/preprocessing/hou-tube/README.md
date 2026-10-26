# hou-tube preprocessing

Builds `_raw/hou-tube` (standard DLC layout) from `_raw/_dlc/hou-tube`.

**Status: experimental, not in the corpus.** `_raw/hou-tube` and a draft
`configs/datasets/hou-tube.yaml` exist and `convert_dataset.py --dataset hou-tube` validates,
but the config is deliberately **gitignored** and `hou-tube` is not in `ALL_DATASETS` while we
decide whether it belongs in the final corpus. To promote it: track the config (remove it from
`.gitignore`) and add `hou-tube` to `ALL_DATASETS` together.

## Why a custom converter

The source is one directory per session-and-view (`<recording>_record_<view>`, views
L/R/TL/TR/BC/TC), each holding labeled PNGs plus an `annotations.yaml`
(`keypoint -> image -> [[x, y]]`, unlabeled = `.nan` or `null`). No CSV, `project.yaml`
or train/test split ships with it, so `convert_hou_tube.py` builds all three.

- **Views merged** into one project (same rig as `cheese-3d`); source directory names are
  used verbatim as session names.
- **Only annotated session-views are converted.** Some recording directories contain
  images but no `annotations.yaml` and are skipped.
- **Split:** session-level, one whole recording held out (`TEST_RECORDINGS` in the script).
- **`ref(head-post)`** (never labeled, not a body part) is excluded in the dataset config;
  every other keypoint maps to an existing canonical name, so the vocabulary is unchanged.
- **Sides:** keypoints are explicitly `(left)`/`(right)` in the source; the session side map
  in the config mirrors `cheese-3d`. No `POST_PROCESS` entry.
