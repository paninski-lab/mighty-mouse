---
name: train-lightning-pose-model
description: Use when the user wants to train a standalone Lightning Pose model on a single _raw/<name>/ dataset — e.g. "train a model on cheese-3d", "let's bootstrap a pseudo-labeling model for X". Not for adding a dataset to the mighty-mouse combined corpus (that's preprocess-new-dataset / the main README's stage 2-3), and not for training on a composite/merged dataset (that uses configs/model.yaml's own defaults, not this skill's fixed defaults).
---

# Training a standalone Lightning Pose model

This covers training a Lightning Pose model directly on **one** `_raw/<name>/`
dataset — e.g. to bootstrap a pseudo-labeling model — as opposed to the combined-corpus
pipeline in `mighty-mouse/scripts/{preprocess/convert_dataset,preprocess/build_dataset,train/train_sweep}.py`. Don't
conflate the two: this workflow never touches `configs/keypoints.yaml`, `model.yaml`,
or `ALL_DATASETS`, and its outputs don't live under `mighty-mouse/` at all.

- Configs live at `poseinterface/configs/<name>.yaml`
- Results live at `poseinterface/results/<name>/<date-time>/`
- `poseinterface/` is **not** a git repo, so neither of the above is version
  controlled — that's expected, not a gap to fix.

## Writing the config

Start from the `lightning-pose` repo's own template,
`<lightning-pose>/scripts/configs/config_default.yaml` (path varies by machine/checkout — ask if
unsure), and fill in the `data:` section for the target dataset:

- `data_dir`: absolute path to `_raw/<name>/`
- `video_dir`: absolute path to `_raw/<name>/videos/`
- `csv_file`: `CollectedData.csv` (the **train** split only — the held-out
  `CollectedData_test.csv` is evaluated separately, it's not part of this config)
- `num_keypoints` / `keypoint_names`: read from `_raw/<name>/project.yaml` (or the
  CSV's `bodyparts` header row directly — verify the two agree). Use the dataset's
  **own** raw keypoint names here, not the canonical `mighty-mouse/configs/keypoints.yaml`
  vocabulary — this config points directly at `_raw/<name>/`, it doesn't go through
  `convert_dataset.py`'s renaming.

## Fixed defaults for this project

**Scope: single-dataset configs only.** These apply to a standalone config trained
on one `_raw/<name>/` dataset, as described in this file. A model trained on a
*composite/combined* dataset (multiple datasets merged via `build_dataset.py`) is a
different workflow entirely and should follow
[`mighty-mouse/configs/model.yaml`](../../../configs/model.yaml)'s own defaults instead
(currently `resnet50_animal_ap10k`, `Adam` @ `1e-3`, step-based schedule, etc.) — do
not apply the values below to that pipeline, and don't edit `model.yaml` to match
this list.

Every standalone config in this project uses these values — don't re-derive or ask
about them per-dataset, just set them:

- `data.image_resize_dims`: `height: 256, width: 256`
- `training.imgaug_hflip`: `false`
- `data.mirrored_column_matches`: `null`
- `model.backbone`: `vits_dinov2`
- `training.optimizer`: `AdamW`
- `training.optimizer_params.learning_rate`: `5e-5`
- `eval.predict_vids_after_training`: `false`

Everything else in `config_default.yaml` (batch sizes, epochs, losses, etc.) stays at
its default unless there's a specific reason to change it for that dataset.

## Gotchas

- **`image_resize_dims`** must each be a multiple of 128 (128/256/384/512). Fixed at
  256x256 per above regardless of the dataset's native aspect ratio — this will
  distort non-square frames somewhat (e.g. cheese-3d's 640x512 raw frames); revisit
  only if keypoint precision on fine features (whiskers, pupils) turns out to suffer.
- **`imgaug_hflip`** only works if left/right keypoint pairs are named with a literal
  `_left`/`_right` suffix — a dataset using `(left)`/`(right)` (like cheese-3d) will
  not pair correctly even if this were turned on. It's fixed to `false` here anyway,
  but don't flip it on without renaming keypoints first.
- **`mirrored_column_matches`** only applies when a single CSV row encodes multiple
  camera views of the same instant as separate columns (true mirrored-rig setups).
  Datasets where each row is a single frame from a single view — with the view
  encoded in the session/path name instead (e.g. cheese-3d's `_L`/`_R`/`_TL`/`_TR`/
  `_BC`/`_TC` suffixed session directories) — are not this case; leave it `null`.
- **Output directory**: use `litpose train`'s `--output_dir` flag rather than editing
  the config's `hydra.run.dir` — `--output_dir` takes precedence, so there's nothing
  to change in the config itself.

## Training

Before training, check programmatically (not by eye) that `num_keypoints ==
len(keypoint_names)` and that `keypoint_names` order matches the CSV's column order.
Then show the user the config and **stop for their explicit sign-off** — this is a
standing rule for this workflow, not a one-off ask.

```bash
litpose train poseinterface/configs/<name>.yaml \
  --output_dir poseinterface/results/<name>/$(date +%Y-%m-%d_%H-%M-%S)
```

The `$(date ...)` is evaluated by the shell at invocation time, giving one
`results/<name>/<date>_<time>/` folder per run.
