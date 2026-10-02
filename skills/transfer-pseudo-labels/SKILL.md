---
name: transfer-pseudo-labels
description: Use when the user wants to write model predictions into one raw dataset's (_raw/<dataset>/) label CSVs — filling empty cells from another dataset's model ("transfer the cheese-3d model's predictions onto cheese-2d", "pseudo-label the missing whisker pad keypoints"), from a collaborator's precomputed predictions CSVs, or iteratively bootstrapping from the dataset's own hand-corrected rows ("train on the first N rows, then overwrite the rest"), or overwriting labels with a confidence-thresholded mean of an ensemble of models. Edits that dataset's CollectedData CSVs in place; never bumps the version.
---

# Transferring pseudo-labels into a raw dataset's label CSVs

Once a standalone Lightning Pose model exists (see `skills/train-lightning-pose-model/`),
its predictions can be written into a `_raw/<dataset>/`'s already-labeled images. Three
variants have come up so far:

- **Cross-dataset gap filling** — a model trained on one dataset fills *empty* cells in
  another, where it's confident (e.g. cheese-3d → cheese-2d).
- **Precomputed predictions** — someone else already ran inference (e.g. a
  collaborator's model you don't have locally) and handed over predictions CSVs.
- **Self-bootstrapping** — the user hand-corrects the leading rows of a dataset, a model
  is trained on just those rows, and its predictions *overwrite* the remaining rows'
  labels/pseudo-labels; repeat with more corrected rows each round (see
  [below](#iterative-self-bootstrapping-loop)).

All three are separate from — and upstream of — the combined-corpus pipeline
(`preprocess-new-dataset`, `convert_dataset.py`): they edit a dataset's own
`_raw/<dataset>/CollectedData.csv`/`CollectedData_test.csv` directly, in place.

`scripts/transfer_pseudo_labels.py` does the actual work. This file covers the
decisions to confirm with the user *before* running it — none of them are recoverable
from the data alone — plus the gotchas the script's design works around.

- **Ensemble overwrite** — several models (e.g. seeds of one config) are run on the target's
  images and combined by a confidence-thresholded mean; see
  [below](#ensemble-overwrite-ensemble_pseudo_labelspy).

## Decision checklist (ask before running)

1. **Which keypoints actually transfer?** The source model's keypoint names and the
   target dataset's CSV columns are not necessarily the same set, even if both datasets
   look related. Check both `project.yaml`s (or the model's `config.yaml` `data.keypoint_names`
   vs. the target CSV's `bodyparts` header row) — don't assume a shared naming convention
   means a shared keypoint set. A keypoint the source model predicts but the target
   dataset doesn't have a column for is a **new-keypoint decision** (see
   `preprocess-new-dataset`'s equivalent question) — ask whether to add the column or
   just skip that keypoint; don't add it silently. Also expect the user to hold some
   keypoints back on purpose (e.g. ones they'll label from scratch, or ones whose
   existing labels are already good) — if their phrasing of the set is loose ("ears,
   eyes, whiskers") and doesn't obviously match the previous run's list, ask.
2. **Confidence threshold.** 0.7 has been the starting point so far, but ask — don't
   default silently.
3. **Fill empty cells only, or overwrite?** Default is fill-only. `--overwrite` replaces
   existing labels too — only appropriate when the existing labels are known to be
   worse (an earlier pseudo-label round). Before overwriting, ask which rows have been
   hand-corrected, and protect them (`--protect_rows_from`). Also ask what should happen
   below threshold (so far: keep whatever the cell had, which is what the script does).
4. **Scope: existing labeled rows only, or also new frames?** Filling gaps in images
   already present in the target CSVs is very different from also running inference on
   additional (currently unlabeled) frames and adding new rows. The script only supports
   the former (existing rows) — if the user wants new frames added too, that's a
   different, bigger task (needs a frame-selection strategy, not just a merge).
5. **Per-view/structural masking beyond the confidence threshold?** Some target cells
   are empty for structural reasons (the keypoint genuinely isn't visible in that
   camera view), not because of an annotation gap. If the source model was trained
   treating those as "occluded" (i.e. NaN in the CSV, so LP trains toward a uniform
   heatmap there rather than a peak), the confidence threshold
   alone is usually the right filter — a model trained that way should predict low
   confidence there. Ask if the user wants additional hard masking on top of that.

## Running it

```
python scripts/transfer_pseudo_labels.py \
    --model_dir <path to a trained model dir, e.g. results/cheese-3d/2026-09-26_15-45-24> \
    --target_dataset <name matching a directory under raw_dir> \
    --keypoints "kp1" "kp2" ... \
    --confidence_threshold 0.7 \
    --dry_run
```

Run with `--dry_run` first and show the user the per-keypoint / per-group inventory it
prints. On confirmation, re-run without `--dry_run` to actually write the target CSVs.

`--csvs` defaults to both `CollectedData.csv` and `CollectedData_test.csv`; pass it
explicitly to restrict to one.

If predictions were already run elsewhere, pass `--predictions_csvs <one per --csvs
entry>` instead of `--model_dir`. A `labeled-data/<target_dataset>/` prefix on their
image paths is stripped automatically, and the script exits if any target image has no
prediction. When the prediction's keypoint name differs from the target column (e.g. a
one-sided target column like `pad_top` filled from `pad_top_right`), write the
`--keypoints` entry as `pred_name=target_name`.

`--overwrite` makes confident predictions replace existing cells too; below-threshold
cells keep what they had. `--protect_rows_from <csv>` (relative to the target dataset
dir, or absolute) shields every target row whose image path appears in that label CSV,
in every `--csvs` file. The inventory reports how many filled cells overwrote an
existing label.

## Iterative self-bootstrapping loop

The workflow used for `kaufman` (2026-09-27/28 — see its `CHANGELOG.md` version 1
entry), when cross-dataset pseudo-labels were too rough to be worth correcting by hand.
Each round:

1. **User names the last hand-corrected row** (an image path). Look it up exactly in
   `CollectedData.csv` before cutting — pasted names have arrived with typos (`cam-2`
   for `cam2`) or URL-encoded (`%2F` for `/`). If there's no exact match, say which
   path you're using instead or ask. Mention if the cut falls mid-session.
2. **Write `_raw/<dataset>/CollectedData_tmp.csv`** = all rows up to and including that
   image, **all** keypoint columns (LP requires the CSV columns to match
   `keypoint_names`). Read it back and check it's value-identical to those live rows.
   Overwriting the previous round's tmp is fine — LP copies the training CSV into each
   model dir, so check that copy matches before overwriting.
3. **Config**: `poseinterface/configs/<dataset>.yaml` with `csv_file:
   CollectedData_tmp.csv` (otherwise per `train-lightning-pose-model`). Log each round
   in the config's header comment (row count, cutoff image, resulting model dir) — it's
   the only record of which model came from which rows. Give the user the `litpose
   train` command; they train.
4. **When the model is back**, before anything else: confirm `<model_dir>/<csv_file>` is
   byte-identical to the current tmp (the model trained on what you think it did), and
   that the leading rows of the live CSV still match the tmp (the user hasn't edited
   them since). Copy both live CSVs to a scratch backup.
5. **Dry run** with `--overwrite --protect_rows_from CollectedData_tmp.csv`. Expected
   counts: (total train rows − protected rows) + all test rows per keypoint, minus a
   handful below threshold. Show it, then write.
6. **Verify against the backup**, cell by cell: protected rows and non-target columns
   unchanged; every changed cell equals a prediction with likelihood >= threshold;
   every below-threshold cell equals its backup value. Also report the median pixel
   shift per keypoint vs. the previous round — it shrinks as the loop converges, and a
   keypoint that moves much more than the rest is worth pointing the user at.
7. **Final round**: once the whole train CSV is hand-corrected, the user trains on the
   full `CollectedData.csv` and only the test CSV is updated — pass `--csvs
   CollectedData_test.csv` (no protection needed) and verify the train CSV is
   byte-identical to the backup. The user then corrects the test CSV and asks for the
   version bump; the tmp CSV can be deleted.

Don't let the user edit the live CSVs between steps 4 and 6 — say so when handing over
the training command.

Expect the confidence threshold to filter almost nothing here: likelihoods sat around
0.999 for kaufman, even on test sessions the model never trained on (frames there vary
little between sessions and animals). That's expected, not a bug — accuracy is judged
by the user's manual pass, not by the threshold.

## Ensemble overwrite (`ensemble_pseudo_labels.py`)

Used for cheese-3d (2026-10-01, version 2): three `cheese-2d` models (same config, different
`rng_seed_data_pt`) pseudo-label `cheese-3d`. Unlike `transfer_pseudo_labels.py`, which takes
one model and only keeps cells that clear the threshold, this one **always replaces** the
requested columns: per cell, members with likelihood >= the threshold (0.9 there) survive,
the new label is their mean (x, y), and it's **blank if none survive** — so it can erase
existing labels. Columns not in `--keypoints` are never touched.

```
python scripts/ensemble_pseudo_labels.py \
    --model_dirs <run0> <run1> <run2> --target_dataset <name> \
    --out_dir results/<name>/ensemble-pseudo-labels \
    --keypoints "pred_name=target_name" ... --dry_run
```

- `--keypoints` is `pred_name=target_name` per entry because the models' names (canonical
  `_left`/`_right`) usually differ from the target's raw column names. Build the list from
  `configs/datasets/<target>.yaml` (target raw name -> canonical name), minus any columns that
  should be left alone (cheese-3d: the pupils).
- A keypoint the models predict but the target lacks (cheese-3d: wrists) has to exist as a
  column first — `skills/add-keypoints-to-raw-dataset`.
- `--out_dir` caches each member's predictions (reused on re-run, so the dry run's inference
  isn't repeated for the real write) and gets `label_change_report.csv`: per keypoint and CSV,
  cells that lost / gained a label and how many members (0-3) cleared the threshold. Share it;
  the changelog entry should summarize it.
- Checklist items 1-5 above still apply (keypoint overlap, threshold, scope), plus: confirm that
  blanking below-threshold cells is intended. Back up the live CSVs first and verify cell by
  cell afterwards (non-edited columns unchanged; edited cells equal the recomputed mean).
- Same as the single-model script: never bumps the version.

## What the script guarantees

- Only writes `(x, y)` for the requested `--keypoints`, only where prediction
  likelihood >= `--confidence_threshold`, only where the target cell was already empty
  (unless `--overwrite`), and never in a row listed in `--protect_rows_from`.
- Never adds rows or keypoint columns, and never changes any cell outside that editable
  set — it asserts this internally (`verify_untouched`) and raises rather than writing
  if it finds a violation.
- Never calls `bump_version.py`. See `skills/bump-dataset-version/`. Versioning
  the result is a separate, user-requested step, after they've verified the fill by hand
  (e.g. in the LP labeling app) — don't run `bump_version.py`, not even `--dry-run`,
  until asked. Report the fill counts and stop there.

## Gotchas this script works around

- **`litpose predict` (the CLI) can't do this** — its CSV-input path always uses the
  model's own training `data_dir`, with no override. The script uses the
  `lightning_pose.api.Model.predict_on_label_csv(..., data_dir=...)` Python API directly
  instead, which does support an explicit `data_dir` for the target dataset's images.
- **Output-path collision:** `predict_on_label_csv` keys its output directory purely off
  the input CSV's basename (`<model_dir>/image_preds/<csv_basename>/predictions.csv`).
  Feeding it a file literally named `CollectedData.csv` would silently overwrite the
  model's own self-eval predictions from training (which live at that exact same path).
  The script works around this by copying the target CSV to a temp file named
  `<target_dataset>__<original_name>` before calling predict — always distinct from
  anything the model has predicted on before.
- **Verify after every write, not just the first attempt.** The script's internal check
  compares against the CSV it read moments earlier; an independent cell-by-cell diff
  against a backup taken before the run is what catches everything else. If a run needs
  to be re-scoped (different keypoints, different threshold) after already writing
  once, diff against a known-clean original copy before trusting the counts — this has
  gone wrong silently mid-session before. Relatedly: never revert or overwrite a
  dataset's live CSVs just because their state is surprising — someone may be editing
  them concurrently in the labeling app. Ask first.
