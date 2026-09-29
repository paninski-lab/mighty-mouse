---
name: add-keypoints-to-raw-dataset
description: Use when the user wants to add new (empty) keypoints to an existing raw dataset in _raw/<dataset>/ so they can label them — e.g. "add nose_top and the whisker pad keypoints to hantman-mv", "I want to label ears in facemap too". Adds the columns to CollectedData*.csv + project.yaml via scripts/add_keypoints.py, then walks through the config mapping and visibility decisions. Not for onboarding a new dataset (preprocess-new-dataset) or filling values from a model (transfer-pseudo-labels).
---

# Adding new keypoints to a raw dataset

A recurring edit: a dataset already in `_raw/<dataset>/` gets new keypoint columns that
the user then labels by hand in the LP app (or later fills with
`transfer-pseudo-labels`). Precedents, all done by hand before this skill existed:
`hantman-mv` eye/nose (2026-09-15) and ears (2026-09-17), `facemap` ears and pupil, `ibl`
nose/pupil/tongue — see each dataset's `scripts/preprocessing/<dataset>/CHANGELOG.md`.

Touches up to four places, which don't cross-check each other:

1. `_raw/<dataset>/CollectedData.csv`, `CollectedData_test.csv`, `project.yaml` — the
   script does this part
2. `configs/datasets/<dataset>.yaml` — the `keypoints` mapping to canonical names
3. `mighty_mouse/convert.py` — the dataset's `POST_PROCESS` hook, if it has one
4. `configs/keypoints.yaml` + `configs/model.yaml` — only if a canonical name is new

## Before touching anything

**Confirm the dataset isn't open in the labeling app.** The user may be editing the
live CSVs concurrently, and the app's next save would silently drop the new columns
(or the script would overwrite the user's unsaved-then-saved labels). Ask; don't infer
it from file timestamps.

## Decision checklist (ask; none of these are recoverable from the data)

1. **Canonical names.** Check each intended canonical name against
   `configs/keypoints.yaml`. If one is missing, adding it to the shared vocabulary
   (`keypoints.yaml` *and* `model.yaml`, same position in both — `convert_dataset.py`
   refuses to run if they disagree) is a corpus-level change: get an explicit go-ahead
   for that separately (see AGENTS.md).

2. **Raw column names — one lateralized column, or explicit left/right columns?** Look
   at the dataset config's `sessions` sides. A `{side}` mapping
   (`pad_top: pad_top_{side}`) puts a row's coordinates only on the side its *session*
   is assigned; the other side gets NaN. That's right when each frame shows one side of
   the animal. If the user will label **both** sides in the same frame (e.g. a front
   view), a `{side}` mapping can't represent it — use explicit raw columns
   (`pad_top_left`, `pad_top_right`) mapped one-to-one. Midline keypoints (e.g.
   `nose_top`) are always a single plain column. Otherwise, match the dataset's existing
   raw naming (side-agnostic raw names like `ear_top`, lateralized in the config).

3. **Visibility for cells that stay empty.** `convert.py` gives an empty cell of a
   mapped keypoint `visible=1` (in dataset, occluded → trained toward a uniform heatmap).
   Check `POST_PROCESS` in `mighty_mouse/convert.py` for this dataset: e.g. `hantman-mv`
   forces every unlabeled `_left` column to `visible=0` (no opinion) except an exempt
   list, and `kaufman` does so for a fixed set of keypoints. Ask for each new keypoint
   which is true when it's unlabeled: *really hidden* (keep 1 — may need adding to an
   exemption list) or *visible but never annotated* (force 0 — may need adding to the
   suppress list). Don't extend a blanket rule by default; the hantman-mv ear exemption
   exists because the blanket rule was wrong for them.

4. **Will every row get labeled?** Until a row is labeled, a mapped keypoint trains as
   occluded there. If labeling will be partial or spread over time, hold off on step 3
   below (the config mapping) until the user says labeling is done, so a conversion run
   in between doesn't pick up half-labeled columns.

## Steps

1. **Add the empty columns** (after the app check above):
   ```
   python scripts/add_keypoints.py <dataset> --keypoints <raw names...> --dry-run
   python scripts/add_keypoints.py <dataset> --keypoints <raw names...>
   ```
   Appends empty x/y columns to both CSVs and names to `project.yaml`'s
   `keypoint_names`, in the given order. It edits the CSVs as text so every existing
   byte (CRLF line endings, 17-digit floats) is preserved, verifies that before
   writing, and refuses if a keypoint already exists. It never touches configs, the
   changelog, or the version.

2. **The user labels them** in the LP app. Wait for them to say they're done.

3. **Map them** in `configs/datasets/<dataset>.yaml` under `keypoints` (per decision 2),
   and update `POST_PROCESS` in `mighty_mouse/convert.py` per decision 3 — keep the
   comment above the hook in sync with what it now does, and extend the hook's cases in
   `tests/test_convert.py`; run `python -m pytest`.

4. **Validate the conversion** without writing to the shared data dir:
   ```
   python scripts/convert_dataset.py --dataset <dataset> --link_frames --data_dir <scratch dir>
   ```
   and spot-check the new keypoints' `visible` values in the output CSVs against
   decision 3.

5. **Record it.** Don't write a separate changelog entry — the next version bump
   (`bump-dataset-version` skill) inserts one. Draft the bullets for that bump's
   message instead, matching the dataset's existing entries: which keypoints were
   added and labeled, the config mapping, and any `POST_PROCESS` change with its
   reasoning. **Only bump when the user asks** — they verify the labels first.
