---
name: build-corpus-version
description: Use when the user wants to build a new version of the full combined corpus — e.g. "build corpus v12", "rebuild the corpus now that the labels changed", "freeze this build". Converts every dataset, builds the merged tags, freezes data/results as head-fixed_vN, and records which dataset versions went in (assets/corpus_history.md).
---

# Building a new corpus version

A corpus version is one frozen `data/head-fixed_vN` (+ matching `results/head-fixed_vN`) built from
every dataset in `ALL_DATASETS`. `assets/corpus_history.md` records, per corpus version, which label
version of each dataset went in — it is the record of how the corpus changed; the *why* lives in the
per-dataset `CHANGELOG.md` files. (`assets/corpus.{md,csv}` is different: it describes the corpus
as it is now.)

This is about the whole corpus. Versioning one dataset's labels is
[`bump-dataset-version`](../bump-dataset-version/SKILL.md); converting a new dataset is
[`preprocess-new-dataset`](../preprocess-new-dataset/SKILL.md).

## Steps

1. **Pick N** = the last column of `assets/corpus_history.md` + 1 (not the highest `head-fixed_vN`
   on disk: older local directories can use a different numbering).
2. **Check the raw datasets are in order.** Each `_raw/<dataset>/VERSION.txt` should be the version
   you intend to build with, and its live CSVs should match that `versions/` snapshot. Only read
   this; don't run `bump_version.py` without the user's go-ahead (they verify label edits first).
3. **Convert every dataset** into a fresh directory, so nothing training from the current one is
   disturbed (`<data>` = the parent of `data_dir` in `paths.yaml`):
   ```
   python scripts/preprocess/convert_dataset.py --dataset <name> --data_dir <data>/head-fixed_vN
   ```
   once per dataset in `ALL_DATASETS` (`--link_frames` if frames are unchanged).
4. **Build the merged tags** into the same `--data_dir` with `build_dataset.py` (all-data tag and
   whatever combinations the user wants; see `docs/build_dataset.md`). The build that covers every
   dataset in `ALL_DATASETS` records the column automatically, because the directory is named `_vN`.
5. **Results directory:** create `results/head-fixed_vN` with the same N.
6. **Check** `assets/corpus_history.md`: the new column's versions should equal each dataset's
   `VERSION.txt`, and datasets that did not change should repeat the previous column.
   If you built in an unversioned directory (`data/head-fixed`) and renamed it afterwards, nothing
   was recorded: run `python -m mighty_mouse.preprocess.history --data_dir <data>/head-fixed_vN`.
   **Don't skip this step** — it is the one that is easy to forget.

Rebuilding a version in place overwrites its column.
