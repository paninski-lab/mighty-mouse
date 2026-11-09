# Repo cleanup: redundancy inventory

Findings from a read-only inspection after the stage reorganization (`pretrain` / `preprocess` /
`train` / `evaluation` mirrored across `skills/`, `scripts/`, `mighty_mouse/`, `tests/`). Nothing
here has been changed yet. Items are ordered by payoff within each section; the suggested order
of work is at the bottom.

Out of scope for now (known, deferred): `mighty_mouse/preprocess/inventory.py` and
`scripts/preprocess/corpus_summary.py` do very similar things and should be merged later.

Guiding rule (from `AGENTS.md`): logic lives in `mighty_mouse/`; `scripts/` are thin CLIs.

## A. Per-dataset converters (`scripts/preprocess/<dataset>/`, ~1,500 lines)

> **Status:** `copy_images` is now shared (tyssowski, hantman-mv and ibl use
> `mighty_mouse.preprocess.convert.copy_images`), and the six single-index converters read
> label CSVs with `read_labels_csv` (aind-vbn-body keeps `pd.read_csv` for its 3-column index).
> Still open: the near-clone tyssowski / hantman-mv skeleton, `load_subject_mapping`, the shared
> split-and-write loop, and the name clash between the two `process_split` functions.

Almost every converter follows one skeleton: read source CSVs, concat, subject-split, write
`CollectedData.csv` / `CollectedData_test.csv`, copy images, rewrite `project.yaml`.

- **`convert_tyssowski.py` and `convert_hantman_mv.py` are near-clones.** The diff is a keypoint
  filter (`KEEP_KEYPOINTS`), a skeleton, and the default split seed.
- **`copy_images` is defined four times**: tyssowski, hantman-mv, ibl, and
  `mighty_mouse/preprocess/convert.py`. The tyssowski and hantman-mv copies are identical. The
  `convert.py` version is the most capable (symlink option, session/frame path mapping).
- **`load_subject_mapping` is identical** in the kaufman and kaufman-water converters.
- **`process_split` exists in both** `mighty_mouse/preprocess/convert.py` and the ibl converter.
- **The split-and-write loop** (for train/test: select rows, `to_csv`, print counts) is repeated
  in about 8 converters.
- **Latent bug:** about 10 converters read label CSVs with raw
  `pd.read_csv(header=[0, 1, 2], index_col=0)` instead of `mighty_mouse.labels.read_labels_csv`.
  That skips `fix_empty_first_row`, so an all-empty first row is silently dropped on round trip.
  Raw `pd.read_csv(..., header=[0, 1, 2])` appears in about 15 files overall.

Proposal: a small shared helper module in `mighty_mouse/preprocess/` (read, split-and-write,
copy images, project.yaml rewrite) so each converter keeps only its dataset-specific logic.

## B. Pseudo-labeling stack

Files: `scripts/preprocess/{active_labeling,transfer_pseudo_labels,ensemble_pseudo_labels}.py`
and `mighty_mouse/preprocess/labeling/{active_labeling,ensemble}.py`.

- **Several variants of `model.predict_on_label_csv(...)`**: `transfer_pseudo_labels.predict_with_model`,
  `labeling.ensemble.predict`, `scripts/evaluation/zeroshot_predict.py`, and
  `mighty_mouse.train.sweep.evaluate_model`. They differ in how the input CSV is built. The
  `ensemble.predict` design (label-free input CSV under a unique basename, so it cannot overwrite
  the model's own self-eval predictions) is the safest; one function could serve all of them.
- **`parse_keypoints` and `verify_untouched` are each defined twice**, once in
  `transfer_pseudo_labels.py` and once in `ensemble_pseudo_labels.py`. The two `verify_untouched`
  bodies check the same invariant with different implementations; the transfer version is
  stricter and has better error messages.
- **Hard-coded ensemble size.** The `ensemble_pseudo_labels.py` report hard-codes
  `members_3/2/1/0` and reads columns by position (`g.iloc[:, 5]`, `g.iloc[:, 6]`). The skill
  promises 3 or 5 members, so a 5-member run would report wrong numbers.
- **Fat scripts.** `scripts/preprocess/active_labeling.py` is ~520 lines (the `Ctx` state class
  and all `cmd_*` logic); `transfer_pseudo_labels.py` has a ~145-line `main`. Both break the
  thin-CLI rule.

## C. Train and evaluation

> **Status:** the results-directory layout now has one owner, `mighty_mouse/train/layout.py`
> (`run_dir`, `parse_run_dir`, `recipe_run_dir`, `RECIPE`), used by `sweep.py`, `eval_suite.py`,
> `zeroshot_score.py` and, through `python -m mighty_mouse.train.layout`, `train_plan.sh`. Still open:
> the import-time `load_paths()` / `RESULTS_DIR` / `DATA_DIR` in `sweep.py` (next step), the
> duplicated aggregation in the two eval scripts, the duplicated sweep arguments, and moving
> `evaluate_model` under `evaluation/`.

- **Import-time work.** `mighty_mouse/train/sweep.py` (lines 35-38) calls `load_paths()` at import
  and defines `DATA_DIR` / `RESULTS_DIR` as module constants. This breaks the "no work at import
  time" rule, is why tests must patch `repo_root`, and forces every importer to have a
  `paths.yaml`.
- **The results-directory layout is encoded in four places.** The
  `<tag>/<losses>/tf<N>/<backbone>/seed<N>` convention lives in `sweep.make_output_dir`,
  `sweep.infer_csv_file`, `LEAF` in `scripts/evaluation/eval_suite.py:35`, and `LEAF` in
  `scripts/evaluation/zeroshot_score.py:40` (a second copy with `/seed0` appended). A third copy
  is in `scripts/train/train_plan.sh:45`. A change to one silently breaks the others.
- **`eval_suite.py` and `zeroshot_score.py` re-aggregate** per-keypoint and pooled median / mean
  separately, and each formats the same "mean / median" table cells with local `mm()` / `pct()`
  closures inside 70-110-line `main()` functions.
- **The two sweep scripts duplicate ~12 `add_argument` blocks.** The docstring says "the CLI args
  are identical", but they have drifted: `--eval_only`, `--output_root` and `--stop_on_failure`
  exist only in `train_sweep.py`; `--machine` and `--poll_interval` only in
  `train_sweep_lightning.py`.
- **`sweep.py` mixes concerns:** sweep planning, command building, and `evaluate_model` (79 lines)
  plus a standalone evaluation CLI. Evaluation belongs under `mighty_mouse/evaluation/`.
- **`sweep.make_job_name`** says in its own docstring that it is "unused locally"; only the
  Lightning script calls it.

## D. Qualitative renderers (`scripts/evaluation/qualitative/`)

- **Fat mains.** `render_compare.py` (~100 lines), `render_panel.py` (~155),
  `ensemble_variance_rows.py` (~210) and `active_round_video.py` (~140) all keep their logic in
  `main()`; it should live in `mighty_mouse/` (probably `evaluation/` or `plots/`).
- **Repeated setup** across the four renderers: `view_of`, `load_paths`, label / prediction CSV
  reading, `Writer`, and frame reading and panel drawing.
- **Duplicate helpers.** `put()` is defined in both `mighty_mouse/plots/house_style.py` and
  `render_panel.py`, with swapped argument order (the `render_panel.py` copy is a local
  re-implementation of the shared one). `render_panel.py` also has its own `val()` accessor.

## E. Smaller items

- **Pretrain:** `output_path` is defined in both `mighty_mouse/pretrain/dedup.py` (`.npz`) and
  `mighty_mouse/pretrain/embed.py` (`.npy`) and differs only in the extension. Both belong with
  the rest of the layout in `mighty_mouse/pretrain/layout.py`.
- **`select_frames`** exists in `mighty_mouse/pretrain/dedup.py` and
  `mighty_mouse/preprocess/labeling/active_labeling.py`. They are different algorithms with the
  same name; rename one rather than merge.

## Suggested order

1. **A**: converter helpers plus the `read_labels_csv` fix (largest line payoff, one real bug).
2. **C**: remove the import-time `load_paths()` and give the results layout a single owner.
3. **B**: unify the prediction call and the `parse_keypoints` / `verify_untouched` pair.
4. **D**: move renderer logic out of `scripts/` and keep the scripts thin.
5. **E**: the small pretrain and naming items, folded into whichever pass touches those files.
6. **Later**: merge `inventory.py` and `corpus_summary.py`.
