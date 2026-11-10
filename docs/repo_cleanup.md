# Repo cleanup: open items

Redundancy and structure issues still open in the repo. Guiding rule (from `AGENTS.md`): logic
lives in `mighty_mouse/`; `scripts/` are thin CLIs.

## A. Per-dataset converters (`scripts/preprocess/<dataset>/`)

Almost every converter follows one skeleton: read source CSVs, concat, subject-split, write
`CollectedData.csv` / `CollectedData_test.csv`, copy images, rewrite `project.yaml`.

- **`convert_tyssowski.py` and `convert_hantman_mv.py` are near-clones.** The differences are a
  keypoint filter (`KEEP_KEYPOINTS`), a skeleton, and the default split seed.
- **`load_subject_mapping` is identical** in the kaufman and kaufman-water converters.
- **`process_split` names two unrelated functions:** the stage-2 conversion step in
  `mighty_mouse/preprocess/convert.py` and the stage-1 face-keypoint inference step in the ibl
  converter.
- **`aind-vbn-body` reads with raw `pd.read_csv`** because its labels use a 3-column image-path
  index, which `read_labels_csv` does not support; `fix_empty_first_row` therefore does not apply
  to it.

Proposal: a small shared helper module in `mighty_mouse/preprocess/` (read, project.yaml
rewrite) so each converter keeps only its dataset-specific logic.

## B. Pseudo-labeling stack

Files: `scripts/preprocess/{active_labeling,transfer_pseudo_labels,ensemble_pseudo_labels}.py`,
`mighty_mouse/preprocess/labeling/{active_labeling,ensemble}.py` and
`scripts/evaluation/qualitative/active_round_video.py`.

- **Several variants of `model.predict_on_label_csv(...)`**:
  `transfer_pseudo_labels.predict_with_model`, `labeling.ensemble.predict`,
  `scripts/evaluation/zeroshot_predict.py` and `mighty_mouse.evaluation.run.evaluate_run`. They
  differ in how the input CSV is built. The `ensemble.predict` design (label-free input CSV under
  a unique basename, so it cannot overwrite the model's own self-eval predictions) is the safest;
  one function could serve all of them. They also differ in a setting:
  `zeroshot_predict.py` clears `training.sampling_temperature` on the loaded model before
  predicting, `evaluate_run` does not.
- **`parse_keypoints` and `verify_untouched` are each defined twice**, in
  `transfer_pseudo_labels.py` and `ensemble_pseudo_labels.py`. The two `verify_untouched` bodies
  check the same invariant differently; the transfer version is stricter and has better error
  messages.
- **Hard-coded ensemble size.** The `ensemble_pseudo_labels.py` report hard-codes
  `members_3/2/1/0` and reads columns by position (`g.iloc[:, 5]`, `g.iloc[:, 6]`). The skill
  promises 3 or 5 members, so a 5-member run would report wrong numbers.
- **Fat scripts.** `scripts/preprocess/active_labeling.py` is ~520 lines (the `Ctx` state class
  and all `cmd_*` logic); `transfer_pseudo_labels.py` has a ~145-line `main`.
- **The active-labeling directory layout is defined twice:** in `Ctx` in
  `scripts/preprocess/active_labeling.py`, and in `active_round_video.py` (`state.yaml`,
  `round_XX/`, `member{i}_*.csv`, `tolabel_manifest.csv`, `pool_scores.csv`), which also holds the
  five frame-selection modes (~190 lines in total).
- **The ensemble spread is computed twice:** `ensemble_spread` in
  `preprocess/labeling/active_labeling.py` and the "ensemble var" mode of
  `mighty_mouse/evaluation/ensemble_stats.py` are the same quantity, `sqrt(var_x + var_y)`.

## C. Train and evaluation

- **`eval_suite.py` and `zeroshot_score.py` aggregate separately:** per-keypoint and pooled median
  / mean errors, and the same "mean / median" table cells. `zeroshot_score.main` is ~115 lines
  and defines its `mm()` / `pct()` closures twice.
- **The two sweep scripts duplicate most of their ~14 `add_argument` blocks.** The docstring says
  "the CLI args are identical", but they have drifted: `--eval_only`, `--output_root` and
  `--stop_on_failure` exist only in `train_sweep.py`; `--machine` and `--poll_interval` only in
  `train_sweep_lightning.py`.

## D. Smaller items

- **Pretrain:** `output_path` is defined in both `mighty_mouse/pretrain/dedup.py` (`.npz`) and
  `mighty_mouse/pretrain/embed.py` (`.npy`) and differs only in the extension. Both belong with
  the rest of the layout in `mighty_mouse/pretrain/layout.py`.
- **Label CSVs read with raw `pd.read_csv(header=[0, 1, 2], index_col=0)`** instead of
  `mighty_mouse.labels.read_labels_csv`, so an all-empty first row is silently dropped:
  `read_preds` and the label read in `scripts/evaluation/zeroshot_score.py`.
