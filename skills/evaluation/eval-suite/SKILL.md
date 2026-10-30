---
name: eval-suite
description: Standard evaluation of the Mighty Mouse recipe's trained models for a corpus version (pixel-error tables on every dataset's test set, leave-one-out zero-shot scores, transfer videos) and the convention for every evaluation deliverable. Run after training finishes or when the user asks to evaluate or compare models. Never trains anything.
---

# Evaluation suite

**lightning-pose:** the tables below read saved predictions and need no lightning-pose. Anything
that loads one of the recipe's models (new predictions, `scripts/evaluation/zeroshot_predict.py`, videos that
need inference) needs lightning-pose on branch `post_sub_lp_delta` (`paninski-lab/lightning-pose`),
as their nonlinear head does not load in stock lightning-pose. Check with
`git -C <lightning-pose clone> branch --show-current`. (Drop this note once the branch is merged
into lightning-pose `main`.)

Every evaluation, standard or ad hoc, goes in its own folder
`<results_dir>/qualitative/<topic>-<MM-DD>/` with a `README.md` whose YAML front matter has
`title, date, data_version, question, models, outputs, finding`, plus the script or command that
produced it. `python scripts/evaluation/qualitative/qualitative_index.py` rebuilds
`<results_dir>/qualitative/INDEX.md` from those READMEs; run it after every delivery. Run folders
are read-only inputs: nothing is written into them. Videos follow the `pose-video` skill.

Scoring convention (`mighty_mouse/evaluation/comparison.py`): only labeled cells count (`visible == 2`), and
the tongue and lip keypoints are left out of every pixel-error number (they are still trained and
drawn). Judge by the median; the mean flags large misses.

## The standard battery (once per corpus version, after `train-plan` reports `done`)

Ask which parts to run; suggest all of them, in order.

1. **Tables:** `python scripts/evaluation/eval_suite.py` - pooled and per-keypoint pixel error of every
   finished trunk and dedicated model on every dataset's test set (`summary.csv`,
   `per_keypoint.csv`, a README table, and per dataset the all-data trunk vs its dedicated model
   per keypoint). Fill in `finding:` with one sentence.
2. **Previous-version comparison:** for datasets whose test set did not change between corpus
   versions, add the previous version's runs (`--run prev-all=<run dir>`) to show old vs new side
   by side, and say clearly when a comparison is not valid (changed test labels or vocabulary).
   When only a dataset's labels changed but not its frames, re-score the old run's saved
   `eval/<ds>/predictions.csv` against the new labels (same image index) instead of re-running
   inference. Report newly added keypoints separately.
3. **Leave-one-out zero-shot:** `python scripts/evaluation/zeroshot_score.py` - each leave-X-out trunk on X's
   labeled keypoints, on X's test frames (`<run>/eval/X/`) and train frames
   (`<run>/zeroshot/X_train_predictions.csv`, written by `train_plan.sh` after each loo run), next
   to the all-data trunk and the dedicated X model: per keypoint mean / median px, detection
   (likelihood >= 0.7) and confident-but-wrong (> 25 px). Keypoints no other dataset labels are
   marked untrained; the pooled TRAINED row is the fair zero-shot number.
4. **Transfer videos:** with `pose-video`, `scripts/evaluation/qualitative/render_panel.py --mode all`
   (and `--mode transfer` when transfer is the question) of the all-data trunk on every dataset's
   test frames.

## Ad-hoc investigations

Same folder shape and README: state the question first, the models and corpus version used, and end
with the finding. Prefer the saved `eval/<dataset>/predictions.csv` over new inference; when new
inference is needed, say which checkpoint and check that it reproduces the saved predictions before
drawing conclusions.
