---
name: active-pseudo-labeling
description: Use when the user wants to label new keypoints in a raw dataset (_raw/<dataset>/) by iterating label -> train a model ensemble (3 or 5 models) -> add the frames the ensemble disagrees on most (prefilled with the ensemble mean) -> repeat, e.g. "label the new ear keypoints in ibl with active pseudo-labeling". Never bumps the version.
---

# Active pseudo-labeling of new keypoints

Why: bootstrapping from the first N hand-labeled rows trains on a few sessions, then asks the
model to generalize to many unseen ones. Here the first batch is drawn from **every** session,
and later batches are the frames where an ensemble disagrees most.

`scripts/preprocess/active_labeling.py` (logic in `mighty_mouse/preprocess/labeling/`) does the bookkeeping; its
docstring lists the subcommands. Run it in the Lightning Pose env (`score` needs it). Files,
per dataset:

- `_raw/<dataset>/CollectedData_tolabel.csv` — rows being labeled in the LP app; **grows**
  each round, all columns (existing labels carried over).
- `_raw/<dataset>/CollectedData_tmp.csv` — the round's **training CSV** (copy of tolabel);
  the model config's `csv_file` points here.
- `poseinterface/results/<dataset>/active/` (sibling of `raw_dir`, *not* `paths.yaml`'s
  `results_dir`) — `state.yaml`, `tolabel_manifest.csv` (source CSV, round added, prefilled),
  `backups/` (never overwritten), `round_XX/` (cached member predictions, `stats.csv`,
  `pool_scores.csv`).

Layout assumed: image paths are `labeled-data/<session>/<frame>.png`, a **session is one
`labeled-data/` subdir** (so e.g. a left and a right-flipped view of one video are two
sessions), and train + test CSVs are pooled (the split is ignored until `finalize`).

## Rules that hold throughout

- **Every row in tolabel is treated as reviewed at training time**, and any empty cell is
  intentional (occluded) — kept as is. So the user says "round done" only after finishing the
  whole batch.
- **Confirm the user has left the project** before `init`, `select`, `reorder`, `sync`,
  `finalize` (they rewrite files the app edits, and the app's next save would clobber them). The
  app doesn't need to be closed: the user just returns to its home screen listing all projects.
  Ask; don't infer. If the user labels a local copy and uploads it, check it against the copy
  on the server (rows, columns, which cells changed) before it replaces anything.
- **tolabel follows the dataset's current labels everywhere except the keypoints being labeled.**
  Whenever `_raw/<dataset>/CollectedData*.csv` change (e.g. a new label version of the dataset),
  run `sync <dataset> --dry_run`, then `sync` (backup + a log of every changed
  cell in `results/<dataset>/active/sync_*_changes.csv`). `train-csv` refuses while they differ;
  `select` syncs the existing rows itself. Sync overwrites the user's in-app edits to other
  keypoints too: report them (they are in the log) so corrections can go to the dataset's owner.
  The dataset's CSVs need not carry the keypoints being labeled; every subcommand
  adds them as empty columns, and `finalize` appends them.
- Original `CollectedData*.csv` are only written by `finalize`, after a backup. No version bump.
- Never touch `CollectedData_tmp.csv` or tolabel while models are training.

## Ask first (every time)

- New keypoints: raw names and order → `add-keypoints-to-raw-dataset` first (steps up to
  adding the columns; the `configs/datasets/<dataset>.yaml` mapping waits until labeling is done).
- **n**, frames per session for the first batch, and the seed. Offer the totals for a few
  n (sessions with fewer than n frames contribute all of theirs).
- Whether the keypoints already hold rough labels to correct (`init --allow_existing`).
- **m** per round, after seeing the stats; eligibility and prefill thresholds (default 0.9).
- Ensemble size: **3 models (seeds 0-2) or 5 (seeds 0-4)**. With 3, one member with a bad (e.g.
  early) best checkpoint dominates the spread; 5 makes the ranking and a median prefill robust to
  one bad member. Spreads from 3 and 5 members are not exactly comparable, so keep one size across
  the rounds of a labeling campaign.
- Train the models themselves, or have Claude run them? (Users usually prefer to; they may
  change the seed by hand.) On a cluster, one job with one seed per GPU waits in the queue once
  instead of once per seed.

## The round, step by step

1. **`init <dataset> --keypoints ... --n N --seed S`** (try `--dry_run`). The user labels in the
   app. With `--allow_existing` the keypoints' current values are carried over as the starting
   labels; without it `init` refuses non-empty keypoints.
2. **`train-csv <dataset>`** — run it after *every* labeling round, once the user says the batch
   is done and *before* they train. Writing it is not optional: training reads
   `CollectedData_tmp.csv`, which does **not** update when `select` appends rows, so models
   trained without it silently miss the whole new batch. Then:
   - First round: write `configs/<dataset>.yaml` per `train-lightning-pose-model` (copy a
     sibling standalone config, set `csv_file: CollectedData_tmp.csv`, `model_name`, keypoints
     from `project.yaml`, verified against the CSV header) and get the user's sign-off.
   - Every round: log it in the config's header comment (rows, resulting model dirs).
   - Give the commands, one per seed (only the seed and output dir differ; tag the dir with the round
     so rounds are easy to tell apart):
     ```
     for s in 0 1 2; do   # 0 1 2 3 4 for a 5-member ensemble
       litpose train configs/<dataset>.yaml \
         --output_dir results/<dataset>/$(date +%Y-%m-%d_%H-%M-%S)_round<R>_seed-$s \
         --overrides training.rng_seed_data_pt=$s
     done
     ```
3. **`score <dataset> --model_dirs a b c`** (every member of the newest round; confirm each is
   `COMPLETED` with its seed). It refuses unless every model dir's training CSV is byte-identical to the
   current `CollectedData_tmp.csv` *and* that CSV has the same rows as tolabel (so a skipped
   `train-csv` is caught); it warns if labels were edited after `train-csv`. Output is noisy
   (checkpoint-loading warnings): redirect to a log and print the report. Predictions are
   cached per round, so a re-run is cheap. Report the table:
   - spread = sqrt(var_x + var_y) over all members; a keypoint counts in a frame only if some
     member has likelihood > 0.9 (so occluded points with low-confidence scatter are ignored);
     frame score = **mean** over eligible keypoints;
   - `baseline` = mean spread on labeled frames. It is in-sample (every labeled frame was
     trained on), so optimistic, and a *mean*, so a few outlier frames can move it a lot — look
     at the median in-sample spread before reading a jump as poor fitting;
   - pool frames are binned by spread / baseline (`x0-1`, `x1-2`, `x2-4`, `x4+`), which move
     whenever the baseline does; the `n>3px` / `n>10px` / `n>20px` counts don't. `pool_ineligible`
     frames (no confident member) are never candidates.
   - To *see* the round (house style, from the cached predictions, no model re-run):
     `python scripts/evaluation/qualitative/active_round_video.py <dataset> --frames pool` (frames not trained
     on, spread over sessions) or `--frames top` (the highest-disagreement frames `select` would
     add). Raw keypoint names get their house colours from `--keypoint_map` (raw -> canonical).
4. **Is the loop converging?** Compare rounds with the *full-pool* frame-score quantiles
   (median / 90th / 99th, printed by `score`), which are comparable across rounds. Don't compare
   counts on "the same remaining pool frames": last round's selection removed that pool's top
   frames, so those counts are capped by construction. Typical pattern: the median and 90th
   percentile tighten quickly while a tail of ~1% of frames keeps rotating between rounds; a
   shrinking 99th percentile / ">20px" count is the sign the tail is finally being learned.
   Keypoints that are mostly occluded in the labels stay ineligible for most frames and take
   more rounds.
5. **`select <dataset> --m M --dry_run`**, then without `--dry_run`. Before asking to write,
   report: how many pool frames, **the number of sessions the batch spans** (and the pool's
   session count, frames per session), train/test split, score range, prefilled cells per
   keypoint. After writing, give the user the **first new row's image name and data row**
   (printed by `select`) so they can find the batch in the app. New rows get the ensemble-mean
   prefill (members >= 0.9) for the user to correct.
   - The batch is **organized by session, not by variance rank**: a session's frames are
     contiguous and in frame order, because labeling is more consistent when a video's frames
     come back to back. The variance ranking only decides *which* frames.
   - **Reserved frames** (e.g. another labeler is labeling them and they'll be added to tolabel
     later): `select ... --exclude _raw/<dataset>/frame_ids_active/<list>.txt` (one image path
     per line). The list is recorded in `state.yaml` (`exclude_files`) and re-read by every later
     `select`, `--m all` included, so it never needs repeating; `select` refuses if a recorded list
     goes missing. Report how many pool frames it excluded.
   - `select` also writes `_raw/<dataset>/frame_ids_active/tolabel_round<R>.txt` (images added
     in each round) and `tolabel_all.txt`; `frame-ids <dataset>` rewrites them from the manifest.
   - **More frames for the batch already selected** (no retraining since, e.g. the user wants 300
     instead of 150): `select <dataset> --m M --extend` uses the same scores, skips the frames already
     in tolabel, tags the new rows with the current round and backs up as `before_round<R>_extend`.
   - Then back to step 2 (`train-csv` after the user reviews the batch).
6. **`reorder <dataset>`** (optional, `--dry_run` first): re-sorts *all* rounds' rows by session
   so the user can check consistency across rounds. Safe to run any time the user is back at the app's home screen;
   `select` appends at the end, so rerun after it if wanted. Row order doesn't affect training
   or `score`.
7. **Stopping.** When the user decides the remaining pool is better fixed by hand:
   `select --m all` adds every remaining pool frame (prefilled); after they finish,
   `finalize <dataset> --dry_run`, then for real. It refuses unless tolabel covers every
   original row and no non-target column differs, and it keeps the originals' row order.
   - The prefill is the mean over confident members by default. Ask whether the final fill
     should use the median instead (`select --m all --prefill_agg median`), which is more
     robust to one bad member.
   - If the user will do the last review directly in the final CSVs, `select --m all`
     followed straight by `finalize` is fine: every pool row then counts as reviewed. Say
     plainly that the unreviewed rows hold ensemble values (the >10 px tail is least
     reliable) and leave the frames with no confident member blank.
8. Map the new keypoints in the dataset config / `POST_PROCESS` (`add-keypoints-to-raw-dataset`
   steps 3–5). Version bump only when asked (`bump-dataset-version`).

## Known limitations / selection choices to raise

- **Frame score is a mean over eligible keypoints, so it dilutes single-keypoint
  disagreements.** A frame with one keypoint 25 px apart and three at 0.4 px scores ~6.6 and
  can miss the top m. In the ibl ears run this starved a mostly-occluded keypoint (e.g. 179
  pool frames had it >10 px, only 18 were in the top 150). Alternatives not implemented:
  max over keypoints, or a per-keypoint quota (top k per keypoint, deduplicated). The default is
  the mean; if one keypoint's `n>10px` count dwarfs its share of the selected batch, tell the
  user and offer the alternatives rather than silently changing it.
- Few members: spread is a noisy estimate (worst with 3), and different frames can top the
  ranking after each retrain. Each member's "best" checkpoint is picked on a ~5 % val split, so
  check the chosen epochs after training: a member stopping very early inflates the spread.
- Baseline is in-sample; use trends, not absolute values.
