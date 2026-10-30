---
name: train-plan
description: Run after a corpus version is built, or when the user asks what to train. Shows which Mighty Mouse recipe models exist or are missing for the current corpus (all-data trunk, dedicated per-dataset models, leave-one-out trunks), asks which stages and seeds to run, and launches them with scripts/train_plan.sh.
---

# Training plan

**Requires lightning-pose on branch `post_sub_lp_delta`** (`paninski-lab/lightning-pose`): the
recipe's sampling temperature, per-dataset zoom, nonlinear head and resumable runs exist only there.
Check before training with `git -C <lightning-pose clone> branch --show-current`;
`scripts/train_sweep.py` also stops with a message if the installed lightning-pose lacks them. (Once
the branch is merged into lightning-pose `main`, drop this section.)

**Never launch training on your own.** `plan`, `dry` and `show` are always safe and are what this
skill runs by default. `run` starts GPU training and is only allowed when the user has named the
stages (and seeds) to run; if in doubt, show the table and ask.

`scripts/train_plan.sh` encodes the recipe (header of the script): ViT-S DINOv3,
`configs/model_zoominout.yaml`, sampling temperature 2, nonlinear head (256 channels), a rolling
checkpoint every 500 steps, best checkpoint kept. Trunks (all-data and leave-one-out) train 14k steps
at 384 x 384 (backbone unfrozen at 3k, LR halved at 6k / 8k / 10k); dedicated models keep the
config's 12k steps at 256 x 256. Results: `<results_dir>/trunks/<tag>_train/...` and
`<results_dir>/dedicated/<dataset>_train/...`. It never retrains a finished run (`--skip_existing`),
so it can be re-run at any time.

1. `bash scripts/train_plan.sh plan` and show the table. Rows: `all` (the all-data trunk),
   `dedicated` (one per dataset), `loo` (one leave-one-out trunk per dataset; `loo:<ds>` /
   `dedicated:<ds>` for one). Status per seed: `done` = evaluated; `resumable` = has a rolling
   checkpoint and continues on the next `run`; `trained` = trained but not evaluated (the next `run`
   only evaluates it); `PARTIAL` = crashed without a checkpoint (delete the directory to retrain);
   `missing`; `NO CSV` = build the merged tag first
   (`python scripts/build_dataset.py --tag <tag> --datasets ... --n_frames -1`).
2. Ask which stages and how many seeds, and wait for the answer. Suggested order: `all` first (what
   downstream experiments use), then `dedicated` (per-dataset references), then `loo` (needed for
   zero-shot / transfer experiments). One seed first.
3. `bash scripts/train_plan.sh show <stages> [--seeds "0;1"]` prints the script `run` would launch;
   check it, then `bash scripts/train_plan.sh run <stages> [--seeds ...] [--parallel N]`. It runs
   detached (`setsid nohup`) and keeps at most N training runs on the machine at once (default 1),
   counting every `scripts/train_sweep.py` process. After every `loo` run it writes zero-shot
   predictions on the left-out dataset's train frames (`scripts/zeroshot_predict.py` ->
   `<run>/zeroshot/<ds>_train_predictions.csv`; test frames are in `<run>/eval/<ds>/`).
   `bash scripts/train_plan.sh status` tails the log (`<results_dir>/_logs/train_plan_*.log`).
   Queued runs read `paths.yaml` when they start: do not repoint it while a plan is running.
4. When done, rerun `plan` to confirm every row is `done`, check the first run's log for
   lightning-pose's `nonlinear head: ConvT -> 256 -> ...` line, and evaluate with the `eval-suite`
   skill.

Adding a dataset: add it to `ALL_DATASETS` and give it a short name at the end of `SHORT_NAMES`
(`mighty_mouse/datasets.py`); its tags and rows follow.
