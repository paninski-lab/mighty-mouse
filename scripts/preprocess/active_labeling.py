#!/usr/bin/env python
"""Label new keypoints in a raw dataset by ensemble-disagreement (active) pseudo-labeling.

All state lives next to the dataset: `_raw/<dataset>/CollectedData_tolabel.csv` (the rows
being labeled in the LP app; grows each round) and `CollectedData_tmp.csv` (the training
CSV for the current round), plus `poseinterface/results/<dataset>/active/` (manifest, state,
backups, per-round prediction caches and reports). The original CollectedData*.csv are
only written by `finalize`. Never bumps the version.

See ../../skills/preprocess/active-pseudo-labeling/SKILL.md for the workflow and what to ask first.
Before any subcommand that writes a CSV the LP app edits (init, select, reorder, sync, finalize)
the user must leave the dataset's project in the app (back to the home screen listing all
projects).
Must be run in the `pose` conda env for `score`.

Subcommands, in the order they are used:
  init       draw --n frames per session (train + test pooled) into CollectedData_tolabel.csv
             (--allow_existing: the keypoints already hold rough labels to correct)
  train-csv  copy the tolabel rows to CollectedData_tmp.csv (the round's training CSV)
  score      run the trained --model_dirs on every original frame; print per-keypoint stats
  reorder    reorder all tolabel rows by session, then frame (all rounds)
  select     append the --m highest-disagreement unlabeled frames (or `all`), prefilled
             with the ensemble mean, to CollectedData_tolabel.csv
  sync       take every tolabel cell outside the keypoints being labeled from the dataset's
             current CollectedData*.csv (after the dataset's labels change; `train-csv` refuses
             while they differ, `select` syncs the existing rows itself)
  finalize   copy the new keypoints from tolabel into CollectedData.csv / CollectedData_test.csv
  frame-ids  (re)write the per-round lists of tolabel images to _raw/<dataset>/frame_ids_active/

`select --exclude <list.txt>` reserves frames (e.g. for another labeler): the list is recorded in
state.yaml and re-read by every later `select`, so those frames can never be drawn into tolabel.
"""

import argparse
import datetime
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.labels import read_labels_csv, session_name
from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.labeling.active_labeling import (
    add_missing_keypoints,
    combine_originals,
    compare_training_csv,
    ensemble_spread,
    exclude_frames,
    frame_scores,
    init_tolabel,
    make_manifest,
    merge_back,
    order_by_session,
    prefill_rows,
    read_frame_ids,
    round_stats,
    sample_per_session,
    select_frames,
    sync_tolabel,
)
from mighty_mouse.preprocess.labeling.ensemble import predict

TRAIN_CSV, TEST_CSV = "CollectedData.csv", "CollectedData_test.csv"
TOLABEL_CSV, TMP_CSV = "CollectedData_tolabel.csv", "CollectedData_tmp.csv"
FRAME_IDS_DIR = "frame_ids_active"  # per-round tolabel image lists, next to the dataset's CSVs


class Ctx:
    """Paths and persisted state for one dataset."""

    def __init__(self, dataset: str):
        paths = load_paths()
        self.dataset = dataset
        self.raw = Path(paths["raw_dir"]) / dataset
        # standalone-model results live in poseinterface/results/<dataset>/, not results_dir
        self.active = self.raw.parents[1] / "results" / dataset / "active"
        self.state_path = self.active / "state.yaml"
        self.manifest_path = self.active / "tolabel_manifest.csv"

    def load_state(self) -> dict:
        if not self.state_path.exists():
            sys.exit(f"No state at {self.state_path} -- run `init` first")
        return yaml.safe_load(self.state_path.read_text())

    def save_state(self, state: dict) -> None:
        self.state_path.write_text(yaml.safe_dump(state))

    def load_tolabel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(tolabel rows, manifest), checked to list the same images."""
        tolabel = read_exact(self.raw / TOLABEL_CSV)
        manifest = pd.read_csv(self.manifest_path, index_col=0)
        if set(tolabel.index) != set(manifest.index) or len(tolabel) != len(manifest):
            sys.exit(f"{TOLABEL_CSV} and the manifest list different images")
        return tolabel, manifest

    def load_originals(self, keypoints: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        """Train + test labels, with empty columns for the `keypoints` the dataset's CSVs lack."""
        train, test = read_exact(self.raw / TRAIN_CSV), read_exact(self.raw / TEST_CSV)
        both, source = combine_originals(train, test)
        return add_missing_keypoints(both, keypoints), source

    def round_dir(self, r: int) -> Path:
        return self.active / f"round_{r:02d}"

    def backup(self, path: Path, tag: str) -> Path:
        dest = self.active / "backups" / f"{path.stem}_{tag}{path.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 2
        while dest.exists():  # never overwrite an earlier backup
            dest = dest.with_name(f"{path.stem}_{tag}_{n}{path.suffix}")
            n += 1
        shutil.copy2(path, dest)
        return dest


def read_exact(path: Path) -> pd.DataFrame:
    """Read a label CSV without the default float parser's last-digit loss."""
    return read_labels_csv(path, float_precision="round_trip")


def current_round(manifest: pd.DataFrame) -> int:
    return int(manifest["round"].max())


def cache_path(ctx: Ctx, r: int, i: int, model_dir: str) -> Path:
    return ctx.round_dir(r) / f"member{i}_{Path(model_dir).name}.csv"


def load_preds(ctx: Ctx, r: int, model_dirs: list[str]) -> list[pd.DataFrame]:
    preds = []
    for i, d in enumerate(model_dirs):
        p = cache_path(ctx, r, i, d)
        if not p.exists():
            sys.exit(f"Missing cached predictions {p} -- run `score` for this round")
        preds.append(read_labels_csv(p))
    return preds


def spreads(ctx: Ctx, state: dict, originals: pd.DataFrame, tolabel_index: pd.Index):
    """(labeled spread, pool spread) for the round's cached predictions."""
    r = state["scored_round"]
    preds = load_preds(ctx, r, state["rounds"][r]["model_dirs"])
    pool_index = originals.index.difference(tolabel_index, sort=False)
    pool_index = originals.index[originals.index.isin(pool_index)]  # original order
    kps, conf = state["keypoints"], state["min_conf"]
    labeled = ensemble_spread(preds, tolabel_index, kps, conf)
    pool = ensemble_spread(preds, pool_index, kps, conf)
    return preds, labeled, pool


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_init(args) -> None:
    ctx = Ctx(args.dataset)
    if (ctx.raw / TOLABEL_CSV).exists() or ctx.state_path.exists():
        sys.exit(f"{TOLABEL_CSV} or state already exists -- refusing to overwrite")
    originals, source = ctx.load_originals(args.keypoints)
    sampled = sample_per_session(originals.index, args.n, args.seed)
    tolabel = init_tolabel(originals, sampled, args.keypoints, args.allow_existing)
    manifest = make_manifest(tolabel.index, source, 0, False)
    per_session = pd.Series([session_name(i) for i in tolabel.index]).value_counts()
    n_sessions = len({session_name(i) for i in originals.index})
    print(f"{len(originals)} frames in {n_sessions} sessions"
          f" -> sampled {len(tolabel)} ({per_session.min()}-{per_session.max()} per session)")
    print(manifest["source"].value_counts().to_dict())
    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    ctx.active.mkdir(parents=True, exist_ok=True)
    tolabel.to_csv(ctx.raw / TOLABEL_CSV)
    manifest.to_csv(ctx.manifest_path)
    ctx.save_state({
        "keypoints": args.keypoints, "min_conf": args.min_conf, "n": args.n, "seed": args.seed,
        "scored_round": None, "rounds": {},
    })
    print(f"Wrote {ctx.raw / TOLABEL_CSV} -- label {args.keypoints} in the LP app.")


def cmd_train_csv(args) -> None:
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, manifest = ctx.load_tolabel()
    r = current_round(manifest)
    if state["scored_round"] is not None and state["scored_round"] >= r and not args.force:
        sys.exit(f"Round {r} was already scored; `select` first (or pass --force)")
    originals, _ = ctx.load_originals(state["keypoints"])
    synced, changes = sync_tolabel(tolabel, originals, state["keypoints"])
    if len(changes) or list(synced.columns) != list(tolabel.columns):
        sys.exit(f"{TOLABEL_CSV} differs from the dataset's current labels in {len(changes)} "
                 "cells outside the keypoints being labeled: run `sync` first (after the user "
                 "leaves the project in the LP app)")
    shutil.copyfile(ctx.raw / TOLABEL_CSV, ctx.raw / TMP_CSV)
    print(f"Round {r}: wrote {ctx.raw / TMP_CSV} ({len(tolabel)} rows)")
    print(manifest.groupby("round").size().rename("rows").to_string())
    kp_cols = [c for c in tolabel.columns if c[1] in state["keypoints"]]
    print("labeled cells per new keypoint:",
          tolabel[kp_cols].notna().sum().groupby(level=1).sum().div(2).astype(int).to_dict())


def cmd_reorder(args) -> None:
    """Reorder every tolabel row by session (then frame), across all rounds."""
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, manifest = ctx.load_tolabel()
    originals, _ = ctx.load_originals(state["keypoints"])
    order = order_by_session(tolabel.index, originals.index)
    new_tol, new_man = tolabel.loc[order], manifest.loc[order]
    sessions = [session_name(i) for i in order]
    runs = sum(a != b for a, b in zip(sessions, sessions[1:], strict=False)) + 1
    print(f"{len(order)} rows, {len(set(sessions))} sessions, {runs} contiguous runs; "
          f"{(new_tol.index != tolabel.index).sum()} rows change position")
    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    ctx.backup(ctx.raw / TOLABEL_CSV, "before_reorder")
    ctx.backup(ctx.manifest_path, "before_reorder")
    new_tol.to_csv(ctx.raw / TOLABEL_CSV)
    new_man.to_csv(ctx.manifest_path)
    back = read_exact(ctx.raw / TOLABEL_CSV)
    pd.testing.assert_frame_equal(back, new_tol)
    pd.testing.assert_frame_equal(back.sort_index(), tolabel.sort_index())
    print(f"Reordered {ctx.raw / TOLABEL_CSV}; values unchanged.")


def cmd_score(args) -> None:
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, manifest = ctx.load_tolabel()
    r = current_round(manifest)
    tmp_csv = ctx.raw / TMP_CSV
    if not tmp_csv.exists():
        sys.exit(f"No {TMP_CSV} -- was `train-csv` run?")
    tmp_bytes = tmp_csv.read_bytes()
    for d in args.model_dirs:
        copy = d / TMP_CSV
        if not copy.exists() or copy.read_bytes() != tmp_bytes:
            sys.exit(f"{d} was not trained on the current {TMP_CSV}")
    diff = compare_training_csv(read_exact(tmp_csv), tolabel)
    if diff["missing"] or diff["extra"]:
        sys.exit(
            f"The models were trained on a {TMP_CSV} that differs from {TOLABEL_CSV} "
            f"({diff['missing']} rows missing from training, {diff['extra']} extra). Was "
            "`train-csv` skipped after the last `select`? Run it and retrain every member."
        )
    if diff["edited"]:
        print(f"WARNING: {diff['edited']} rows of {TOLABEL_CSV} were edited after `train-csv`")

    originals, _ = ctx.load_originals(state["keypoints"])
    ctx.round_dir(r).mkdir(parents=True, exist_ok=True)
    from lightning_pose.api import Model  # slow import; `pose` env only

    with tempfile.TemporaryDirectory() as tmp:
        input_csv = Path(tmp) / "all_input.csv"
        originals.iloc[:, :0].to_csv(input_csv)
        for i, d in enumerate(args.model_dirs):
            cache = cache_path(ctx, r, i, str(d))
            if cache.exists():
                continue
            model = Model.from_dir2(d)
            p = predict(model, input_csv, ctx.raw, Path(tmp), f"{ctx.dataset}__active_all.csv")
            p.to_csv(cache)
    state["scored_round"] = r
    state["rounds"][r] = {"model_dirs": [str(d) for d in args.model_dirs], "rows": len(tolabel)}
    _, labeled, pool = spreads(ctx, state, originals, tolabel.index)
    stats = round_stats(labeled, pool)
    stats.to_csv(ctx.round_dir(r) / "stats.csv", index=False)
    frame_scores(pool).rename("score").to_csv(ctx.round_dir(r) / "pool_scores.csv")
    ctx.save_state(state)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(f"\nRound {r}: {len(tolabel)} labeled frames, {len(pool)} in the unlabeled pool")
    print(f"Spread = sqrt(var_x + var_y) across {len(args.model_dirs)} members (px); a keypoint "
          f"counts only if some member has likelihood > {state['min_conf']}.")
    print("`baseline` = mean spread on labeled frames (in-sample, so optimistic); x<a>-<b> = "
          "pool frames whose spread is between a and b times that baseline.\n")
    print(stats.to_string(index=False))
    if r > 0:
        prev = ctx.round_dir(r - 1) / "stats.csv"
        if prev.exists():
            old = pd.read_csv(prev).set_index("keypoint")
            print("\nbaseline change vs. previous round:",
                  (stats.set_index("keypoint")["baseline"] - old["baseline"]).round(2).to_dict())
    scores = frame_scores(pool)
    print(f"\npool frame scores: {scores.notna().sum()} eligible, quantiles "
          f"{scores.quantile([.5, .9, .99]).round(2).to_dict()}; "
          f"labeled-frame mean {frame_scores(labeled).mean():.2f}")


def excluded_frames(ctx: Ctx, state: dict) -> pd.Index:
    """Every image reserved by the exclude lists recorded in state.yaml (re-read each time,
    so a list that grows keeps excluding its new frames)."""
    ids = pd.Index([])
    for f in state.get("exclude_files", []):
        path = Path(f) if Path(f).is_absolute() else ctx.raw / f
        if not path.exists():
            sys.exit(f"exclude list {path} (recorded in {ctx.state_path}) is missing -- "
                     "refusing to select")
        ids = ids.append(read_frame_ids(path.read_text()))
    return ids.unique()


def write_frame_ids(ctx: Ctx, manifest: pd.DataFrame) -> None:
    """tolabel_round<R>.txt per round plus tolabel_all.txt, from the manifest."""
    out = ctx.raw / FRAME_IDS_DIR
    out.mkdir(exist_ok=True)
    for r, group in manifest.groupby("round"):
        (out / f"tolabel_round{int(r):02d}.txt").write_text("\n".join(group.index) + "\n")
    (out / "tolabel_all.txt").write_text("\n".join(manifest.index) + "\n")
    print(f"frame lists: {out}/tolabel_round*.txt, tolabel_all.txt ({len(manifest)} images)")


def cmd_frame_ids(args) -> None:
    ctx = Ctx(args.dataset)
    _, manifest = ctx.load_tolabel()
    write_frame_ids(ctx, manifest)


def cmd_select(args) -> None:
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, manifest = ctx.load_tolabel()
    r = current_round(manifest)
    target = r + 1
    if args.extend:
        # more frames for the round already selected, from the same (previous round's) scores:
        # nothing was trained since, and the frames already in tolabel are not in the pool
        if r < 1 or state["scored_round"] != r - 1:
            sys.exit(f"--extend adds frames to round {r}, which needs the scores of round {r - 1} "
                     f"(scored_round={state['scored_round']})")
        target = r
    elif state["scored_round"] != r:
        sys.exit(f"Round {r} hasn't been scored (scored_round={state['scored_round']})")
    originals, source = ctx.load_originals(state["keypoints"])
    preds, labeled, pool = spreads(ctx, state, originals, tolabel.index)
    m = None if args.m == "all" else int(args.m)
    for f in args.exclude or []:
        path = Path(f).resolve()
        if not path.exists():
            sys.exit(f"--exclude {f}: no such file")
        raw = ctx.raw.resolve()
        rel = str(path.relative_to(raw)) if path.is_relative_to(raw) else str(path)
        if rel not in state.setdefault("exclude_files", []):
            state["exclude_files"].append(rel)
    excluded = excluded_frames(ctx, state)
    if len(excluded.intersection(tolabel.index)):
        n = len(excluded.intersection(tolabel.index))
        print(f"WARNING: {n} reserved frames are already in {TOLABEL_CSV}")
    scores = frame_scores(pool)
    n_pool = len(scores)
    scores = exclude_frames(scores, excluded)
    if state.get("exclude_files"):
        print(f"excluding {n_pool - len(scores)} pool frames reserved by {state['exclude_files']} "
              f"({len(excluded)} listed)")
    chosen = order_by_session(select_frames(scores, m), originals.index)
    pool_sessions = len({session_name(i) for i in pool.index})
    rows = prefill_rows(
        originals.loc[chosen], preds, state["keypoints"], args.prefill_threshold, args.prefill_agg
    )
    new_manifest = make_manifest(chosen, source, target, True)

    sc = scores.loc[chosen]
    per_session = pd.Series([session_name(i) for i in chosen]).value_counts()
    print(f"Selecting {len(chosen)} of {len(scores)} eligible pool frames for round {target} "
          f"({sc.isna().sum()} with no eligible keypoint); "
          f"{new_manifest['source'].value_counts().to_dict()}")
    print(f"from {len(per_session)} of {pool_sessions} pool sessions "
          f"(frames/session: median {per_session.median():g}, max {per_session.max()})")
    if sc.notna().any():
        print(f"scores of selected: min {sc.min():.2f}, median {sc.median():.2f}, "
              f"max {sc.max():.2f} (labeled-frame mean {frame_scores(labeled).mean():.2f})")
    kp_cols = [c for c in rows.columns if c[1] in state["keypoints"]]
    print("prefilled cells per keypoint:",
          rows[kp_cols].notna().sum().groupby(level=1).sum().div(2).astype(int).to_dict())
    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    tolabel, changes = sync_tolabel(tolabel, originals, state["keypoints"])
    if len(changes):
        print(f"{len(changes)} cells of the existing rows updated to the dataset's current labels "
              f"(log: {write_sync_log(ctx, changes)})")
    tag = f"before_round{target}" + ("_extend" if args.extend else "")
    ctx.backup(ctx.raw / TOLABEL_CSV, tag)
    ctx.backup(ctx.manifest_path, tag)
    pd.concat([tolabel, rows]).to_csv(ctx.raw / TOLABEL_CSV)
    pd.concat([manifest, new_manifest]).to_csv(ctx.manifest_path)
    ctx.save_state(state)  # records any new --exclude list
    write_frame_ids(ctx, pd.concat([manifest, new_manifest]))
    first = new_manifest.index[0]
    print(f"Appended to {ctx.raw / TOLABEL_CSV} -- review the new rows in the LP app.")
    print(f"First new row (data row {len(tolabel) + 1} of {len(tolabel) + len(rows)}): {first}")


def write_sync_log(ctx: Ctx, changes: pd.DataFrame) -> Path:
    """Save the cells `sync_tolabel` changed (never overwrites an earlier log)."""
    path = ctx.active / f"sync_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_changes.csv"
    changes.to_csv(path, index=False)
    return path


def cmd_sync(args) -> None:
    """Take every tolabel cell outside the keypoints being labeled from the dataset's CSVs."""
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, _ = ctx.load_tolabel()
    originals, _ = ctx.load_originals(state["keypoints"])
    synced, changes = sync_tolabel(tolabel, originals, state["keypoints"])
    dropped = sorted({c[1] for c in tolabel.columns} - {c[1] for c in synced.columns})
    added = sorted({c[1] for c in synced.columns} - {c[1] for c in tolabel.columns})
    n_rows = changes["image"].nunique()
    print(f"{len(changes)} cells in {n_rows} of {len(tolabel)} rows differ from the dataset's "
          "current labels")
    if len(changes):
        print("per keypoint:", changes.groupby("keypoint").size().to_dict())
    if dropped or added:
        print(f"keypoints dropped: {dropped}, added: {added}")
    elif list(synced.columns) != list(tolabel.columns):
        print("columns reordered to follow the dataset's CSVs")
    if not len(changes) and list(synced.columns) == list(tolabel.columns):
        print("Already in sync.")
        return
    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    ctx.backup(ctx.raw / TOLABEL_CSV, "before_sync")
    synced.to_csv(ctx.raw / TOLABEL_CSV)
    pd.testing.assert_frame_equal(read_exact(ctx.raw / TOLABEL_CSV), synced)
    print(f"Wrote {ctx.raw / TOLABEL_CSV}; changed cells logged in {write_sync_log(ctx, changes)}")


def cmd_finalize(args) -> None:
    ctx = Ctx(args.dataset)
    state = ctx.load_state()
    tolabel, manifest = ctx.load_tolabel()
    kps = state["keypoints"]
    results = {}
    for name in (TRAIN_CSV, TEST_CSV):
        orig = read_exact(ctx.raw / name)
        try:
            results[name] = (orig, merge_back(orig, tolabel, kps))
        except ValueError as e:
            sys.exit(f"{name}: {e}")
    for name, (_, new) in results.items():
        cols = [c for c in new.columns if c[1] in kps]
        counts = new[cols].notna().sum().groupby(level=1).sum().div(2).astype(int)
        print(name, "labeled cells:", counts.to_dict(), f"of {len(new)} rows")
    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    for name, (orig, new) in results.items():
        ctx.backup(ctx.raw / name, "before_finalize")
        new.to_csv(ctx.raw / name)
        back = read_exact(ctx.raw / name)
        keep = [c for c in orig.columns if c[1] not in kps]
        pd.testing.assert_frame_equal(back[keep], orig[keep])
        pd.testing.assert_frame_equal(back, new)
    print(f"Wrote {TRAIN_CSV} and {TEST_CSV}. Version NOT bumped; map the keypoints in the "
          "dataset config and verify, then bump when ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("dataset", help="directory under raw_dir")
        p.set_defaults(fn=fn)
        return p

    p = add("init", cmd_init)
    p.add_argument("--keypoints", nargs="+", required=True, help="new keypoints being labeled")
    p.add_argument("--n", type=int, required=True, help="frames drawn per session")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_conf", type=float, default=0.9,
                   help="a keypoint is eligible if some member is above this")
    p.add_argument("--allow_existing", action="store_true",
                   help="keypoints already hold (rough) labels to correct; carry them over")
    p.add_argument("--dry_run", action="store_true")

    p = add("train-csv", cmd_train_csv)
    p.add_argument("--force", action="store_true")

    p = add("reorder", cmd_reorder)
    p.add_argument("--dry_run", action="store_true")

    p = add("score", cmd_score)
    p.add_argument("--model_dirs", nargs="+", required=True, type=Path)

    p = add("select", cmd_select)
    p.add_argument("--m", required=True, help="number of frames to add, or `all`")
    p.add_argument("--prefill_threshold", type=float, default=0.9)
    p.add_argument("--prefill_agg", choices=["mean", "median"], default="mean")
    p.add_argument("--exclude", nargs="+", metavar="LIST",
                   help="frame-id lists (one image path per line) never to select; recorded in "
                        "state.yaml and applied to every later select")
    p.add_argument("--extend", action="store_true",
                   help="add the frames to the round already selected (same scores, nothing "
                        "trained since) instead of starting the next round")
    p.add_argument("--dry_run", action="store_true")

    p = add("sync", cmd_sync)
    p.add_argument("--dry_run", action="store_true")

    p = add("finalize", cmd_finalize)
    p.add_argument("--dry_run", action="store_true")

    add("frame-ids", cmd_frame_ids)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
