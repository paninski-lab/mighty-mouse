#!/usr/bin/env python
"""
Overlay video of one scored active pseudo-labeling round, in house style.

Uses the member predictions `scripts/preprocess/active_labeling.py score` cached in
results/<dataset>/active/round_XX/ (one per model, on every original frame; no model is
re-run) and renders them with ensemble_variance_rows.py: dot = median of the round's models,
cross = +-1 SD across them, frames from the dataset's raw labeled-data/. Which frames:

  --frames pool   frames the round's models were NOT trained on (originals minus tolabel),
                  --n per view, evenly spread over sessions
  --frames top    the --n pool frames with the highest disagreement (pool_scores.csv;
                  ignores `select`'s reserved-frame lists)
  --frames selected  exactly the frames `select` added from this round's scores (the next
                  labeling batch)
  --frames test   every frame of CollectedData_test.csv   (rows also in tolabel WERE trained on)
  --frames train  every frame of CollectedData.csv        (idem)

    python scripts/evaluation/qualitative/active_round_video.py ibl --frames test \
        --keypoint_map <raw -> canonical map>.yaml [--include d[1-4]_tip]

Delivery (skills/evaluation/pose-video, output hygiene):
<results_dir>/qualitative/<dataset>-active-r<R>-<MM-DD>/ with
videos/<what>__<model tokens>_nogt.mp4, csv/ (member predictions on the drawn frames + the
renderer's tables) and README.md (front matter for
scripts/evaluation/qualitative/qualitative_index.py,
model-token table, exact commands).
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths, repo_root
from mighty_mouse.plots.views import view_of

RENDERER = Path(__file__).with_name("ensemble_variance_rows.py")
SPLIT_CSV = {"train": "CollectedData.csv", "test": "CollectedData_test.csv"}


def spread_per_view(frames: pd.Index, dataset: str, n: int) -> pd.Index:
    """At most n frames per camera view, evenly spaced through `frames` (so across sessions),
    in their original order; same view rule as ensemble_variance_rows.py."""
    views = pd.Series([view_of(dataset, Path(f).parent.name) for f in frames], index=frames)
    keep = []
    for _, view in views.groupby(views, sort=True):
        keep += list(view.index[np.unique(np.linspace(0, len(view) - 1,
                                                      min(n, len(view))).astype(int))])
    return frames[frames.isin(keep)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="directory under raw_dir")
    ap.add_argument("--round", type=int, default=None,
                    help="round to draw (default: the last scored one)")
    ap.add_argument("--frames", choices=["pool", "top", "selected", "test", "train"],
                    default="pool")
    ap.add_argument("--n", type=int, default=60,
                    help="pool: frames per view; top: frames in total")
    ap.add_argument("--include", default="", help="comma regexes: keep only matching keypoints")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--per_view", type=int, default=None,
                    help="test / train: at most this many frames per view, evenly spread "
                         "through the split (default: every frame)")
    ap.add_argument("--median_panel", action="store_true",
                    help="add, under each panel, the frame minus its session's median frame "
                         "(dim frames)")
    ap.add_argument("--keypoint_map", default=None,
                    help="raw -> canonical names for the house colours "
                         "(default: configs/datasets/<dataset>.yaml)")
    ap.add_argument("--raw_dir", type=Path, default=None, help="default: paths.yaml raw_dir")
    ap.add_argument("--active_dir", type=Path, default=None,
                    help="default: <raw_dir>/../results/<dataset>/active "
                         "(as scripts/preprocess/active_labeling.py)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="default: <results_dir>/qualitative/<dataset>-active-r<R>-<MM-DD>")
    args = ap.parse_args()

    paths = load_paths()
    raw = (args.raw_dir or Path(paths["raw_dir"])) / args.dataset
    active = args.active_dir or raw.parents[1] / "results" / args.dataset / "active"
    state = yaml.safe_load((active / "state.yaml").read_text())
    r = state["scored_round"] if args.round is None else args.round
    if r is None or r not in state["rounds"]:
        sys.exit(f"round {r} has not been scored -- run "
                 f"`active_labeling.py score {args.dataset}` first")
    model_dirs = state["rounds"][r]["model_dirs"]
    rdir = active / f"round_{r:02d}"
    caches = [rdir / f"member{i}_{Path(d).name}.csv" for i, d in enumerate(model_dirs)]
    preds = [pd.read_csv(c, header=[0, 1, 2], index_col=0) for c in caches]

    trained = pd.read_csv(active / "tolabel_manifest.csv", index_col=0)
    trained = trained[trained["round"] <= r].index
    pool = preds[0].index.difference(trained, sort=False)
    if args.frames in SPLIT_CSV:
        split = pd.read_csv(raw / SPLIT_CSV[args.frames], header=[0, 1, 2], index_col=0).index
        frames = preds[0].index[preds[0].index.isin(split)]
        n_all = len(frames)
        if args.per_view:
            frames = spread_per_view(frames, args.dataset, args.per_view)
        n_seen = int(frames.isin(trained).sum())
        subset = (f"{len(frames)} of the {n_all} {args.frames}-split frames "
                  f"({args.per_view} per view, evenly spread)"
                  if args.per_view else f"all {n_all} {args.frames}-split frames")
        subset += f" ({n_seen} were in this round's training rows)"
        n_frames = len(frames)
    elif args.frames == "selected":
        # the frames `select` added to tolabel from this round's scores (= the next round's
        # labeling batch, reserved frames excluded), in tolabel order (by session)
        manifest = pd.read_csv(active / "tolabel_manifest.csv", index_col=0)
        chosen = manifest.index[manifest["round"] == r + 1]
        if not len(chosen):
            sys.exit(f"no frames were selected from round {r}'s scores yet "
                     f"(manifest has no round {r + 1})")
        frames = chosen[chosen.isin(preds[0].index)]
        subset = f"the {len(frames)} frames selected for labeling round {r + 1} (not trained on)"
        n_frames = len(frames)
    elif args.frames == "top":
        # NaN = no keypoint scored on that frame; `select` never picks those
        scores = pd.read_csv(rdir / "pool_scores.csv", index_col=0)["score"].dropna()
        frames = (scores.loc[scores.index.intersection(pool)]
                  .sort_values(ascending=False).index[:args.n])
        # back to dataset order, so a session's frames stay together
        frames = pool[pool.isin(frames)]
        subset, n_frames = (f"top {len(frames)} disagreement pool frames (not trained on)",
                            len(frames))
    else:
        # --n per view, evenly spaced through the dataset order (so across sessions), as the
        # renderer would pick them; pre-picked here so csv/ only holds the drawn frames
        frames = spread_per_view(pool, args.dataset, args.n)
        subset, n_frames = "pool frames (not trained on)", args.n

    cfg = yaml.safe_load((Path(model_dirs[0]) / "config.yaml").read_text())
    model_tok = f"active-r{r}-{cfg['model']['backbone'].replace('_', '-')}-ens{len(preds)}"
    what = f"{args.dataset}-{args.frames}" + (f"-{args.include.replace(',', '+')}"
                                              if args.include else "")
    what += f"-{args.per_view}perview" if args.per_view and args.frames in SPLIT_CSV else ""
    what += "-medsub" if args.median_panel else ""
    name = f"{what}__{model_tok}_nogt"
    today = datetime.date.today()
    out = args.out_dir or (Path(paths["results_dir"]) / "qualitative"
                           / f"{args.dataset}-active-r{r}-{today:%m-%d}")
    (out / "csv").mkdir(parents=True, exist_ok=True)
    pred_csvs = []
    for i, p in enumerate(preds):
        f = out / "csv" / f"{name}_member{i}.csv"
        p.loc[frames].to_csv(f)
        pred_csvs.append(str(f))

    kmap = args.keypoint_map or repo_root() / "configs" / "datasets" / f"{args.dataset}.yaml"
    cmd = [sys.executable, str(RENDERER), "--preds", *pred_csvs, "--image_dir", str(raw),
           "--no_gt", "--target", args.dataset, "--subset", subset, "--keypoint_map", str(kmap),
           "--modes", "ensemble var", "--frames", str(n_frames), "--conf", str(args.conf),
           "--fps", str(args.fps),
           "--title", f"{model_tok} (median of {len(preds)})", "--out", str(out / f"{name}.mp4")]
    if args.include:
        cmd += ["--include", args.include]
    if args.median_panel:
        cmd += ["--median_panel"]
    subprocess.run(cmd, check=True)

    readme = out / "README.md"
    if not readme.exists():
        readme.write_text(
            f"---\ntitle: {args.dataset} active pseudo-labeling round {r}, ensemble median\n"
            f"date: {today}\ndata_version: _raw/{args.dataset} (active round {r}, "
            f"{state['rounds'][r]['rows']} tolabel rows)\n"
            f"question: How do the round-{r} models' predictions look on "
            f"{args.dataset}'s frames?\n"
            f"models: {model_tok} = {', '.join(Path(d).name for d in model_dirs)}\n"
            "outputs: see below\nfinding: (to fill after viewing)\n---\n\n"
            f"# {args.dataset}, active round {r}\n\n"
            f"Dot = median of the {len(preds)} models, cross = +-1 SD across them (true px), "
            f"drawn if any model conf >= {args.conf}, ring if their mean conf is below it. "
            "No ground truth (`_nogt`). Predictions are "
            f"the ones `active_labeling.py score` cached in `{rdir}`.\n\n"
            f"| token | run directories |\n|---|---|\n| `{model_tok}` | "
            + "<br>".join(f"`{d}`" for d in model_dirs) + " |\n")
    commit = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    note = (" Lower panels: the same overlays on the frame minus its session's median frame "
            "(per-pixel median of every frame in that session folder; gray = unchanged, "
            "contrast stretched per frame)."
            if args.median_panel else "")
    with open(readme, "a") as fh:
        fh.write(f"\n## videos/{name}.mp4\n\n{subset}.{note}\n\nmouse-pose `{commit}`:\n\n"
                 f"```\n{' '.join(sys.argv)}\n```\n")
    print(f"README: {readme}")


if __name__ == "__main__":
    main()
