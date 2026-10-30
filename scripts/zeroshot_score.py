#!/usr/bin/env python
"""
Leave-one-out zero-shot scores: every leave-X-out trunk on the dataset X it never saw, on BOTH
X's test frames (<run>/eval/X/predictions.csv) and train frames
(<run>/zeroshot/X_train_predictions.csv, scripts/zeroshot_predict.py), with the all-data trunk and
the dedicated X model (which saw X's train labels) on X's test frames as references.

Per (dataset, model, split, keypoint), over X's labels with visible == 2
(mighty_mouse.comparison.visible_errors; tongue and lips excluded): n, median / mean px error,
detection = share of labels predicted with likelihood >= --conf, confident-but-wrong = share
predicted with likelihood >= --conf AND error > --wrong_px. A keypoint no other dataset supervises
(not in the leave-out trunk's `trainable` union, from <data_dir>/dataset_inventory.json) is
marked untrained: a zero-shot error there says nothing about transfer. Pooled rows: ALL = every
labeled cell; TRAINED = only keypoints the leave-out trunk was trained on (the fair zero-shot
number). Errors are in each dataset's own image pixels (frame sizes differ between datasets).

    python scripts/zeroshot_score.py                    # results_dir / data_dir from paths.yaml
    python scripts/zeroshot_score.py --area trunks-x --dedicated dedicated-x

Writes <results_dir>/qualitative/zeroshot-<MM-DD>/{summary.csv, per_keypoint.csv, README.md}; the
README has the pooled table and, per left-out dataset, a per-keypoint table ("mean / median" px).
Reads saved predictions only (no lightning-pose needed).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.comparison import trained_keypoints, visible_errors
from mighty_mouse.datasets import ALL_DATASETS, tag_of
from mighty_mouse.paths import load_paths

LEAF   = "supervised/sampling-T2/tf1/vits_dinov3/seed0"
ZERO   = "zero-shot (leave-out)"
POOLED = ("TRAINED", "ALL")


def read_preds(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    return p[[c for c in p.columns if c[2] in ("x", "y", "likelihood")]]


def score(labels: pd.DataFrame, preds: pd.DataFrame, conf: float, wrong_px: float,
          trained: set[str]) -> pd.DataFrame:
    """One row per labeled keypoint, plus pooled rows TRAINED (keypoints in `trained`) and ALL."""
    err = visible_errors(labels, preds)
    sp  = preds.columns[0][0]

    def row(name, e, c):
        return dict(keypoint=name, n=len(e), median_px=np.median(e), mean_px=e.mean(),
                    detect=(c >= conf).mean(), conf_wrong=((c >= conf) & (e > wrong_px)).mean())

    rows, pooled = [], {"TRAINED": ([], []), "ALL": ([], [])}
    for k in err.columns:
        m = err[k].notna()
        if not m.any():
            continue
        e = err.loc[m, k].to_numpy(float)
        c = preds.loc[err.index[m], (sp, k, "likelihood")].to_numpy(float)
        rows.append(row(k, e, c))
        for name in POOLED:
            if name == "ALL" or k in trained:
                pooled[name][0].append(e)
                pooled[name][1].append(c)
    for name, (es, cs) in pooled.items():
        if es:
            rows.append(row(name, np.concatenate(es), np.concatenate(cs)))
    return pd.DataFrame(rows)


def keypoint_table(K: pd.DataFrame, ds: str, conf: float, wrong_px: float,
                   trained: set[str]) -> list[str]:
    """Markdown rows: one per labeled keypoint of `ds` (pooled rows last), "mean / median" px."""
    sub = K[K.dataset == ds]

    def get(model, split, kp, col):
        r = sub[(sub.model == model) & (sub.split == split) & (sub.keypoint == kp)]
        return r[col].iloc[0] if len(r) else np.nan

    def mm(model, split, kp):
        m, md = get(model, split, kp, "mean_px"), get(model, split, kp, "median_px")
        return "" if np.isnan(m) else f"{m:.1f} / {md:.1f}"

    def pct(split, kp, col):
        v = get(ZERO, split, kp, col)
        return "" if np.isnan(v) else f"{v:.0%}"

    def n(split, kp):
        v = get(ZERO, split, kp, "n")
        return "" if np.isnan(v) else str(int(v))

    kps = list(dict.fromkeys(sub[~sub.keypoint.isin(POOLED)].keypoint)) + list(POOLED)
    out = [f"| keypoint | n test / train | zero-shot TEST | zero-shot TRAIN | all-data TEST "
           f"| dedicated TEST | detect (>= {conf}) test / train "
           f"| conf-but-wrong (> {wrong_px:g} px) test / train |",
           "|---|---|---|---|---|---|---|---|"]
    for kp in kps:
        if kp in POOLED:
            name = f"**{kp} keypoints (pooled)**"
        else:
            name = kp if kp in trained else f"{kp} *(untrained: no other dataset labels it)*"
        out.append(
            f"| {name} | {n('test', kp)} / {n('train', kp)} | {mm(ZERO, 'test', kp)} "
            f"| {mm(ZERO, 'train', kp)} | {mm('all-data trunk', 'test', kp)} "
            f"| {mm('dedicated', 'test', kp)} | {pct('test', kp, 'detect')} / "
            f"{pct('train', kp, 'detect')} | {pct('test', kp, 'conf_wrong')} / "
            f"{pct('train', kp, 'conf_wrong')} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=None, help="default: paths.yaml")
    ap.add_argument("--data_dir",    type=Path, default=None, help="default: paths.yaml")
    ap.add_argument("--area",      default="trunks",    help="trunk folder under results_dir")
    ap.add_argument("--dedicated", default="dedicated", help="dedicated folder under results_dir")
    ap.add_argument("--conf",      type=float, default=0.7)
    ap.add_argument("--wrong_px",  type=float, default=25.0)
    ap.add_argument("--out",       type=Path, default=None)
    args = ap.parse_args()
    paths       = load_paths() if args.results_dir is None or args.data_dir is None else {}
    results_dir = args.results_dir or Path(paths["results_dir"])
    data_dir    = args.data_dir or Path(paths["data_dir"])

    inventory = json.loads((data_dir / "dataset_inventory.json").read_text())["datasets"]
    loo, all_run = {}, None
    for tag_dir in sorted((results_dir / args.area).glob("*_train")):
        run = tag_dir / LEAF
        if not (run / "eval").is_dir():
            continue
        zs = list((run / "zeroshot").glob("*_train_predictions.csv"))
        if len(zs) == 1:   # leave-ONE-out; trunks leaving out several datasets are not this table
            loo[zs[0].name[: -len("_train_predictions.csv")]] = run
        elif tag_dir.name == f"{tag_of(ALL_DATASETS)}_train":
            all_run = run

    rows, trained_of = [], {}
    for ds in [d for d in ALL_DATASETS if d in loo]:
        trained = trained_of[ds] = trained_keypoints(set(ALL_DATASETS) - {ds}, inventory)
        labels  = {s: pd.read_csv(data_dir / f"CollectedData_{ds}_{s}.csv", header=[0, 1, 2],
                                  index_col=0)
                   for s in ("test", "train")}
        models = [(ZERO, loo[ds], {"test": loo[ds] / "eval" / ds / "predictions.csv",
                                   "train": loo[ds] / "zeroshot" / f"{ds}_train_predictions.csv"})]
        if all_run:
            models.append(("all-data trunk", all_run,
                           {"test": all_run / "eval" / ds / "predictions.csv"}))
        ded = results_dir / args.dedicated / f"{ds}_train" / LEAF
        if (ded / "eval" / ds / "predictions.csv").exists():
            models.append(("dedicated", ded, {"test": ded / "eval" / ds / "predictions.csv"}))
        for model, run, splits in models:
            for split, f in splits.items():
                if not f.exists():
                    continue
                t = score(labels[split], read_preds(f), args.conf, args.wrong_px, trained)
                t.insert(0, "split", split)
                t.insert(0, "model", model)
                t.insert(0, "dataset", ds)
                t["run"] = str(run.relative_to(results_dir))
                rows.append(t)
    if not rows:
        raise SystemExit("no leave-one-out runs with zero-shot predictions under "
                         f"{results_dir / args.area}")
    K = pd.concat(rows, ignore_index=True)
    out = args.out or results_dir / "qualitative" / f"zeroshot-{date.today():%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    K.round(4).to_csv(out / "per_keypoint.csv", index=False)
    S = K[K.keypoint == "TRAINED"].drop(columns="keypoint")
    S.round(4).to_csv(out / "summary.csv", index=False)

    def pick(ds, model, split):
        r = S[(S.dataset == ds) & (S.model == model) & (S.split == split)]
        return r.iloc[0] if len(r) else None

    def mm(ds, model, split):
        r = pick(ds, model, split)
        return "" if r is None else f"{r.mean_px:.1f} / {r.median_px:.1f}"

    def pct(ds, split, col):
        r = pick(ds, ZERO, split)
        return "" if r is None else f"{r[col]:.0%}"

    table = [f"| left-out dataset | zero-shot TEST | zero-shot TRAIN | all-data TEST "
             f"| dedicated TEST | detect (>= {args.conf}) test / train "
             f"| conf-but-wrong (> {args.wrong_px:g} px) test / train |",
             "|---|---|---|---|---|---|---|"]
    left_out = [d for d in ALL_DATASETS if d in loo]
    for ds in left_out:
        table.append(
            f"| {ds} | {mm(ds, ZERO, 'test')} | {mm(ds, ZERO, 'train')} "
            f"| {mm(ds, 'all-data trunk', 'test')} | {mm(ds, 'dedicated', 'test')} "
            f"| {pct(ds, 'test', 'detect')} / {pct(ds, 'train', 'detect')} "
            f"| {pct(ds, 'test', 'conf_wrong')} / {pct(ds, 'train', 'conf_wrong')} |")
    readme = [
        "---", f"title: Leave-one-out zero-shot, {data_dir.name} ({args.area})",
        f"date: {date.today()}", f"data_version: {data_dir.name}",
        "question: How well does a trunk that never saw dataset X predict X's labeled keypoints?",
        f"models: {args.area} leave-one-out trunks, all-data trunk, {args.dedicated}",
        "outputs: summary.csv, per_keypoint.csv", "finding: (to fill after reading)", "---", "",
        "# Leave-one-out zero-shot", "",
        "Each leave-X-out trunk scored on X's labeled keypoints (visible == 2) on X's test AND",
        "train frames (it saw neither); the all-data trunk and the dedicated X model (both",
        "trained on X's train frames) on X's test frames are the references. detect = labels",
        f"predicted with likelihood >= {args.conf}; conf-but-wrong = likelihood >= {args.conf}",
        f"and error > {args.wrong_px:g} px. Pooled over the keypoints the leave-out trunk was",
        "trained on. Cells: **mean / median px**.", "", *table, "",
        "## Per keypoint, per left-out dataset", "",
        "An empty dedicated cell = that dedicated model has not been evaluated yet.", "",
    ]
    for ds in left_out:
        readme += [f"### {ds}", "",
                   *keypoint_table(K, ds, args.conf, args.wrong_px, trained_of[ds]), ""]
    readme += ["Runs:", ""]
    readme += [f"- leave-{d}-out: `{loo[d].relative_to(results_dir)}`" for d in left_out]
    if all_run:
        readme.append(f"- all-data: `{all_run.relative_to(results_dir)}`")
    readme += ["", "```",
               f"python scripts/zeroshot_score.py --area {args.area} --dedicated {args.dedicated}",
               "```"]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print("\n".join(table))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
