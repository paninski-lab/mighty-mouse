#!/usr/bin/env python
"""Overwrite a raw dataset's pseudo-labels with a confidence-thresholded ensemble mean.

Runs every model in --model_dirs on every image already in the target dataset's label
CSVs (via the lightning_pose Python API, with the target's own data_dir). For each
requested keypoint and image, the members whose likelihood is >= --confidence_threshold
survive; the new label is the mean (x, y) of the survivors, or empty if none survive.
Existing labels in the requested columns are replaced regardless of their previous value
(including being blanked). Columns not named in --keypoints are never touched.

--keypoints entries are `pred_name=target_name` (or a bare name if they match): the
models' keypoint names (here the canonical _left/_right vocabulary) differ from the
target CSV's raw column names (cheese-3d uses `eye(front)(left)` etc.).

Per-model predictions are cached in --out_dir (reused if present), and a report of how
many cells gained / lost a label per keypoint is written there. Never bumps the version.

See ../skills/transfer-pseudo-labels/SKILL.md (ensemble section) for the workflow and
the decisions to confirm first.

Must be run in the `pose` conda env. Dry run first:
    python scripts/ensemble_pseudo_labels.py \\
        --model_dirs results/cheese-2d/<run0> results/cheese-2d/<run1> results/cheese-2d/<run2> \\
        --target_dataset cheese-3d --out_dir results/cheese-3d/ensemble-pseudo-labels \\
        --keypoints "lowerlip" "upperlip_left=upperlip(left)" ... --dry_run
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths

CSV_NAMES_DEFAULT = ["CollectedData.csv", "CollectedData_test.csv"]


def parse_keypoints(specs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for spec in specs:
        pred_kp, _, target_kp = spec.partition("=")
        pairs.append((pred_kp, target_kp or pred_kp))
    return pairs


def predict(model, csv_path: Path, data_dir: Path, tmp: Path, tag: str) -> pd.DataFrame:
    """Run `model` on the images listed in `csv_path`. Only the image paths matter, so a
    label-free CSV with the model's own keypoint columns is used as input (the target's
    columns needn't match the model's). Its basename is unique per target/csv, so it can't
    collide with the model's self-eval predictions (see transfer_pseudo_labels.py)."""
    index = read_labels_csv(csv_path).index
    cols = pd.MultiIndex.from_tuples(
        [("scorer", kp, xy) for kp in model.cfg.data.keypoint_names for xy in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    temp_csv = tmp / tag
    pd.DataFrame(np.nan, index=index, columns=cols).to_csv(temp_csv)
    result = model.predict_on_label_csv(
        csv_file=temp_csv, data_dir=data_dir, compute_metrics=False, add_train_val_test_set=False,
    )
    return result.predictions


def ensemble(
    orig: pd.DataFrame, preds: list[pd.DataFrame], keypoints: list[tuple[str, str]], threshold: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Return (new labels, {target_kp: n_surviving_members per row}).

    A prediction survives if its likelihood is >= threshold. The new (x, y) is the mean
    over survivors, NaN if none survive. Only the target columns in `keypoints` change.
    """
    new = orig.copy()
    scorer = orig.columns.get_level_values(0)[0]
    n_surv = {}
    for pred_kp, kp in keypoints:
        xs, ys, oks = [], [], []
        for p in preds:
            p = p.loc[orig.index]
            scorer_p = p.columns.get_level_values(0)[0]
            ok = (p[(scorer_p, pred_kp, "likelihood")] >= threshold).to_numpy()
            xs.append(np.where(ok, p[(scorer_p, pred_kp, "x")].to_numpy(), np.nan))
            ys.append(np.where(ok, p[(scorer_p, pred_kp, "y")].to_numpy(), np.nan))
            oks.append(ok)
        n = np.sum(oks, axis=0)
        with np.errstate(all="ignore"):  # all-NaN slices -> NaN (the blank case)
            new[(scorer, kp, "x")] = np.nanmean(xs, axis=0)
            new[(scorer, kp, "y")] = np.nanmean(ys, axis=0)
        n_surv[kp] = n
    return new, n_surv


def verify_untouched(orig: pd.DataFrame, new: pd.DataFrame, keypoints: list[tuple[str, str]]) -> None:
    assert list(orig.columns) == list(new.columns) and list(orig.index) == list(new.index)
    edited = {kp for _, kp in keypoints}
    for col in orig.columns:
        if col[1] in edited:
            continue
        o, n = orig[col].astype(float), new[col].astype(float)
        assert ((o.isna() == n.isna()) & ~((o - n).abs() > 1e-6)).all(), f"untouched column changed: {col}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--target_dataset", required=True)
    parser.add_argument("--keypoints", nargs="+", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--csvs", nargs="+", default=CSV_NAMES_DEFAULT)
    parser.add_argument("--out_dir", required=True, type=Path, help="per-model predictions cache + report")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for d in args.model_dirs:
        if not d.is_dir():
            sys.exit(f"No such model dir: {d}")
    keypoints = parse_keypoints(args.keypoints)
    raw_dir = Path(load_paths()["raw_dir"]) / args.target_dataset
    for name in args.csvs:
        if not (raw_dir / name).exists():
            sys.exit(f"Missing {name} in {raw_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from lightning_pose.api import Model  # slow import; `pose` env only

    models = None
    results, report_rows, member_rows = {}, [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.csvs:
            orig = read_labels_csv(raw_dir / name)
            preds = []
            for i, d in enumerate(args.model_dirs):
                cache = args.out_dir / f"{Path(name).stem}_member{i}_{d.name}.csv"
                if cache.exists():
                    p = pd.read_csv(cache, header=[0, 1, 2], index_col=0)
                else:
                    models = models or {}
                    if i not in models:
                        models[i] = Model.from_dir2(d)
                    p = predict(models[i], raw_dir / name, raw_dir, Path(tmp), f"{args.target_dataset}__{name}")
                    p.to_csv(cache)
                preds.append(p)
            new, n_surv = ensemble(orig, preds, keypoints, args.confidence_threshold)
            verify_untouched(orig, new, keypoints)
            results[name] = new

            scorer = orig.columns.get_level_values(0)[0]
            for _, kp in keypoints:
                had = orig[(scorer, kp, "x")].notna().to_numpy()
                has = new[(scorer, kp, "x")].notna().to_numpy()
                n = n_surv[kp]
                report_rows.append({
                    "csv": name, "keypoint": kp, "rows": len(orig),
                    "labeled_before": int(had.sum()), "labeled_after": int(has.sum()),
                    "lost_label (before, blank now)": int((had & ~has).sum()),
                    "gained_label (blank before, now)": int((~had & has).sum()),
                    "members_3": int((n == 3).sum()), "members_2": int((n == 2).sum()),
                    "members_1": int((n == 1).sum()), "members_0": int((n == 0).sum()),
                })
    report = pd.DataFrame(report_rows)
    report.to_csv(args.out_dir / "label_change_report.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 30)
    print(report.to_string(index=False))
    for name, g in report.groupby("csv"):
        print(f"\n{name}: cells labeled {g.labeled_before.sum()} -> {g.labeled_after.sum()}; "
              f"lost {g.iloc[:, 5].sum()}, gained {g.iloc[:, 6].sum()}")
        print("  keypoints that lost labels:", sorted(g[g.iloc[:, 5] > 0].keypoint))
        print("  keypoints that gained labels:", sorted(g[g.iloc[:, 6] > 0].keypoint))
        print("  keypoints with zero labels now:", sorted(g[g.labeled_after == 0].keypoint))

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    for name, new in results.items():
        new.to_csv(raw_dir / name)
    print(f"\nWrote {[str(raw_dir / n) for n in args.csvs]}")
    print("Note: version NOT bumped — verify, then run bump_version.py yourself.")


if __name__ == "__main__":
    main()
