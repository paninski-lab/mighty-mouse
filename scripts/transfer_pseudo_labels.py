#!/usr/bin/env python
"""Transfer pseudo-labels from a trained Lightning Pose model (or precomputed
predictions) onto a raw dataset's existing labeled CSVs — either another dataset's, or
the model's own training dataset when bootstrapping from hand-corrected rows.

Runs the source model's inference (via the lightning_pose Python API — the `litpose
predict` CLI can't override `data_dir` for CSV inputs, which is needed here since the
CSV belongs to a different dataset than the one the model trained on) against every
image already present in the target dataset's label CSVs, then fills in (x, y) for the
requested keypoints wherever the target cell is currently empty AND the prediction's
likelihood is >= --confidence_threshold.

Alternatively, pass --predictions_csvs instead of --model_dir to fill from predictions
someone has already run (e.g. a collaborator's model you don't have locally) — one
LP-format predictions CSV per target CSV, same order as --csvs. Image paths in those
files may carry an extra `<target_dataset>/` directory under `labeled-data/` (as
combined-corpus paths do); that's stripped before matching rows to the target CSV.

A --keypoints entry can be `pred_name=target_name` when the prediction's keypoint name
differs from the target CSV's column name (e.g. `pad_top_right=pad_top`); a bare name
means the two are the same.

--overwrite also replaces cells that already hold a label (e.g. an earlier, worse
round of pseudo-labels) whenever the new prediction clears the threshold; cells below
it keep whatever they had. Pair it with --protect_rows_from <csv> to shield rows that
have been hand-corrected: any target row whose image path appears in that label CSV
(e.g. the subset the model was trained on) is never modified.

What this script deliberately does NOT do:
- add new rows/images (only fills cells for images already in the target CSVs)
- touch any cell that already has a label (unless --overwrite is passed)
- touch any row listed in --protect_rows_from
- touch any keypoint not named in --keypoints
- modify the target dataset's schema (the keypoint names passed in --keypoints must
  already be columns in the target CSV — this never adds new keypoint columns)
- call bump_version.py — bumping the target dataset's version is a separate,
  user-requested step, after they've verified the result

Must be run in the `pose` conda env:
    python scripts/transfer_pseudo_labels.py ...

See ../skills/transfer-pseudo-labels/SKILL.md for the full workflow and the checklist
of decisions to confirm with the user before running this (keypoint-name overlap
between source model and target dataset, confidence threshold, scope, per-view
masking).

Usage example (dry run first):
    python scripts/transfer_pseudo_labels.py \\
        --model_dir /media/mattw/poseinterface/results/cheese-3d/2026-09-26_15-45-24 \\
        --target_dataset cheese-2d \\
        --keypoints "pad(top)(left)" "pad(side)(left)" "pad(top)(right)" "pad(side)(right)" \\
        --confidence_threshold 0.7 \\
        --dry_run

    # from precomputed predictions, with renaming
    python scripts/transfer_pseudo_labels.py \\
        --predictions_csvs preds_train.csv preds_test.csv \\
        --target_dataset kaufman \\
        --keypoints nose_tip pad_top_right=pad_top \\
        --dry_run

    # overwrite existing (pseudo-)labels, except rows the model was trained on
    python scripts/transfer_pseudo_labels.py \\
        --model_dir /media/mattw/poseinterface/results/kaufman/2026-09-27_17-35-42 \\
        --target_dataset kaufman \\
        --keypoints eye_back nose_tip \\
        --overwrite --protect_rows_from CollectedData_tmp.csv \\
        --dry_run
"""
import argparse
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths

CSV_NAMES_DEFAULT = ["CollectedData.csv", "CollectedData_test.csv"]


def guess_group(image_path: str) -> str:
    """Best-effort grouping key for the per-group inventory (e.g. camera view).

    Takes the last underscore-separated token of the image's parent directory name,
    dropping a trailing HH-MM-SS timestamp token if present. This is a generic
    heuristic, not dataset-specific: it happens to recover the camera view for the
    cheese-2d/cheese-3d naming convention (session..._<VIEW>[_HH-MM-SS]), but for a
    dataset with a different directory-naming scheme it may just recover something
    else (or nothing useful) — it's informational only, never used to decide which
    cells get filled.
    """
    parts = image_path.split("/")
    if len(parts) < 2:
        return "UNKNOWN"
    tokens = parts[-2].split("_")
    if re.fullmatch(r"\d{2}-\d{2}-\d{2}", tokens[-1]):
        tokens = tokens[:-1]
    return tokens[-1] if tokens else "UNKNOWN"


def parse_keypoints(specs: list[str]) -> list[tuple[str, str]]:
    """Parse --keypoints entries into (pred_name, target_name) pairs."""
    pairs = []
    for spec in specs:
        pred_kp, _, target_kp = spec.partition("=")
        pairs.append((pred_kp, target_kp or pred_kp))
    return pairs


def load_predictions_csv(path: Path, target_dataset: str, target_index: pd.Index) -> pd.DataFrame:
    """Load a precomputed LP predictions CSV and reindex it to match `target_index`.

    Strips a `labeled-data/<target_dataset>/` prefix down to `labeled-data/` so
    combined-corpus-style image paths line up with the raw dataset's own. Exits if any
    target image has no prediction.
    """
    preds = read_labels_csv(path)
    preds.index = preds.index.str.replace(
        f"labeled-data/{target_dataset}/", "labeled-data/", n=1, regex=False
    )
    if preds.index.duplicated().any():
        sys.exit(f"{path}: duplicate image paths after prefix stripping")
    missing = target_index.difference(preds.index)
    if len(missing):
        sys.exit(f"{path}: no predictions for {len(missing)} target image(s), e.g. {missing[0]}")
    return preds.loc[target_index]


def predict_with_model(model, orig_path: Path, temp_csv: Path, data_dir: Path) -> pd.DataFrame:
    """Run `model` on every image in the label CSV at `orig_path`.

    The CSV is first copied to `temp_csv`, whose basename must be distinct from every
    csv this model dir has ever predicted on before (predict_on_label_csv keys its
    output dir purely off the csv basename — reusing "CollectedData.csv" would silently
    overwrite the model's own self-eval predictions from training).
    """
    temp_csv.write_bytes(orig_path.read_bytes())
    result = model.predict_on_label_csv(
        csv_file=temp_csv,
        data_dir=data_dir,
        compute_metrics=False,
        add_train_val_test_set=False,
    )
    print(f"predictions written to: {model.image_preds_dir() / temp_csv.name / 'predictions.csv'}")
    return result.predictions


def fill_missing(
    orig: pd.DataFrame,
    preds: pd.DataFrame,
    keypoints: list[tuple[str, str]],
    threshold: float,
    overwrite: bool = False,
    protected_rows: pd.Index | None = None,
):
    """Fill (x, y) for each (pred_kp, target_kp) pair in `keypoints`, copying the
    prediction for `pred_kp` into `target_kp` wherever orig is NaN (or anywhere, if
    `overwrite`) and the prediction's likelihood >= threshold. Rows in
    `protected_rows` are never modified.

    Returns (filled_df, counts, n_overwritten, cells_filled_per_row, editable):
      - counts: {target_keypoint: n_cells_filled}, including overwritten cells
      - n_overwritten: {target_keypoint: n_previously_labeled_cells_replaced}
      - editable: boolean frame (orig's shape) of the cells this call was allowed to
        change — pass to verify_untouched
      - cells_filled_per_row: {image_path: n_keypoints_filled_for_this_row}, omitting
        rows with 0 fills — used only for the per-group inventory, not to decide fills.

    Raises KeyError if a keypoint is missing from either dataframe's columns.
    """
    orig = orig.copy()
    scorer = orig.columns.get_level_values(0)[0]
    pred_scorer = preds.columns.get_level_values(0)[0]

    counts: dict[str, int] = {}
    n_overwritten: dict[str, int] = {}
    cells_filled_per_row: dict[str, int] = defaultdict(int)
    editable = pd.DataFrame(False, index=orig.index, columns=orig.columns)
    unprotected = ~orig.index.isin(protected_rows if protected_rows is not None else [])

    for pred_kp, kp in keypoints:
        x_col, y_col = (scorer, kp, "x"), (scorer, kp, "y")
        px_col, py_col, pl_col = (
            (pred_scorer, pred_kp, "x"),
            (pred_scorer, pred_kp, "y"),
            (pred_scorer, pred_kp, "likelihood"),
        )

        if x_col not in orig.columns or y_col not in orig.columns:
            raise KeyError(f"keypoint {kp!r} not found in target CSV columns")
        if px_col not in preds.columns or pl_col not in preds.columns:
            raise KeyError(
                f"keypoint {pred_kp!r} not found in predictions (model wasn't trained on it?)"
            )

        missing_mask = orig[x_col].isna()
        allowed = unprotected & (True if overwrite else missing_mask)
        editable[x_col] = allowed
        editable[y_col] = allowed
        pred_conf = preds.loc[orig.index, pl_col]
        fill_mask = allowed & (pred_conf >= threshold)
        n_overwritten[kp] = int((fill_mask & ~missing_mask).sum())

        n = int(fill_mask.sum())
        if n:
            orig.loc[fill_mask, x_col] = preds.loc[orig.index[fill_mask], px_col].values
            orig.loc[fill_mask, y_col] = preds.loc[orig.index[fill_mask], py_col].values
            for path in orig.index[fill_mask]:
                cells_filled_per_row[path] += 1
        counts[kp] = n

    return orig, counts, n_overwritten, dict(cells_filled_per_row), editable


def verify_untouched(orig: pd.DataFrame, filled: pd.DataFrame, editable: pd.DataFrame) -> None:
    """Sanity check: columns/index unchanged, and every cell outside `editable` is
    unchanged in `filled` (same value, or still empty). Raises AssertionError on any
    violation — this is the safety net for an irreversible-ish edit to a dataset's
    ground-truth label file.
    """
    assert list(orig.columns) == list(filled.columns), "columns changed"
    assert list(orig.index) == list(filled.index), "row order/index changed"
    for col in orig.columns:
        locked = ~editable[col]
        o = orig.loc[locked, col].astype(float)
        f = filled.loc[locked, col].astype(float)
        bad = (o.isna() != f.isna()) | ((o - f).abs() > 1e-6)
        if bad.any():
            raise AssertionError(
                f"{bad.sum()} cell(s) outside the editable set changed in column {col}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model_dir", type=Path, help="path to a trained Lightning Pose model dir",
    )
    source.add_argument(
        "--predictions_csvs", nargs="+", type=Path,
        help="precomputed LP predictions CSVs to fill from instead of running a model, "
        "one per --csvs entry, in the same order",
    )
    parser.add_argument(
        "--target_dataset", required=True,
        help="name of the target dataset (must match a directory under raw_dir, see paths.yaml)",
    )
    parser.add_argument(
        "--keypoints", required=True, nargs="+",
        help="keypoint names to transfer (must already be columns in the target CSV(s), "
        "and must be keypoints the source model was trained on); use pred_name=target_name "
        "to fill a target column from a differently-named prediction",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.7,
        help="only fill cells whose prediction likelihood is >= this (default: 0.7)",
    )
    parser.add_argument(
        "--csvs", nargs="+", default=CSV_NAMES_DEFAULT,
        help=f"which of the target dataset's label CSVs to touch (default: {CSV_NAMES_DEFAULT})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="also replace cells that already hold a label when the prediction clears "
        "the threshold",
    )
    parser.add_argument(
        "--protect_rows_from",
        help="label CSV (relative to the target dataset dir, or absolute) whose image paths "
        "are never modified in any target CSV — e.g. the hand-corrected subset the model "
        "trained on",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="print the inventory without writing anything",
    )
    args = parser.parse_args()

    if args.model_dir and not args.model_dir.is_dir():
        sys.exit(f"No such model dir: {args.model_dir}")
    if args.predictions_csvs:
        if len(args.predictions_csvs) != len(args.csvs):
            sys.exit(f"--predictions_csvs needs one file per --csvs entry ({args.csvs})")
        for path in args.predictions_csvs:
            if not path.exists():
                sys.exit(f"No such predictions CSV: {path}")
    keypoints = parse_keypoints(args.keypoints)

    raw_dir = Path(load_paths()["raw_dir"]) / args.target_dataset
    if not raw_dir.is_dir():
        sys.exit(f"No such raw dataset directory: {raw_dir}")

    for name in args.csvs:
        if not (raw_dir / name).exists():
            sys.exit(f"Missing {name} in {raw_dir}")

    protected_rows = None
    if args.protect_rows_from:
        protect_path = raw_dir / args.protect_rows_from
        if not protect_path.exists():
            sys.exit(f"No such --protect_rows_from CSV: {protect_path}")
        protected_rows = read_labels_csv(protect_path).index
        print(f"protecting {len(protected_rows)} row(s) listed in {protect_path}")

    if args.model_dir:
        # Delayed: slow import, and only available in the `pose` conda env.
        from lightning_pose.api import Model

        model = Model.from_dir2(args.model_dir)

    grand_counts: dict[str, dict[str, int]] = {}
    grand_overwritten: dict[str, dict[str, int]] = {}
    grand_group_counts: dict[str, int] = defaultdict(int)
    filled_frames: dict[str, pd.DataFrame] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, name in enumerate(args.csvs):
            orig_path = raw_dir / name
            orig = read_labels_csv(orig_path)

            if args.predictions_csvs:
                preds = load_predictions_csv(
                    args.predictions_csvs[i], args.target_dataset, orig.index
                )
                print(f"{name}: using predictions from {args.predictions_csvs[i]}")
            else:
                tmp_csv = tmp / f"{args.target_dataset}__{name}"
                preds = predict_with_model(model, orig_path, tmp_csv, raw_dir)

            filled, counts, n_overwritten, cells_filled_per_row, editable = fill_missing(
                orig, preds, keypoints, args.confidence_threshold, args.overwrite, protected_rows,
            )
            verify_untouched(orig, filled, editable)
            if protected_rows is not None:
                print(f"{name}: {int(orig.index.isin(protected_rows).sum())} protected row(s)")

            for path, n in cells_filled_per_row.items():
                grand_group_counts[guess_group(path)] += n

            grand_counts[name] = counts
            grand_overwritten[name] = n_overwritten
            filled_frames[name] = filled

    condition = "any unprotected cell" if args.overwrite else "previously empty"
    print(f"\n=== fill inventory (cells filled, {condition} + confidence >= "
          f"{args.confidence_threshold}) ===")
    keypoint_totals: dict[str, int] = defaultdict(int)
    for name, counts in grand_counts.items():
        total = sum(counts.values())
        print(f"\n{name}: {total} cells")
        for kp, n in counts.items():
            if n:
                n_over = grand_overwritten[name][kp]
                note = f" ({n_over} overwrote an existing label)" if n_over else ""
                print(f"  {kp}: +{n}{note}")
            keypoint_totals[kp] += n
    print(f"\ntotal cells filled across all CSVs: {sum(keypoint_totals.values())}")
    print("per-keypoint total:")
    for kp, n in keypoint_totals.items():
        print(f"  {kp}: {n}")
    print("\nper-group (best-effort, see guess_group docstring):")
    for group, n in sorted(grand_group_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {group}: {n}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    for name, filled in filled_frames.items():
        filled.to_csv(raw_dir / name)
    print(f"\nWrote updated CSVs: {[str(raw_dir / n) for n in args.csvs]}")
    print(
        "Note: version NOT bumped — run bump_version.py yourself once you've verified "
        "the result."
    )


if __name__ == "__main__":
    main()
