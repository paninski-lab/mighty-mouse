#!/usr/bin/env python
"""
Standard evaluation tables for the Mighty Mouse recipe's models of a corpus version.

Discovers the finished runs under <results_dir>/trunks* and <results_dir>/dedicated*
(scripts/train_plan.sh layout), or takes explicit --run name=path pairs, and from the
pixel_error.csv that evaluate_model writes for every dataset's test set produces:

  summary.csv        one row per (run, dataset): pooled median / mean pixel error over labeled
                     keypoints (visible == 2; tongue and lips excluded), frames, keypoints
  per_keypoint.csv   one row per (run, dataset, keypoint): median / mean px error, n labels
  README.md          the summary as a table, plus per dataset the all-data trunk vs that
                     dataset's dedicated model per keypoint; front matter for
                     scripts/qualitative/qualitative_index.py

Reads saved predictions only (no lightning-pose needed). Run names: all, LOO-<dataset>,
dedicated:<dataset> (a variant folder such as trunks-x adds @x; seeds other than 0 add #seed<N>).

    python scripts/eval_suite.py                                  # current results_dir
    python scripts/eval_suite.py --results_dir <path> --out <dir>
    python scripts/eval_suite.py --run all=<run_dir> --run prev-all=<old run_dir>
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.comparison import ALWAYS_EXCLUDED
from mighty_mouse.datasets import ALL_DATASETS, SHORT_NAMES
from mighty_mouse.paths import load_paths

LEAF = "supervised/sampling-T2/tf1/vits_dinov3"


def run_label(area: str, tag: str) -> str:
    """Readable name of a run: all / LOO-<ds> / dedicated:<ds>, plus @<variant> for an area such
    as trunks-<variant>; a tag that is none of these keeps its raw name."""
    variant = area.split("-", 1)[1] if "-" in area else ""
    suffix  = f"@{variant}" if variant else ""
    if area.startswith("dedicated"):
        return f"dedicated:{tag}{suffix}"
    shorts  = set(tag.split("+"))
    missing = [d for d in ALL_DATASETS if SHORT_NAMES.get(d) not in shorts]
    if not missing:
        return f"all{suffix}"
    if len(missing) == 1 and len(shorts) == len(ALL_DATASETS) - 1:
        return f"LOO-{missing[0]}{suffix}"
    return tag + suffix


def discover(results_dir: Path) -> dict[str, Path]:
    """Every evaluated run under results_dir/trunks*/ and results_dir/dedicated*/."""
    runs = {}
    for area_dir in sorted([*results_dir.glob("trunks*"), *results_dir.glob("dedicated*")]):
        for tag_dir in sorted(area_dir.glob("*_train")):
            for seed_dir in sorted(tag_dir.glob(f"{LEAF}/seed*")):
                if (seed_dir / "eval").is_dir():
                    name = run_label(area_dir.name, tag_dir.name[: -len("_train")])
                    seed = f"#{seed_dir.name}" if seed_dir.name != "seed0" else ""
                    runs[name + seed] = seed_dir
    return runs


def errors(run: Path, dataset: str) -> pd.DataFrame | None:
    """A run's per-frame, per-keypoint errors on one test set (None if not evaluated)."""
    f = run / "eval" / dataset / "pixel_error.csv"
    if not f.exists():
        return None
    df   = pd.read_csv(f, index_col=0)
    keep = [c for c in df.columns
            if c not in ALWAYS_EXCLUDED and c != "set" and not c.startswith("Unnamed")]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def md_table(piv: pd.DataFrame) -> str:
    cols = list(piv.columns)
    rows = ["| dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for ds, r in piv.iterrows():
        rows.append(f"| {ds} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in r) + " |")
    return "\n".join(rows)


def keypoint_section(ds: str, runs: dict[str, Path], S: pd.DataFrame,
                     K: pd.DataFrame) -> list[str]:
    """Markdown: per keypoint of `ds`, the all-data trunk(s) vs its dedicated model(s), in
    mean / median px."""
    base = {r: r.split("#")[0].split("@")[0] for r in runs}
    cols = ([r for r in runs if base[r] == "all"]
            + [r for r in runs if base[r] == f"dedicated:{ds}"])
    sub  = K[(K.dataset == ds) & K.run.isin(cols)] if len(K) else K
    if sub.empty:
        return []
    out = ["| keypoint | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in dict.fromkeys(sub.keypoint):
        cells = []
        for c in cols:
            r = sub[(sub.run == c) & (sub.keypoint == k)]
            cells.append(f"{r.mean_px.iloc[0]:.1f} / {r.median_px.iloc[0]:.1f}" if len(r) else "")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    pooled = []
    for c in cols:
        r = S[(S.run == c) & (S.dataset == ds)]
        pooled.append(f"**{r.mean_px.iloc[0]:.1f} / {r.median_px.iloc[0]:.1f}**" if len(r) else "")
    out.append("| **pooled** | " + " | ".join(pooled) + " |")
    return [f"### {ds}", "", *out, ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=None,
                    help="default: paths.yaml results_dir")
    ap.add_argument("--data_version", default=None,
                    help="for the README (default: the name of paths.yaml data_dir)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                    help="add (or replace) a run by name; repeatable")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <results_dir>/qualitative/eval-suite-<MM-DD>)")
    args = ap.parse_args()
    paths        = load_paths() if args.results_dir is None or args.data_version is None else {}
    results_dir  = args.results_dir or Path(paths["results_dir"])
    data_version = args.data_version or Path(paths["data_dir"]).name

    runs = discover(results_dir)
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = Path(path)
    if not runs:
        raise SystemExit(f"no evaluated runs under {results_dir}/trunks* or /dedicated*")
    out = args.out or results_dir / "qualitative" / f"eval-suite-{date.today():%m-%d}"
    out.mkdir(parents=True, exist_ok=True)

    rows, kp_rows = [], []
    for name, run in runs.items():
        for ds in ALL_DATASETS:
            e = errors(run, ds)
            if e is None:
                continue
            vals = e.to_numpy().ravel()
            vals = vals[np.isfinite(vals)]
            rows.append(dict(
                run=name, dataset=ds, frames=len(e), keypoints=int(e.notna().any().sum()),
                labels=int(vals.size),
                median_px=float(np.median(vals)) if vals.size else np.nan,
                mean_px=float(np.mean(vals)) if vals.size else np.nan,
            ))
            for k in e.columns:
                v = e[k].dropna()
                if len(v):
                    kp_rows.append(dict(run=name, dataset=ds, keypoint=k, n=len(v),
                                        median_px=float(v.median()), mean_px=float(v.mean())))
    S, K = pd.DataFrame(rows), pd.DataFrame(kp_rows)
    S.round(3).to_csv(out / "summary.csv", index=False)
    K.round(3).to_csv(out / "per_keypoint.csv", index=False)

    piv = S.pivot(index="dataset", columns="run", values="median_px").reindex(ALL_DATASETS)
    sections = [line for ds in ALL_DATASETS for line in keypoint_section(ds, runs, S, K)]
    today = date.today()
    lines = [
        "---", f"title: Evaluation suite {today}", f"date: {today}",
        f"data_version: {data_version}",
        "question: How do the recipe's models score on every dataset's test set?",
        f"models: {', '.join(runs)}", "outputs: summary.csv, per_keypoint.csv",
        "finding: (fill in after reading the table)", "---", "",
        "# Evaluation suite", "",
        "Median pixel error over labeled keypoints (visible == 2; tongue and lips excluded), each",
        "run scored on every dataset's test set. A dedicated model scored on another dataset only",
        "measures the keypoints the two share.", "", md_table(piv.round(2)), "",
        "## Per keypoint: all-data trunk vs dedicated (test set, mean / median px)", "",
        "Leave-one-out zero-shot tables: `scripts/zeroshot_score.py`.", "", *sections,
        "Per-keypoint numbers: `per_keypoint.csv`. Runs:", "",
    ]
    for n, p in runs.items():
        shown = p.relative_to(results_dir) if p.is_relative_to(results_dir) else p
        lines.append(f"- `{n}`: `{shown}`")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(piv.round(2).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
