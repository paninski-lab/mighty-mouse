"""
Inventory of the head-fixed corpus: one table row per dataset, plus a machine-readable manifest.

Per dataset: label version, train/test frame and session counts, label counts (cells with
visible != 0, occluded included), the number of keypoints per region of configs/keypoints.yaml
(left/right pairs counted once), and the two keypoint masks below. Also asserts that the CSVs
have the canonical column layout (x, y, visible for each keypoint of configs/keypoints.yaml, in
order) and that train and test sessions are disjoint, and reports raw labeled-data directories
the conversion never used (e.g. kondo's iter7_* sets) without adding them.

    python -m mighty_mouse.preprocess.inventory --source corpus|raw [--raw_dir <raw>]
                                                [--data_dir <corpus>] [--out_dir <dir>]

Two sources, one implementation:
  corpus  the converted corpus in <data_dir> (convert_dataset.py runs this when it finishes).
          Writes dataset_inventory.{json,md,csv} into <data_dir>; datasets that are not
          converted yet are skipped. The JSON is read by the evaluation code.
  raw     converts the raw datasets in memory (bump_version.py runs this after a bump), so it
          needs no corpus. Writes only assets/corpus.{md,csv}, which are checked in.

Masks (full lists in the JSON):
  direct     keypoints with at least one visible=2 label in the train CSV
  trainable  direct, plus _left/_right partners of lateralized direct keypoints
             (hflip augmentation gives these training signal without direct labels)
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from mighty_mouse.configs import (
    collapse_side,
    default_configs_dir,
    hflip_partner,
    load_canonical_keypoints,
    load_dataset_config,
    load_regions,
)
from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import COORDS, read_labels_csv
from mighty_mouse.paths import load_paths, repo_root
from mighty_mouse.preprocess.convert import (
    TEST_CSV,
    TRAIN_CSV,
    post_process,
    process_split,
    raw_dataset_dir,
)

SOURCES = ("corpus", "raw")

# ── per-dataset extraction ───────────────────────────────────────────────────

def column_layout(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(keypoint, coord) of every column, in order."""
    return list(zip(df.columns.get_level_values(1), df.columns.get_level_values(2), strict=True))


def sessions(df: pd.DataFrame) -> set[str]:
    """Session names parsed from labeled-data/<dataset>/<session>/<frame> row paths."""
    return {Path(i).parts[2] for i in df.index}


def direct_mask(df: pd.DataFrame) -> set[str]:
    """Keypoints with at least one visible=2 label."""
    vis = df.loc[:, df.columns.get_level_values(2) == "visible"]
    kps = vis.columns.get_level_values(1)
    return {k for k, col in zip(kps, vis.columns, strict=True) if (df[col] == 2).any()}


def trainable_mask(direct: set[str], canonical: list[str]) -> set[str]:
    """Direct keypoints plus hflip partners that exist in the canonical vocabulary."""
    partners = {p for k in direct if (p := hflip_partner(k)) and p in canonical}
    return direct | partners


def count_labels(df: pd.DataFrame) -> int:
    """Number of (frame, keypoint) cells with visible != 0 -- labeled and occluded alike.
    visible=0 cells (keypoints the dataset doesn't cover, an unassessed side, ...) don't count."""
    return int((df.xs("visible", axis=1, level=2) != 0).to_numpy().sum())


def unreferenced_raw_sessions(raw_dataset: Path, referenced: set[str]) -> list[str]:
    """Raw labeled-data directories of a raw dataset folder that the conversion never consumed."""
    raw_labeled = raw_dataset / "labeled-data"
    if not raw_labeled.is_dir():
        return []
    return sorted(
        d.name for d in raw_labeled.iterdir() if d.is_dir() and d.name not in referenced
    )


def dataset_entry(
    name: str,
    canonical: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    regions: dict[str, str],
    version: str = "",
) -> dict:
    """Build the inventory entry for one dataset from its converted label frames.

    Args:
        name: dataset name
        canonical: keypoints from configs/keypoints.yaml
        train: converted train labels (canonical layout)
        test: converted test labels (canonical layout)
        regions: keypoint -> region, from `load_regions`
        version: the dataset's label version ("" if unversioned)

    Raises:
        ValueError: if a split's columns are not x, y, visible for each canonical keypoint in
            order, or if train and test share a session
    """
    expected = [(k, c) for k in canonical for c in COORDS]
    for split, df in (("train", train), ("test", test)):
        if column_layout(df) != expected:
            raise ValueError(
                f"{name} {split} labels deviate from the canonical layout (x, y, visible "
                f"per keypoint of configs/keypoints.yaml, in order): {len(df.columns)} vs "
                f"{len(expected)} columns, or wrong order"
            )

    train_sessions = sessions(train)
    test_sessions  = sessions(test)
    overlap        = train_sessions & test_sessions
    if overlap:
        raise ValueError(
            f"{name}: train/test sessions overlap: {sorted(overlap)} — the evaluation "
            f"depends on session disjointness, refusing to continue"
        )

    direct    = direct_mask(train)
    collapsed = {collapse_side(k) for k in direct}
    collapsed_region = {collapse_side(k): r for k, r in regions.items()}
    region_counts = {r: sum(collapsed_region[k] == r for k in collapsed)
                     for r in dict.fromkeys(regions.values())}
    return {
        "version":         version,
        "train_frames":    len(train),
        "test_frames":     len(test),
        "train_sessions":  len(train_sessions),
        "test_sessions":   len(test_sessions),
        "train_labels":    count_labels(train),
        "test_labels":     count_labels(test),
        "regions":         region_counts,
        "direct":          sorted(direct),
        "trainable":       sorted(trainable_mask(direct, canonical)),
    }


# ── sources ──────────────────────────────────────────────────────────────────

def load_converted(data_dir: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(train, test) converted labels of a dataset in the corpus, None if it isn't converted."""
    paths = [data_dir / f"CollectedData_{name}_{split}.csv" for split in ("train", "test")]
    if not all(p.exists() for p in paths):
        return None
    return tuple(read_labels_csv(p) for p in paths)


def convert_in_memory(
    raw_folder: Path, name: str, config: dict, canonical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(train, test) labels as the conversion would write them, None if the raw dataset lacks a
    label CSV. Runs the real conversion (process_split + the dataset's POST_PROCESS hook)."""
    paths = [raw_folder / TRAIN_CSV, raw_folder / TEST_CSV]
    if not all(p.exists() for p in paths):
        return None
    return tuple(
        post_process(process_split(read_labels_csv(p), config, name, canonical), config, name)
        for p in paths
    )


# ── tables ───────────────────────────────────────────────────────────────────

def make_table(entries: dict[str, dict]) -> tuple[list[str], list[list]]:
    """Header and rows of the table, one row per dataset plus a total row.

    The total row sums frames and labels; a region column there counts the datasets with at
    least one direct keypoint in that region.
    """
    regions = list(next(iter(entries.values()))["regions"]) if entries else []
    header = ["dataset", "version", "train_frames", "test_frames", "train_sessions",
              "test_sessions", "train_labels", "test_labels", *regions, "direct", "trainable"]
    rows = [
        [name, e["version"], e["train_frames"], e["test_frames"], e["train_sessions"],
         e["test_sessions"], e["train_labels"], e["test_labels"],
         *[e["regions"][r] for r in regions], len(e["direct"]), len(e["trainable"])]
        for name, e in entries.items()
    ]
    n_regions = len(regions)
    total = ["TOTAL", "", *[sum(r[i] for r in rows) for i in (2, 3)], "", "",
             *[sum(r[i] for r in rows) for i in (6, 7)],
             *[f"{sum(r[8 + i] > 0 for r in rows)} / {len(rows)}" for i in range(n_regions)],
             "", ""]
    return header, rows + [total]


def write_tables(header: list[str], rows: list[list], out_dir: Path, stem: str) -> list[Path]:
    """Write <stem>.csv (opens in Google Sheets) and <stem>.md (renders on GitHub) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.md"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path]


def run(
    source: str, raw_dir: Path, data_dir: Path | None = None, out_dir: Path | None = None
) -> list[Path]:
    """Build the inventory from `source` and write it.

    Args:
        source: "corpus" (converted CSVs in `data_dir`; also writes the JSON manifest) or "raw"
            (in-memory conversion of the raw datasets)
        raw_dir: raw datasets (for in-memory conversion, versions and unreferenced dirs)
        data_dir: converted corpus, required for "corpus"
        out_dir: where to write (default: `data_dir` for "corpus", the repo's assets/ for "raw")

    Returns:
        the files written

    Raises:
        ValueError: if a dataset's labels deviate from the canonical layout or its train and
            test sessions overlap
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    if source == "corpus" and data_dir is None:
        raise ValueError("the corpus source needs a data_dir")
    configs_dir = default_configs_dir()
    canonical   = load_canonical_keypoints(configs_dir)
    regions     = load_regions(configs_dir)

    entries, skipped = {}, []
    for name in ALL_DATASETS:
        config = load_dataset_config(configs_dir, name)
        folder = raw_dataset_dir(raw_dir, name, config)
        splits = (load_converted(data_dir, name) if source == "corpus"
                  else convert_in_memory(folder, name, config, canonical))
        if splits is None:
            skipped.append(name)
            continue
        version_file = folder / "VERSION.txt"
        version = version_file.read_text().strip() if version_file.exists() else ""
        entry = dataset_entry(name, canonical, *splits, regions, version)
        entry["unreferenced_raw"] = unreferenced_raw_sessions(
            folder, sessions(splits[0]) | sessions(splits[1]))
        entries[name] = entry
        unref = entry["unreferenced_raw"]
        flag = f"  ⚠ unreferenced raw: {unref}" if unref else ""
        print(f"  {name:15s} train {entry['train_frames']:5,}/{entry['train_sessions']:3d} sess"
              f"  test {entry['test_frames']:5,}/{entry['test_sessions']:3d} sess"
              f"  direct {len(entry['direct']):2d}  trainable {len(entry['trainable']):2d}{flag}")

    if skipped:
        missing = "converted CSVs in the corpus" if source == "corpus" else "raw label CSVs"
        print(f"  skipped (no {missing}): {', '.join(skipped)}")

    if source == "corpus":
        out_dir = out_dir or data_dir
        stem = "dataset_inventory"
    else:
        out_dir = out_dir or repo_root() / "assets"
        stem = "corpus"
    written = write_tables(*make_table(entries), out_dir, stem)
    if source == "corpus":
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(
            {"canonical_keypoints": canonical, "datasets": entries}, indent=2) + "\n")
        written.append(json_path)
    for path in written:
        print(f"→ {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the corpus and write the dataset inventory.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source",   required=True, choices=SOURCES,
                        help="corpus: converted CSVs in data_dir (also writes the JSON); "
                             "raw: convert the raw datasets in memory")
    parser.add_argument("--raw_dir",  type=Path, default=None,
                        help="raw datasets (default: paths.yaml raw_dir)")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="converted corpus (default: paths.yaml data_dir; corpus source)")
    parser.add_argument("--out_dir",  type=Path, default=None,
                        help="output directory (default: data_dir for corpus, assets/ for raw)")
    args = parser.parse_args()

    paths    = load_paths()
    raw_dir  = args.raw_dir or Path(paths["raw_dir"])
    data_dir = args.data_dir or (Path(paths["data_dir"]) if args.source == "corpus" else None)
    run(args.source, raw_dir, data_dir, args.out_dir)


if __name__ == "__main__":
    main()
