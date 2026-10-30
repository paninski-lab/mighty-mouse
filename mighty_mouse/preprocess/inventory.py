"""
Read-only inventory of the converted head-fixed corpus.

Validates that every per-dataset CSV has the canonical column layout (x, y, visible for
each keypoint of configs/keypoints.yaml, in order), derives the three per-dataset keypoint
masks (direct / trainable / evaluation), asserts train/test session disjointness, and reports
raw labeled-data directories that the conversion configs never reference (e.g. kondo's iter7_*
sets) without adding them.

Run before freezing a data version, so documentation cannot silently drift:

    python -m mighty_mouse.preprocess.inventory [--data_dir <corpus>] [--raw_dir <raw>]

Writes:
  docs/dataset_inventory.md          human-readable inventory (checked into the repo)
  <data_dir>/dataset_inventory.json  machine-readable manifest (travels with the data)

Masks:
  direct     keypoints with at least one visible=2 label in the train CSV
  trainable  direct, plus _left/_right partners of lateralized direct keypoints
             (hflip augmentation gives these training signal without direct labels)
  eval       keypoints with at least one visible=2 label in the test CSV
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.configs import default_configs_dir, load_canonical_keypoints
from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import COORDS
from mighty_mouse.paths import load_paths, repo_root

# ── per-dataset extraction ───────────────────────────────────────────────────

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if df.index[0] == df.index.name:  # stray header row from some writers
        df = df.iloc[1:]
    return df


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


def hflip_partner(name: str) -> str | None:
    if name.endswith("_left"):
        return name[:-5] + "_right"
    if name.endswith("_right"):
        return name[:-6] + "_left"
    return None


def trainable_mask(direct: set[str], canonical: list[str]) -> set[str]:
    """Direct keypoints plus hflip partners that exist in the canonical vocabulary."""
    partners = {p for k in direct if (p := hflip_partner(k)) and p in canonical}
    return direct | partners


def unreferenced_raw_sessions(name: str, referenced: set[str], raw_dir: Path) -> list[str]:
    """Raw labeled-data directories the conversion never consumed."""
    raw_labeled = raw_dir / name / "labeled-data"
    if not raw_labeled.is_dir():
        return []
    return sorted(
        d.name for d in raw_labeled.iterdir() if d.is_dir() and d.name not in referenced
    )


def dataset_entry(name: str, canonical: list[str], data_dir: Path, raw_dir: Path) -> dict:
    """Build the full inventory entry for one dataset.

    Args:
        name: dataset name
        canonical: keypoints from configs/keypoints.yaml
        data_dir: converted corpus
        raw_dir: raw datasets (for unreferenced labeled-data directories)

    Raises:
        ValueError: if a CSV's columns are not x, y, visible for each canonical keypoint in
            order, or if train and test share a session
    """
    train = read_csv(data_dir / f"CollectedData_{name}_train.csv")
    test  = read_csv(data_dir / f"CollectedData_{name}_test.csv")

    expected = [(k, c) for k in canonical for c in COORDS]
    for split, df in (("train", train), ("test", test)):
        if column_layout(df) != expected:
            raise ValueError(
                f"{name} {split} CSV columns deviate from the canonical layout (x, y, visible "
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

    direct = direct_mask(train)
    config = yaml.safe_load((repo_root() / "configs" / "datasets" / f"{name}.yaml").read_text())

    return {
        "train_frames":    len(train),
        "test_frames":     len(test),
        "train_sessions":  len(train_sessions),
        "test_sessions":   len(test_sessions),
        "direct":          sorted(direct),
        "trainable":       sorted(trainable_mask(direct, canonical)),
        "eval":            sorted(direct_mask(test)),
        "views":           config.get("sessions", {}),
        "excluded":        (config.get("exclude") or {}).get("keypoints") or [],
        "unreferenced_raw": unreferenced_raw_sessions(
            name, train_sessions | test_sessions, raw_dir
        ),
    }


# ── report generation ────────────────────────────────────────────────────────

def make_markdown(canonical: list[str], entries: dict[str, dict]) -> str:
    lines = [
        "# Dataset inventory",
        "",
        "Generated by `python -m mighty_mouse.preprocess.inventory` — do not edit by hand.",
        "",
        f"Canonical vocabulary: **{len(canonical)} keypoints**. "
        "Evaluation scores only labeled cells (visible == 2). Which datasets label a keypoint "
        "directly, and which train it only through hflip partners, is in the direct / "
        "trainable columns.",
        "",
        "Every dataset's test sessions are disjoint from its train sessions (asserted on "
        "every run of this command).",
        "",
        "| dataset | train frames | train sessions | test frames | test sessions "
        "| direct | trainable | eval |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in entries.items():
        lines.append(
            f"| `{name}` | {e['train_frames']:,} | {e['train_sessions']} "
            f"| {e['test_frames']:,} | {e['test_sessions']} "
            f"| {len(e['direct'])} | {len(e['trainable'])} | {len(e['eval'])} |"
        )
    total_train = sum(e["train_frames"] for e in entries.values())
    total_test  = sum(e["test_frames"] for e in entries.values())
    lines.append(f"| **total** | **{total_train:,}** | | **{total_test:,}** | | | | |")

    lines += ["", "## Pairwise direct-keypoint overlap", "",
              "| | " + " | ".join(f"`{n}`" for n in entries) + " |",
              "|---|" + "---|" * len(entries)]
    for a, ea in entries.items():
        row = [f"| `{a}`"]
        for _b, eb in entries.items():
            row.append(str(len(set(ea["direct"]) & set(eb["direct"]))))
        lines.append(" | ".join(row) + " |")

    for name, e in entries.items():
        lines += ["", f"## `{name}`", ""]
        if e["views"]:
            view_counts = pd.Series(list(e["views"].values())).value_counts().to_dict()
            summary = ", ".join(f"{v} sessions {k}-view" for k, v in sorted(view_counts.items()))
            lines.append(f"- **Views:** {summary}")
        else:
            lines.append("- **Views:** n/a")
        if e["excluded"]:
            lines.append(f"- **Excluded source keypoints:** {', '.join(e['excluded'])}")
        lines.append(f"- **Direct ({len(e['direct'])}):** " + ", ".join(e["direct"]))
        hflip_only = sorted(set(e["trainable"]) - set(e["direct"]))
        if hflip_only:
            lines.append(f"- **Hflip-only trainable:** {', '.join(hflip_only)}")
        if e["unreferenced_raw"]:
            lines.append(
                "- **⚠ Unreferenced raw labeled-data dirs (deliberately not converted):** "
                + ", ".join(e["unreferenced_raw"])
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the converted corpus and regenerate the dataset inventory.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="converted corpus (default: paths.yaml data_dir)")
    parser.add_argument("--raw_dir",  type=Path, default=None,
                        help="raw datasets (default: paths.yaml raw_dir)")
    args = parser.parse_args()

    paths    = load_paths() if args.data_dir is None or args.raw_dir is None else {}
    data_dir = args.data_dir or Path(paths["data_dir"])
    raw_dir  = args.raw_dir or Path(paths["raw_dir"])

    canonical = load_canonical_keypoints(default_configs_dir())
    entries   = {name: dataset_entry(name, canonical, data_dir, raw_dir) for name in ALL_DATASETS}

    manifest = {
        "canonical_keypoints": canonical,
        "datasets":            entries,
    }
    json_path = data_dir / "dataset_inventory.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")

    md_path = repo_root() / "docs" / "dataset_inventory.md"
    md_path.write_text(make_markdown(canonical, entries))

    print(f"canonical keypoints: {len(canonical)}")
    for name, e in entries.items():
        flag = f"  ⚠ unreferenced raw: {e['unreferenced_raw']}" if e["unreferenced_raw"] else ""
        print(f"  {name:15s} train {e['train_frames']:5,}/{e['train_sessions']:3d} sess"
              f"  test {e['test_frames']:5,}/{e['test_sessions']:3d} sess"
              f"  direct {len(e['direct']):2d}  trainable {len(e['trainable']):2d}"
              f"  eval {len(e['eval']):2d}{flag}")
    print(f"→ {md_path}")
    print(f"→ {json_path}")


if __name__ == "__main__":
    main()
