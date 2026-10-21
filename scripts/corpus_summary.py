#!/usr/bin/env python
"""Write a one-row-per-dataset summary of the combined corpus.

For each dataset in ALL_DATASETS: train/test frame counts (one frame = one labeled image
row in the raw CollectedData[_test].csv, minus sessions the dataset config excludes),
the label version, the number of canonical keypoints the dataset provides, and that
count broken down by the region headers in configs/keypoints.yaml. Left/right pairs are
collapsed into one keypoint (lr-flip augmentation makes the side irrelevant).

The TOTAL row sums frames; its region columns count datasets with >0 keypoints there.

Writes assets/corpus.csv (opens in Google Sheets) and assets/corpus.md (renders on
GitHub). Run automatically at the end of scripts/bump_version.py; also safe to run by hand.
"""
import csv
import re
from pathlib import Path

from mighty_mouse.configs import default_configs_dir, load_dataset_config
from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import session_name
from mighty_mouse.paths import load_paths, repo_root

REGION_RE = re.compile(r"^\s*#\s*─+\s*(.+?)\s*─+\s*$")
KEYPOINT_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_]+)\s*(?:#.*)?$")
SIDE_RE = re.compile(r"_(left|right)$")


def collapse_side(name: str) -> str:
    return SIDE_RE.sub("", name)


def load_regions(configs_dir: Path) -> dict[str, str]:
    """Map each (uncommented) canonical keypoint to its region, per the comment headers."""
    region, out = None, {}
    for line in (configs_dir / "keypoints.yaml").read_text().splitlines():
        if m := REGION_RE.match(line):
            region = m.group(1)
        elif (m := KEYPOINT_RE.match(line)) and region:
            out[m.group(1)] = region
    return out


def count_frames(csv_path: Path, excluded_sessions: set[str]) -> int:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))[3:]
    return sum(1 for r in rows if r and session_name(r[0]) not in excluded_sessions)


def dataset_keypoints(cfg: dict, canonical: dict[str, str]) -> set[str]:
    """Canonical keypoints (side-collapsed) a dataset config provides."""
    excluded = set(cfg["exclude"]["keypoints"])
    out = set()
    for src, target in cfg["keypoints"].items():
        if src in excluded:
            continue
        if "{side}" in target:
            names = [target.format(side=s) for s in ("left", "right")]
        else:
            names = [target]
        for n in names:
            if n not in canonical:
                raise ValueError(f"{n!r} (from source keypoint {src!r}) is not in keypoints.yaml")
            out.add(collapse_side(n))
    return out


def build_table() -> tuple[list[str], list[list]]:
    configs_dir = default_configs_dir()
    canonical = load_regions(configs_dir)
    collapsed_region = {collapse_side(k): r for k, r in canonical.items()}
    regions = list(dict.fromkeys(canonical.values()))
    raw_dir = Path(load_paths()["raw_dir"])

    header = ["dataset", "version", "train_frames", "test_frames", "num_keypoints", *regions]
    rows = []
    for ds in ALL_DATASETS:
        cfg = load_dataset_config(configs_dir, ds)
        excluded = set(cfg["exclude"]["sessions"])
        kps = dataset_keypoints(cfg, canonical)
        version_file = raw_dir / ds / "VERSION.txt"
        rows.append([
            ds,
            version_file.read_text().strip() if version_file.exists() else "",
            count_frames(raw_dir / ds / "CollectedData.csv", excluded),
            count_frames(raw_dir / ds / "CollectedData_test.csv", excluded),
            len(kps),
            *[sum(collapsed_region[k] == r for k in kps) for r in regions],
        ])
    # frames are summed; each region column counts the datasets with >0 keypoints in it
    total = ["TOTAL", "", sum(r[2] for r in rows), sum(r[3] for r in rows), "",
             *[f"{sum(r[5 + i] > 0 for r in rows)} / {len(rows)}" for i in range(len(regions))]]
    return header, rows + [total]


def write_tables(header: list[str], rows: list[list], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "corpus.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (out_dir / "corpus.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = repo_root() / "assets"
    write_tables(*build_table(), out_dir)
    print(f"Wrote {out_dir / 'corpus.csv'} and {out_dir / 'corpus.md'}")


if __name__ == "__main__":
    main()
