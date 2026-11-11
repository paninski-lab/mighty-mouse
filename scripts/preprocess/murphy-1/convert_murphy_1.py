#!/usr/bin/env python3
"""
Build _raw/murphy-1 (single merged project) from the three per-camera DeepLabCut
projects at _raw/_dlc/murphy-1/{cama,camb,camc}.

Each source project has the same 29-keypoint schema (4 digits x tip/mid/joint per paw,
`r_mid_A`/`l_mid_A`, `snout_A`, `mouth_tip_A`, `spout_tip_A`) and one
`labeled-data/<session>/CollectedData_tf.csv` per session folder. This script:

- keeps only the digit tips (`r{1-4}_tip_A`, `l{1-4}_tip_A`) plus `snout_A` and
  `mouth_tip_A`; drops mid/joint points, `r_mid_A`/`l_mid_A` and `spout_tip_A` (an object,
  not a body part). Output names are the source names; mapping to the canonical
  vocabulary is a stage-2 concern (`configs/datasets/murphy-1.yaml`).
- merges the three cameras into one project (`view_names: []`); sessions are renamed
  `<source folder>_<cam>` so the same folder name in two cameras can't collide.
- skips session folders without a CollectedData_tf.csv (and the `*.cache` files).
- removes duplicate frames: byte-identical images within a camera (e.g. `batch3_*`/`b2*`
  copies of a session, or `FS_Trial31` vs `FS_L2_..._Trial31`). The copy in the
  best-named folder is kept (a recognizable-subject folder over an ambiguous one, an
  unprefixed name over `batch3_`/`b2`/`b4_`).
- splits at the subject level (see `subject_of`), via mighty_mouse.preprocess.subject_split.
  Only the images referenced by a CSV row are copied.

Usage:
    python scripts/preprocess/murphy-1/convert_murphy_1.py [--seed N]
"""

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.subject_split import subject_split

CAMERAS = ["cama", "camb", "camc"]
SOURCE_CSV = "CollectedData_tf.csv"
SCORER = "scorer"

# Digit tips + snout + mouth_tip (the latter two may be removed later).
KEEP_KEYPOINTS = (
    [f"{side}{i}_tip_A" for side in "rl" for i in range(1, 5)] + ["snout_A", "mouth_tip_A"]
)

# Target ~10-15% of frames in test; midpoint, because the greedy subject split only ever
# overshoots its target (see mighty_mouse.preprocess.subject_split).
TEST_FRACTION = 0.125
TEST_RANGE = (0.10, 0.15)

BATCH_PREFIX = re.compile(r"^(batch3_|b2|b4_)")
# Animal IDs seen in session names, followed by '_', a (side)(cage) code like "L2"/"r3",
# a digit, or test/temp suffixes. "U"/"G" need the lookahead so "unknow_..." isn't "U".
SUBJECT = re.compile(r"^(fv1|g8s|ez|fb|fg|fj|fs|fu|az|sf|g|u)(?:_|[lr]?\d|test|teest|temp|$)")
# "sf_Trial*" sessions are treated as FS (probable typo): they must not split from FS.
SUBJECT_ALIAS = {"SF": "FS"}


def subject_of(session: str) -> str:
    """A session folder's subject (animal), from its source folder name (no `_<cam>` suffix).

    Strips `batch3_`/`b2`/`b4_` prefixes, matches the animal ID and uppercases it. Names with
    no recognizable ID (e.g. `unknow_Trial23`, `outliers`, `paw_camA`, `test`) each count as
    their own individual: the subject is the folder name itself.
    """
    m = SUBJECT.match(BATCH_PREFIX.sub("", session).lower())
    if m is None:
        return f"?{session}"
    subject = m.group(1).upper()
    return SUBJECT_ALIAS.get(subject, subject)


def folder_priority(session: str) -> tuple[bool, bool, str]:
    """Sort key for which duplicate copy to keep (smallest wins)."""
    return (subject_of(session).startswith("?"), bool(BATCH_PREFIX.match(session)), session)


def load_camera(source_dir: Path, cam: str) -> pd.DataFrame:
    """All CollectedData_tf.csv rows of one camera, indexed (session, frame), with only
    KEEP_KEYPOINTS columns and a `src` column holding the source image path."""
    frames = []
    for csv_path in sorted((source_dir / cam / "labeled-data").glob(f"*/{SOURCE_CSV}")):
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=[0, 1, 2])
        # session = the folder holding the CSV and images (the CSV's own session column can
        # disagree with it, e.g. in `test/`)
        df.index = pd.MultiIndex.from_arrays(
            [[csv_path.parent.name] * len(df), df.index.get_level_values(2)],
            names=["session", "frame"],
        )
        frames.append(df)
    df = pd.concat(frames)
    cols = [(SCORER, kp, c) for kp in KEEP_KEYPOINTS for c in ("x", "y")]
    df.columns = pd.MultiIndex.from_tuples([(SCORER, b, c) for _, b, c in df.columns])
    df = df[cols]
    df["src"] = [
        str(source_dir / cam / "labeled-data" / s / f) for s, f in df.index
    ]
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop byte-identical images within one camera, keeping the best-named folder's copy."""
    digests = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df["src"]]
    prio = [folder_priority(s) for s, _ in df.index]
    order = sorted(range(len(df)), key=lambda i: (digests[i], prio[i], df.index[i][1]))
    keep, seen = [], set()
    for i in order:
        if digests[i] not in seen:
            seen.add(digests[i])
            keep.append(i)
    return df.iloc[sorted(keep)]


def pick_split(counts: dict[str, int], seed: int | None) -> tuple[int, set[str], set[str]]:
    """Subject split with the given seed, or the first seed (from 0) landing in TEST_RANGE."""
    total = sum(counts.values())
    seeds = [seed] if seed is not None else range(1000)
    for s in seeds:
        train, test = subject_split(counts, s, TEST_FRACTION)
        frac = sum(counts[k] for k in test) / total
        if seed is not None or TEST_RANGE[0] <= frac <= TEST_RANGE[1]:
            return s, train, test
    raise RuntimeError("no seed landed in the target test range")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="subject-split seed (default: first in range)"
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="default: <raw_dir>/murphy-1")
    args = parser.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    source_dir = raw_dir / "_dlc" / "murphy-1"
    out_dir = args.out_dir or raw_dir / "murphy-1"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for cam in CAMERAS:
        df = load_camera(source_dir, cam)
        n_before = len(df)
        df = dedupe(df)
        print(f"{cam}: {n_before} frames, {n_before - len(df)} duplicates removed -> {len(df)}")
        df["subject"] = [subject_of(s) for s, _ in df.index]
        df.index = pd.MultiIndex.from_arrays(
            [[f"{s}_{cam}" for s, _ in df.index], df.index.get_level_values("frame")],
            names=["session", "frame"],
        )
        parts.append(df)
    combined = pd.concat(parts)
    # ambiguous folders are separate individuals per camera too
    combined["subject"] = [
        f"{subj}_{sess.rsplit('_', 1)[1]}" if subj.startswith("?") else subj
        for subj, (sess, _) in zip(combined["subject"], combined.index, strict=True)
    ]
    print(f"combined: {len(combined)} frames, {combined['subject'].nunique()} subjects")

    counts = combined["subject"].value_counts().to_dict()
    seed, train_subjects, test_subjects = pick_split(counts, args.seed)
    print(f"split seed {seed}: {len(train_subjects)} train / {len(test_subjects)} test subjects")
    print(f"  test subjects: {sorted(test_subjects)}")

    combined["path"] = [f"labeled-data/{s}/{f}" for s, f in combined.index]
    src = combined.pop("src")
    subjects = combined.pop("subject")
    combined = combined.set_index("path")
    src.index = subjects.index = combined.index
    combined.index.name = None

    for name, split_subjects in (("train", train_subjects), ("test", test_subjects)):
        split_df = combined.loc[subjects.isin(split_subjects)]
        csv_name = "CollectedData.csv" if name == "train" else "CollectedData_test.csv"
        split_df.to_csv(out_dir / csv_name)
        print(f"  {name}: {len(split_df)} frames ({len(split_df) / len(combined):.1%})")

    n = 0
    for path, src_path in src.items():
        dst = out_dir / path
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(src_path).read_bytes())
            n += 1
    print(f"  images: {n} copied")

    project = {"keypoint_names": KEEP_KEYPOINTS, "schema_version": 1, "view_names": []}
    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, default_flow_style=False, sort_keys=False)
    (out_dir / "videos").mkdir(exist_ok=True)
    print("Done.")


if __name__ == "__main__":
    main()
