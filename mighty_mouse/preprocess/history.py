"""
Corpus history: which label version of each dataset went into each corpus version.

assets/corpus_history.md is a matrix, one row per dataset and one column per corpus version
(`data/head-fixed_vN` -> column `vN`), each cell the dataset's label version (`VERSION.txt`) at the
time that corpus version was built. It is a record, not a description: assets/corpus.{md,csv}
describe the corpus as it is now, this file says how the corpus got there. The "why" of each
change belongs in the per-dataset CHANGELOG.md files.

    python -m mighty_mouse.preprocess.history [--data_dir <corpus>] [--raw_dir <raw>]

build_dataset.py calls `record_corpus_version` after any build that covers every dataset in
ALL_DATASETS. The version N comes from the directory name, so a corpus built in an unversioned
directory (data/head-fixed) is not recorded until it is frozen to head-fixed_vN; run this CLI
then. Recording a version again overwrites its column (a rebuild in place).

Cell markers:
  3    the dataset's label version (MW's scheme, `_raw/<dataset>/VERSION.txt`)
  h2   a dataset version from before that scheme (hash-keyed, e.g. cheese-3d@v2) that no
       `_raw/<dataset>/versions/` snapshot matches
  ?    the dataset was built but has no VERSION.txt
  –    the dataset was not in that corpus version
"""

import argparse
import re
from pathlib import Path

from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.paths import load_paths, repo_root

MISSING = "–"
UNKNOWN = "?"
VERSION_DIR_RE = re.compile(r"_v(\d+)$")

INTRO = ("Label version of each dataset (`_raw/<dataset>/VERSION.txt`) in each corpus version "
         "(`data/head-fixed_vN`). Updated when a full corpus is built "
         "(`scripts/preprocess/build_dataset.py`).")
LEGEND = ("`hN` = a pre-versioning, hash-keyed dataset version with no matching snapshot "
          "(e.g. cheese-3d@v2); `?` = no VERSION.txt; `–` = not in that corpus version.")


def history_path() -> Path:
    return repo_root() / "assets" / "corpus_history.md"


def corpus_version_of(data_dir: Path) -> str | None:
    """'v3' for a directory named head-fixed_v3, None if the name carries no version."""
    match = VERSION_DIR_RE.search(Path(data_dir).name)
    return f"v{match.group(1)}" if match else None


def dataset_versions(raw_dir: Path, datasets: list[str]) -> dict[str, str]:
    """Each dataset's current label version, UNKNOWN where it has no VERSION.txt."""
    from mighty_mouse.configs import default_configs_dir, load_dataset_config
    from mighty_mouse.preprocess.convert import raw_dataset_dir

    versions = {}
    for name in datasets:
        folder = raw_dataset_dir(raw_dir, name, load_dataset_config(default_configs_dir(), name))
        version_file = folder / "VERSION.txt"
        versions[name] = version_file.read_text().strip() if version_file.exists() else UNKNOWN
    return versions


def parse_history(text: str) -> dict[str, dict[str, str]]:
    """{corpus version: {dataset: cell}} from the markdown matrix (MISSING cells are dropped)."""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.startswith("|")]
    if len(rows) < 2:
        return {}
    columns = rows[0][1:]
    history = {col: {} for col in columns}
    for row in rows[2:]:  # rows[1] is the |---| separator
        for col, cell in zip(columns, row[1:], strict=False):
            if cell != MISSING:
                history[col][row[0]] = cell
    return history


def render_history(history: dict[str, dict[str, str]]) -> str:
    """The markdown matrix: columns in version order, rows in ALL_DATASETS order then the rest."""
    columns = sorted(history, key=lambda c: int(c[1:]))
    present = {name for col in columns for name in history[col]}
    names = [n for n in ALL_DATASETS if n in present] + sorted(present - set(ALL_DATASETS))
    lines = ["| dataset | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    lines += [f"| {name} | " + " | ".join(history[c].get(name, MISSING) for c in columns) + " |"
              for name in names]
    return f"{INTRO}\n\n" + "\n".join(lines) + f"\n\n{LEGEND}\n"


def record_corpus_version(
    data_dir: Path, raw_dir: Path, datasets: list[str] | None = None, out_path: Path | None = None
) -> Path | None:
    """Write the dataset versions of the corpus in `data_dir` into its column of the history.

    Args:
        data_dir: the corpus directory; must be named <something>_vN, else nothing is recorded
        raw_dir: raw datasets, where each VERSION.txt lives
        datasets: datasets in the corpus (default: ALL_DATASETS)
        out_path: the history file (default: assets/corpus_history.md)

    Returns:
        the file written, or None if `data_dir` has no version in its name
    """
    column = corpus_version_of(data_dir)
    if column is None:
        print(f"NOTE: {Path(data_dir).name} has no _vN suffix, so this corpus is not recorded in "
              "the corpus history. After freezing it to head-fixed_vN, run "
              "`python -m mighty_mouse.preprocess.history --data_dir <that directory>`.")
        return None
    out_path = out_path or history_path()
    history = parse_history(out_path.read_text()) if out_path.exists() else {}
    new = dataset_versions(raw_dir, datasets or ALL_DATASETS)
    if column in history and history[column] != new:
        print(f"NOTE: overwriting the existing {column} column of {out_path.name}")
    history[column] = new
    out_path.write_text(render_history(history))
    print(f"→ {out_path} ({column})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a corpus version's dataset versions in assets/corpus_history.md.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="the corpus directory, named *_vN (default: paths.yaml data_dir)")
    parser.add_argument("--raw_dir", type=Path, default=None,
                        help="root of raw datasets (default: from paths.yaml)")
    args = parser.parse_args()
    paths = load_paths()
    data_dir = args.data_dir or Path(paths["data_dir"])
    raw_dir = args.raw_dir or Path(paths["raw_dir"])
    record_corpus_version(data_dir, raw_dir)


if __name__ == "__main__":
    main()
