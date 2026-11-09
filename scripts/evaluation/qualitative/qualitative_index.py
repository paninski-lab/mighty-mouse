#!/usr/bin/env python
"""
Build <results_dir>/qualitative/INDEX.md from each delivery folder's README front matter.

Each qualitative delivery is a folder `<results_dir>/qualitative/<topic>-<MM-DD>/` whose
README.md starts with a front matter block; its keys are described in
mighty_mouse/evaluation/delivery.py.

    python scripts/evaluation/qualitative/qualitative_index.py            # current results_dir
    python scripts/evaluation/qualitative/qualitative_index.py --results_dir <path>
"""

import argparse
from pathlib import Path

from mighty_mouse.evaluation.delivery import collect_deliveries, index_markdown
from mighty_mouse.paths import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=None,
                    help="default: paths.yaml results_dir")
    args = ap.parse_args()
    results_dir = args.results_dir or Path(load_paths()["results_dir"])
    q = results_dir / "qualitative"
    deliveries = collect_deliveries(q)
    (q / "INDEX.md").write_text(index_markdown(results_dir.name, deliveries))
    n_indexed = sum(1 for _, fm in deliveries if fm)
    print(f"{n_indexed} indexed, {len(deliveries) - n_indexed} without front matter "
          f"-> {q / 'INDEX.md'}")


if __name__ == "__main__":
    main()
