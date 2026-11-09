"""Where a training run lives under a results root, and the Mighty Mouse recipe.

A run directory is

    <root>/<tag>_train/<losses>/[sampling-T<T>/]tf<N>/<backbone>/seed<N>

where <root> is the results_dir or an area folder under it (trunks*, dedicated*). Everything that
builds or reads these paths (sweep scripts, train_plan.sh, evaluation) goes through here, so the
layout is defined once. No work at import time; `python -m mighty_mouse.train.layout` exposes the
same functions to shell scripts:

    run-dir ROOT TAG SEED    the recipe's run directory
    sweep-args               the train_sweep.py flags that select the recipe
"""

import argparse
from pathlib import Path

TRAIN_SUFFIX = "_train"

# the Mighty Mouse recipe's identity within a run directory (its other settings live in
# configs/model_zoominout.yaml and scripts/train/train_plan.sh)
RECIPE = {"losses": [], "temperature": 2, "train_frames": 1, "backbone": "vits_dinov3"}


def losses_tag(losses: list[str]) -> str:
    return "supervised" if not losses else "+".join(sorted(losses))


def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def temperature_tag(temperature) -> str | None:
    """Path/name component for a sampling temperature; None when the run is stock.

    T=1 (or unset) gets no component, so frame-proportional runs keep the paths they had
    before temperatures existed and --skip_existing keeps recognizing them: T=1 IS the stock
    loader.
    """
    if temperature in (None, 1, "1"):
        return None
    return f"sampling-T{temperature}"


def train_name(tag: str) -> str:
    """`<tag>_train`: the first component of a run directory (CollectedData_<tag>_train.csv)."""
    return f"{tag}{TRAIN_SUFFIX}"


def tag_of_train_name(name: str) -> str:
    """Inverse of `train_name`."""
    if not name.endswith(TRAIN_SUFFIX):
        raise ValueError(f"not a '<tag>{TRAIN_SUFFIX}' name: {name}")
    return name[: -len(TRAIN_SUFFIX)]


def run_dir(root: Path, train: str, losses: list[str], temperature, train_frames, backbone,
            seed) -> Path:
    """Run directory for one sweep combo.

    Args:
        root: results_dir, or an area folder under it
        train: `<tag>_train` (see `train_name`)
        losses: unsupervised losses used (empty = supervised)
        temperature: sampling temperature (None or 1 = stock, no path component)
        train_frames: train frames per dataset
        backbone: backbone name (made path-safe)
        seed: random seed
    """
    parts = [train, losses_tag(losses)]
    if (tag := temperature_tag(temperature)) is not None:
        parts.append(tag)
    parts += [f"tf{train_frames}", sanitize(backbone), f"seed{seed}"]
    return Path(root).joinpath(*parts)


def parse_run_dir(path: Path | str) -> dict:
    """Inverse of `run_dir`, read from the end of the path so no root is needed.

    Returns:
        dict with `train_name`, `losses_tag`, `temperature` (str, None if stock), `train_frames`,
        `backbone` (as sanitized) and `seed` (the latter two numbers as strings)

    Raises:
        ValueError: if the path does not end in a run-directory layout
    """
    parts = Path(path).parts
    if len(parts) < 5 or not parts[-1].startswith("seed") or not parts[-3].startswith("tf"):
        raise ValueError(f"not a run directory: {path}")
    i, temperature = -4, None
    if parts[i].startswith("sampling-T"):
        temperature = parts[i][len("sampling-T"):]
        i -= 1
    if len(parts) < -i + 1:
        raise ValueError(f"not a run directory: {path}")
    return {
        "train_name": parts[i - 1], "losses_tag": parts[i], "temperature": temperature,
        "train_frames": parts[-3][len("tf"):], "backbone": parts[-2],
        "seed": parts[-1][len("seed"):],
    }


def recipe_run_dir(area_dir: Path, tag: str, seed=0) -> Path:
    """The Mighty Mouse recipe's run directory for `tag` under an area folder."""
    return run_dir(area_dir, train_name(tag), **RECIPE, seed=seed)


def recipe_seed_dirs(area_dir: Path, tag: str) -> list[Path]:
    """Existing seed directories of the recipe's runs for `tag`, in seed order."""
    return sorted(recipe_run_dir(area_dir, tag).parent.glob("seed*"))


def sweep_args() -> str:
    """train_sweep.py flags that select the recipe."""
    return (f"--train_frames {RECIPE['train_frames']} --backbones {RECIPE['backbone']} "
            f"--sampling_temperatures {RECIPE['temperature']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run-dir", help="print the recipe's run directory")
    p.add_argument("root", type=Path, help="results_dir or an area folder under it")
    p.add_argument("tag")
    p.add_argument("seed")
    sub.add_parser("sweep-args", help="print the train_sweep.py flags that select the recipe")
    args = parser.parse_args()
    if args.command == "run-dir":
        print(recipe_run_dir(args.root, args.tag, args.seed))
    else:
        print(sweep_args())


if __name__ == "__main__":
    main()
