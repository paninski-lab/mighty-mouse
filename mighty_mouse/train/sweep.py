"""
Shared utilities for Lightning Pose training sweeps over the head-fixed dataset.

Used by both `scripts/train/train_sweep.py` (local, sequential) and
`scripts/train/train_sweep_lightning.py` (Lightning AI, parallel), so combo
generation, naming, and command-building stay identical across backends —
only *how* a command gets executed differs between the two scripts.

Also invocable standalone to run evaluation only. This exists because a
Lightning AI Job is an independent remote process: there's no "come back to
this process after training finishes" step like the local sweep script has,
so evaluation has to be its own chainable command:

    python -m mighty_mouse.train.sweep --output_dir <dir> --csv_file <CollectedData_..._train.csv>

Multi-dataset sampling temperature, the per-dataset zoom of configs/model_zoominout.yaml, the
nonlinear head and resumable runs need lightning-pose on branch `post_sub_lp_delta`
(paninski-lab/lightning-pose; LP_BRANCH below). Stock lightning-pose reads none of their config
keys and would silently train without them, so the sweeps call check_lightning_pose() before
any run that uses one. Drop the check once that branch is merged into lightning-pose main.
"""

import argparse
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths, repo_root

_paths      = load_paths()
DATA_DIR    = Path(_paths["data_dir"])
RESULTS_DIR = Path(_paths["results_dir"])
CONFIG_FILE = repo_root() / "configs" / "model.yaml"

# Every model is evaluated against each dataset's test CSV. Pixel error is NaN
# for keypoints absent from a given dataset; the plotting script handles filtering.
EVAL_DATASETS = ALL_DATASETS

# The lightning-pose branch (paninski-lab/lightning-pose) that reads these config keys; stock
# lightning-pose ignores them.
LP_BRANCH      = "post_sub_lp_delta"
LP_BRANCH_KEYS = ("sampling_temperature", "dataset_names", "imgaug_per_dataset_zoom",
                  "head_hidden_channels", "resume_every_n_steps")


# ── naming / output dirs ─────────────────────────────────────────────────────

def csv_stem(csv_file: str) -> str:
    """CollectedData_<tag>_train.csv -> <tag>_train"""
    return Path(csv_file).stem.split("_", 1)[1]


def infer_csv_file(output_dir: Path) -> str:
    """Inverse of csv_stem applied inside make_output_dir: the first path
    component under RESULTS_DIR is always csv_stem(csv_file), so the original
    train CSV filename can be recovered from output_dir alone."""
    rel = Path(output_dir).resolve().relative_to(RESULTS_DIR.resolve())
    return f"CollectedData_{rel.parts[0]}.csv"


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


def make_output_dir(csv_file, backbone, train_frames, seed, losses, temperature=None) -> Path:
    parts = [csv_stem(csv_file), losses_tag(losses)]
    if (tag := temperature_tag(temperature)) is not None:
        parts.append(tag)
    parts += [f"tf{train_frames}", sanitize(backbone), f"seed{seed}"]
    return RESULTS_DIR.joinpath(*parts)


def make_job_name(csv_file, backbone, train_frames, seed, losses, temperature=None) -> str:
    """Lightning job name for one sweep combo (unused locally, but shares the
    same inputs as make_output_dir so job name <-> output dir stay traceable)."""
    parts = [sanitize(csv_stem(csv_file)), sanitize(backbone), losses_tag(losses)]
    if (tag := temperature_tag(temperature)) is not None:
        parts.append(sanitize(tag))
    parts += [f"tf{train_frames}", f"s{seed}"]
    return "__".join(parts)


# ── sweep combo generation ───────────────────────────────────────────────────

def parse_semicolon_list(s: str) -> list[str]:
    return s.split(";")


def build_combos(csv_files, backbones, train_frames, seeds, temperatures=None) -> list[tuple]:
    """Cartesian product of one sweep, in (csv_file, backbone, train_frames, seed, temperature)
    order; temperatures=None means one stock frame-proportional run per combo."""
    return list(product(csv_files, backbones, train_frames, seeds, temperatures or [None]))


# ── lightning-pose branch check ──────────────────────────────────────────────

def needs_lp_branch(temperature=None, config_file=None, extra_overrides=()) -> bool:
    """Whether a run uses a setting only lightning-pose's LP_BRANCH reads.

    Args:
        temperature: the run's sampling temperature (None / 1 = stock)
        config_file: training config; any LP_BRANCH_KEYS key set to a non-null value counts
        extra_overrides: Hydra override strings; any that names an LP_BRANCH_KEYS key counts

    Returns:
        True if stock lightning-pose would silently ignore part of the run's recipe
    """
    if temperature_tag(temperature) is not None:
        return True
    if any(key in o for o in extra_overrides for key in LP_BRANCH_KEYS):
        return True
    if config_file is None:
        return False

    def set_keys(node) -> set[str]:
        if not isinstance(node, dict):
            return set()
        keys = {k for k, v in node.items() if v is not None}
        return keys.union(*(set_keys(v) for v in node.values()))

    cfg = yaml.safe_load(Path(config_file).read_text())
    return bool(set_keys(cfg) & set(LP_BRANCH_KEYS))


def check_lightning_pose() -> None:
    """Exit unless the installed lightning-pose is LP_BRANCH (paninski-lab/lightning-pose).

    Looks for two classes that exist only on that branch: the rolling-checkpoint callback
    and the temperature sampler.

    Raises:
        SystemExit: with instructions, if either is missing
    """
    try:
        from lightning_pose.callbacks import ResumeCheckpoint  # noqa: F401
        from lightning_pose.data.samplers import TemperatureSampler  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"This run uses settings only lightning-pose branch {LP_BRANCH} reads "
            f"({', '.join(LP_BRANCH_KEYS)}); stock lightning-pose would silently ignore them. "
            f"Check out that branch of paninski-lab/lightning-pose in the lightning-pose clone "
            f"this environment imports (git switch {LP_BRANCH}). ({e})"
        ) from e


# ── command building ─────────────────────────────────────────────────────────

def make_train_command(
    csv_file, backbone, train_frames, seed, losses, output_dir, debug=False,
    temperature=None, config_file=None, extra_overrides=None,
) -> list[str]:
    """Build the `litpose train ...` argv for one sweep combo.

    A non-stock temperature, or a config with per-dataset zoom, passes ALL_DATASETS as
    data.dataset_names (both look up each frame's dataset); both need lightning-pose LP_BRANCH.
    ``config_file`` replaces configs/model.yaml; ``extra_overrides`` (Hydra strings) are
    appended last, e.g. ``["+model.head_hidden_channels=256"]``.
    """
    lr = 5e-5 if "vit" in backbone else 1e-3
    losses_hydra = f"[{','.join(losses)}]"

    overrides = [
        f"data.data_dir={DATA_DIR}",
        f"data.csv_file={csv_file}",
        f"model.backbone={backbone}",
        f"model.losses_to_use={losses_hydra}",
        f"training.train_frames={train_frames}",
        f"training.rng_seed_data_pt={seed}",
        f"training.optimizer_params.learning_rate={lr}",
    ]

    # the temperature sampler and per-dataset zoom both look up each frame's dataset;
    # ++ adds or replaces a key, so any config works
    zoom = config_file is not None and "imgaug_per_dataset_zoom" in Path(config_file).read_text()
    if temperature_tag(temperature) is not None or zoom:
        overrides.append(f"++data.dataset_names=[{','.join(EVAL_DATASETS)}]")
    if temperature_tag(temperature) is not None:
        # quoted so "inf" stays a string
        overrides.append(f"++training.sampling_temperature='{temperature}'")

    if "vitb_sam" in backbone:
        overrides.append("training.train_batch_size=16")

    if debug:
        # Step-based keys, as the configs schedule by step (lightning-pose refuses a mix of step
        # and epoch keys): two batches per epoch, six steps, validation every two steps
        # (Lightning counts val_check_interval in steps only when check_val_every_n_epoch is
        # null); milestones must stay <= max_steps.
        overrides += [
            "+training.limit_train_batches=2",
            "training.min_steps=2",
            "training.max_steps=6",
            "training.unfreezing_step=1",
            "training.check_val_every_n_epoch=null",
            "++training.val_check_interval=2",
            "training.lr_scheduler_params.multisteplr.milestone_steps=[2]",
            "eval.predict_vids_after_training=false",
        ]

    overrides += list(extra_overrides or [])

    return (
        ["litpose", "train", str(config_file or CONFIG_FILE),
         "--output_dir", str(output_dir),
         "--overrides"] + overrides
    )


def make_eval_command(output_dir, csv_file) -> list[str]:
    """Build the standalone-evaluation argv for one sweep combo (chained onto
    make_train_command's output with `&&` for a single Lightning job command)."""
    return [
        "python", "-m", "mighty_mouse.train.sweep",
        "--output_dir", str(output_dir),
        "--csv_file", str(csv_file),
    ]


def make_extract_command() -> str:
    """Shell snippet that extracts DATA_DIR from a sibling `.tar` archive if
    DATA_DIR doesn't already exist, meant to run once at the start of each
    Lightning job (chained before make_train_command's output with `&&`).

    Each Lightning Job is an isolated snapshot of the launching Studio's own
    filesystem, not a shared mount multiple jobs write into concurrently — so
    unlike a HuggingFace-style dataset cache on shared storage, there's no race
    to guard against here. Every job independently checking "does this exist
    yet" and extracting its own local copy if not is safe to duplicate as-is.

    Also ensures DATA_DIR/videos exists: litpose's config always points
    `video_dir` at this path even though the head-fixed dataset trains from
    already-extracted frames, and the archive doesn't necessarily contain an
    entry for it if it was empty when tarred (`mkdir -p` after the
    test/extract step so it's created either way, whether or not extraction
    ran).

    Returns a shell string (not an argv list, unlike the other make_*_command
    functions) since it needs `||` — only meaningful for Lightning jobs; the
    local sequential script never calls this because DATA_DIR is expected to
    already exist on disk for local runs.
    """
    archive = f"{DATA_DIR}.tar"
    return (
        f'test -d "{DATA_DIR}" || tar -xf "{archive}" -C "{DATA_DIR.parent}"'
        f' && mkdir -p "{DATA_DIR / "videos"}"'
    )


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    output_dir: Path, csv_file: str, keep_checkpoints: bool = False,
    datasets: list[str] | None = None,
) -> None:
    """Evaluate a trained model against every per-dataset test CSV, then clean
    up the scratch prediction files litpose leaves behind in output_dir and,
    unless keep_checkpoints, delete model checkpoints (*.ckpt) — evaluation
    predictions/pixel-errors under output_dir/eval/<dataset>/ are what's kept
    long-term.

    Pixel errors count labeled cells only (visible == 2; tongue and lips are
    left out, see mighty_mouse/evaluation/comparison.py); comparison_metrics.json pools
    them. keep_checkpoints=True is needed for any model used after evaluation
    (zero-shot predictions, fine-tuning). datasets: evaluate only these test
    sets (default: every dataset in ALL_DATASETS)."""
    from lightning_pose.api import Model
    from PIL import Image

    from mighty_mouse.evaluation.comparison import comparison_summary, visible_errors

    output_dir = Path(output_dir)

    print("  Loading model...")
    model = Model.from_dir(output_dir)

    for eval_name in (datasets or EVAL_DATASETS):
        test_csv = DATA_DIR / f"CollectedData_{eval_name}_test.csv"

        if not test_csv.exists():
            print(f"  WARNING: {test_csv} not found — skipping {eval_name} eval")
            continue

        print(f"  Predicting on {eval_name} test set ({test_csv.name})...")
        result = model.predict_on_label_csv(
            csv_file=test_csv,
            data_dir=DATA_DIR,
            compute_metrics=False,
        )
        preds_df = result.predictions

        labels_df = read_labels_csv(test_csv)
        if labels_df.index[0] == labels_df.index.name:
            labels_df = labels_df.iloc[1:]

        error_df = visible_errors(labels_df, preds_df)
        preds_df = preds_df.loc[labels_df.index]
        diagonals = []
        for name in labels_df.index:
            with Image.open(DATA_DIR / name) as image:
                diagonals.append(float(np.linalg.norm(image.size)))
        summary = comparison_summary(error_df, np.asarray(diagonals))

        save_dir = output_dir / "eval" / eval_name
        save_dir.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(save_dir / "predictions.csv")
        error_df.to_csv(save_dir / "pixel_error.csv")
        (save_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

        mean_err = summary["mean_px"]
        mean_txt = "n/a (no labeled cell)" if mean_err is None else f"{mean_err:.2f} px"
        print(f"    Mean pixel error: {mean_txt}  →  {save_dir}")

    for fname in ("predictions.csv", "predictions_pixel_error.csv"):
        p = output_dir / fname
        if p.exists():
            p.unlink()
    image_preds = output_dir / "image_preds"
    if image_preds.exists():
        shutil.rmtree(image_preds)

    if keep_checkpoints:
        n_kept = len(list(output_dir.rglob("*.ckpt")))
        print(f"  Kept {n_kept} checkpoint file(s) (--keep_checkpoints)")
    else:
        n_deleted = 0
        for ckpt in output_dir.rglob("*.ckpt"):
            ckpt.unlink()
            n_deleted += 1
        print(f"  Deleted {n_deleted} checkpoint file(s)")


def _main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained head-fixed model against every per-dataset test CSV. "
                     "Also useful standalone for finishing sweep jobs that trained successfully "
                     "but stalled before their chained eval step completed.",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Model output_dir passed to `litpose train`",
    )
    parser.add_argument(
        "--csv_file", default=None,
        help="Train CSV filename the model was trained on (for logging only). "
             "Inferred from output_dir's path structure if omitted.",
    )
    parser.add_argument(
        "--keep_checkpoints", action="store_true",
        help="Keep *.ckpt files after evaluation (needed for zero-shot predictions and "
             "fine-tuning).",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated test sets to evaluate (default: every dataset in ALL_DATASETS).",
    )
    args = parser.parse_args()
    csv_file = args.csv_file or infer_csv_file(args.output_dir)
    datasets = [d for d in args.datasets.split(",") if d] if args.datasets else None
    print(f"output_dir: {args.output_dir}")
    print(f"csv_file:   {csv_file}")
    evaluate_model(args.output_dir, csv_file, keep_checkpoints=args.keep_checkpoints,
                   datasets=datasets)


if __name__ == "__main__":
    _main()
