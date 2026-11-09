"""
Shared utilities for Lightning Pose training sweeps over the head-fixed dataset.

Combo generation, naming, and command-building live here so they stay identical across the
local (sequential) and Lightning AI (parallel) backends — only *how* a command gets executed
differs.

Evaluation lives in mighty_mouse/evaluation/run.py; a Lightning AI Job has no "come back to
this process after training finishes" step, so make_eval_command builds it as a chainable
command.

Multi-dataset sampling temperature, the per-dataset zoom of configs/model_zoominout.yaml, the
nonlinear head and resumable runs need lightning-pose on branch `post_sub_lp_delta`
(paninski-lab/lightning-pose; LP_BRANCH below). Stock lightning-pose reads none of their config
keys and would silently train without them, so the sweeps call check_lightning_pose() before
any run that uses one. Drop the check once that branch is merged into lightning-pose main.
"""

from itertools import product
from pathlib import Path

import yaml

from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.paths import repo_root
from mighty_mouse.train.layout import (
    losses_tag,
    run_dir,
    sanitize,
    temperature_tag,
)

# The lightning-pose branch (paninski-lab/lightning-pose) that reads these config keys; stock
# lightning-pose ignores them.
LP_BRANCH      = "post_sub_lp_delta"
LP_BRANCH_KEYS = ("sampling_temperature", "dataset_names", "imgaug_per_dataset_zoom",
                  "head_hidden_channels", "resume_every_n_steps")


# ── naming / output dirs ─────────────────────────────────────────────────────

def csv_stem(csv_file: str) -> str:
    """CollectedData_<tag>_train.csv -> <tag>_train"""
    return Path(csv_file).stem.split("_", 1)[1]


def make_output_dir(
    root: Path, csv_file, backbone, train_frames, seed, losses, temperature=None,
) -> Path:
    """Run directory for one sweep combo under `root` (the results_dir or an --output_root)."""
    return run_dir(root, csv_stem(csv_file), losses, temperature, train_frames, backbone, seed)


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
    csv_file, backbone, train_frames, seed, losses, output_dir, data_dir, debug=False,
    temperature=None, config_file=None, extra_overrides=None,
) -> list[str]:
    """Build the `litpose train ...` argv for one sweep combo.

    A non-stock temperature, or a config with per-dataset zoom, passes ALL_DATASETS as
    data.dataset_names (both look up each frame's dataset); both need lightning-pose LP_BRANCH.
    ``data_dir`` is the corpus directory holding the CSVs and labeled-data/. ``config_file``
    replaces configs/model.yaml; ``extra_overrides`` (Hydra strings) are
    appended last, e.g. ``["+model.head_hidden_channels=256"]``.
    """
    lr = 5e-5 if "vit" in backbone else 1e-3
    losses_hydra = f"[{','.join(losses)}]"

    overrides = [
        f"data.data_dir={data_dir}",
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
        overrides.append(f"++data.dataset_names=[{','.join(ALL_DATASETS)}]")
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
        ["litpose", "train", str(config_file or repo_root() / "configs" / "model.yaml"),
         "--output_dir", str(output_dir),
         "--overrides"] + overrides
    )


def make_eval_command(output_dir) -> list[str]:
    """Build the standalone-evaluation argv for one sweep combo (chained onto
    make_train_command's output with `&&` for a single Lightning job command)."""
    return [
        "python", str(repo_root() / "scripts" / "evaluation" / "evaluate_run.py"),
        "--run", str(output_dir),
    ]


def make_extract_command(data_dir: Path) -> str:
    """Shell snippet that extracts `data_dir` from a sibling `.tar` archive if
    it doesn't already exist, meant to run once at the start of each
    Lightning job (chained before make_train_command's output with `&&`).

    Each Lightning Job is an isolated snapshot of the launching Studio's own
    filesystem, not a shared mount multiple jobs write into concurrently — so
    unlike a HuggingFace-style dataset cache on shared storage, there's no race
    to guard against here. Every job independently checking "does this exist
    yet" and extracting its own local copy if not is safe to duplicate as-is.

    Also ensures data_dir/videos exists: litpose's config always points
    `video_dir` at this path even though the head-fixed dataset trains from
    already-extracted frames, and the archive doesn't necessarily contain an
    entry for it if it was empty when tarred (`mkdir -p` after the
    test/extract step so it's created either way, whether or not extraction
    ran).

    Returns a shell string (not an argv list, unlike the other make_*_command
    functions) since it needs `||` — only meaningful for Lightning jobs; the
    local sequential script never calls this because data_dir is expected to
    already exist on disk for local runs.
    """
    data_dir = Path(data_dir)
    archive = f"{data_dir}.tar"
    return (
        f'test -d "{data_dir}" || tar -xf "{archive}" -C "{data_dir.parent}"'
        f' && mkdir -p "{data_dir / "videos"}"'
    )
