#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI —
local, sequential.

Loops over all combinations of csv_files × backbones × train_frames × seeds
and calls `litpose train` for each, then evaluates in-process.

For remote, parallel execution on Lightning AI, see train_sweep_lightning.py —
it shares all combo/naming/command-building logic with this script via
mighty_mouse.train, so the two only differ in how a command actually gets run.

Output lands at:
  <results_dir or --output_root>/<tag>/<losses_tag>[/sampling-T<T>]/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Merged CSVs (produced by build_dataset.py --tag <tag>) are evaluated against
all per-dataset test sets. Pixel error is NaN for keypoints absent from a dataset.

Usage examples:
  # dry run to preview all commands
  python scripts/train_sweep.py --dry_run \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2"

  # full sweep
  python scripts/train_sweep.py \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2" \\
      --backbones "vits_dino"

  # safe to re-run after interruption (a run with a rolling checkpoint continues)
  python scripts/train_sweep.py --skip_existing ...

  # the Mighty Mouse recipe (scripts/train_plan.sh builds these commands); needs lightning-pose
  # branch post_sub_lp_delta, which the script checks before training
  python scripts/train_sweep.py --config_file configs/model_zoominout.yaml \\
      --csv_files CollectedData_<tag>_train.csv --backbones vits_dinov3 \\
      --sampling_temperatures 2 --keep_checkpoints --skip_existing \\
      --extra_overrides "+model.head_hidden_channels=256;+training.resume_every_n_steps=500"
"""

import argparse
import subprocess
import sys
from pathlib import Path

from mighty_mouse.train import (
    RESULTS_DIR,
    build_combos,
    check_lightning_pose,
    csv_stem,
    evaluate_model,
    make_output_dir,
    make_train_command,
    needs_lp_branch,
    parse_semicolon_list,
)


def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep (local, sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_files", default="CollectedData_face+ibl+cheese_train.csv",
        help='semicolon-separated CSV filenames relative to data_dir',
    )
    parser.add_argument(
        "--train_frames", default="1",
        help='semicolon-separated frame counts; 1 = all frames',
    )
    parser.add_argument("--seeds", default="0", help='semicolon-separated rng seeds, e.g. "0;1;2"')
    parser.add_argument(
        "--backbones", default="resnet50_animal_ap10k", help='semicolon-separated backbone names',
    )
    parser.add_argument(
        "--losses_to_use", default="", help='comma-separated loss names; empty = supervised only',
    )
    parser.add_argument(
        "--sampling_temperatures", default="",
        help='semicolon-separated multi-dataset sampling temperatures, e.g. "1;2;inf"; '
             "1 or empty = the stock frame-proportional loader (no path component). Other "
             "values need lightning-pose branch post_sub_lp_delta.",
    )
    parser.add_argument(
        "--config_file", type=Path, default=None,
        help="training config (default: configs/model.yaml), e.g. configs/model_zoominout.yaml",
    )
    parser.add_argument(
        "--extra_overrides", default="",
        help='semicolon-separated Hydra overrides added to every run, e.g. '
             '"+model.head_hidden_channels=256" (not part of the output path)',
    )
    parser.add_argument(
        "--output_root", type=Path, default=None,
        help="write runs here instead of results_dir (same layout below it)",
    )
    parser.add_argument("--debug", action="store_true", help="Smoke-test run (6 steps)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip combos whose output dir already exists, unless it holds a rolling "
             "checkpoint (+training.resume_every_n_steps): those continue",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; only run evaluation on existing model dirs",
    )
    parser.add_argument(
        "--keep_checkpoints", action="store_true",
        help="Keep *.ckpt after evaluation (needed for zero-shot predictions and fine-tuning)",
    )
    parser.add_argument(
        "--stop_on_failure", action="store_true", help="Stop the sweep at its first failure",
    )
    args = parser.parse_args()
    if args.config_file is not None and not args.config_file.is_file():
        parser.error(f"--config_file {args.config_file} does not exist")

    csv_files    = parse_semicolon_list(args.csv_files)
    train_frames = parse_semicolon_list(args.train_frames)
    seeds        = parse_semicolon_list(args.seeds)
    backbones    = parse_semicolon_list(args.backbones)
    losses       = [x for x in args.losses_to_use.split(",") if x]
    temperatures = [t for t in args.sampling_temperatures.split(";") if t]
    extra        = [o for o in args.extra_overrides.split(";") if o]

    combos = build_combos(csv_files, backbones, train_frames, seeds, temperatures)
    print(f"Total jobs: {len(combos)}")

    def output_for(combo) -> Path:
        output = make_output_dir(*combo[:4], losses, combo[4])
        return args.output_root / output.relative_to(RESULTS_DIR) if args.output_root else output

    if args.skip_existing and not args.eval_only:
        combos = [
            c for c in combos
            if not output_for(c).exists() or (output_for(c) / "resume" / "resume.ckpt").exists()
        ]
        print(f"After skipping existing: {len(combos)} remaining")

    if not (args.dry_run or args.eval_only) and any(
        needs_lp_branch(c[4], args.config_file, extra) for c in combos
    ):
        check_lightning_pose()

    # collected rather than raised, so one bad combo doesn't stop the sweep, but the
    # process still exits non-zero instead of reporting success after a failure
    failures: list[str] = []
    for combo in combos:
        csv_file, backbone, train_frames_n, seed, temperature = combo
        output_dir = output_for(combo)
        label = f"{csv_stem(csv_file)} | tf={train_frames_n} | {backbone} | seed={seed}"
        if temperature:
            label += f" | T={temperature}"
        print(f"\n── {label}")

        if not args.eval_only:
            cmd = make_train_command(
                csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug,
                temperature, config_file=args.config_file, extra_overrides=extra,
            )
            print("   " + " ".join(cmd))

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR: training failed (exit {e.returncode}), skipping eval...")
                    failures.append(f"{label} (training exit {e.returncode})")
                    if args.stop_on_failure:
                        break
                    continue

        if not args.dry_run:
            if not output_dir.exists():
                print(f"  WARNING: output dir not found, skipping eval: {output_dir}")
                failures.append(f"{label} (no output dir)")
                continue
            try:
                evaluate_model(output_dir, csv_file, keep_checkpoints=args.keep_checkpoints)
            except Exception as e:
                print(f"  ERROR: evaluation failed: {e}")
                failures.append(f"{label} (evaluation: {e})")
                if args.stop_on_failure:
                    break

    if args.dry_run:
        print(f"\n(dry run — {len(combos)} commands printed, nothing executed)")
        return

    print(f"\nComplete: {len(combos) - len(failures)}/{len(combos)} succeeded.")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
