#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI —
Lightning AI, parallel.

Shares all combo/naming/command-building logic with train_sweep.py via
mighty_mouse.train.sweep — the CLI args are identical. The only real difference:
each combo is launched as an independent Lightning Job instead of run in a
sequential loop, and since a Job is a fresh remote process (no "come back to
this process after training" step), data extraction, training, and evaluation
are chained into one shell command per job:

    test -d <data_dir> || tar -xf <data_dir>.tar -C <data_dir's parent>
    && mkdir -p <data_dir>/videos
    && litpose train ...
    && python -m mighty_mouse.train.sweep --output_dir ... --csv_file ...

Each Job is an isolated snapshot of the launching Studio's own filesystem, not
a shared mount multiple jobs write into — so every job independently checking
"does this exist yet, if not extract it" is safe (see make_extract_command's
docstring in mighty_mouse/train/sweep.py). This is a different situation from a
HuggingFace-style dataset cache on genuinely shared storage, which would need
locking or pre-extraction to avoid concurrent-write races.

Output lands at the same place as the local script:
  <results_dir>/<tag>/<losses_tag>[/sampling-T<T>]/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Prerequisites (not handled by this script):
  - This machine's paths.yaml must have data_dir pointing at wherever the
    Studio keeps a `<data_dir>.tar` archive of data/head-fixed_vN (e.g. built
    with `tar -cf head-fixed_v2.tar -C data head-fixed_v2` and uploaded to the
    Studio once) — creating and uploading that archive isn't handled here.
  - results_dir should point at persistent storage that outlives an individual
    job's isolated filesystem (e.g. a teamspace-mounted drive), since results
    need to survive after the job's compute is torn down.
  - The Studio (or whatever environment `lightning_sdk` launches jobs into)
    needs `litpose` on PATH and `mighty_mouse` installed (`pip install -e .`).
  - --sampling_temperatures other than 1, configs/model_zoominout.yaml and the
    nonlinear-head / resume overrides need lightning-pose branch post_sub_lp_delta
    in the Studio; the script checks it before launching any job that uses them.

Run from within a Lightning AI studio:
    python scripts/train/train_sweep_lightning.py \\
        --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl_train.csv" \\
        --train_frames "200;400;600" \\
        --seeds "0;1;2" \\
        --backbones "vits_dino" \\
        --machine L4

Run from outside Lightning AI (set LIGHTNING_API_KEY env var first):
    LIGHTNING_API_KEY=<key> python scripts/train/train_sweep_lightning.py ...

  # dry run to preview all jobs without launching anything
  python scripts/train/train_sweep_lightning.py --dry_run ...
"""

import argparse
import shlex
import time
from pathlib import Path

from mighty_mouse.train.sweep import (
    build_combos,
    check_lightning_pose,
    make_eval_command,
    make_extract_command,
    make_job_name,
    make_output_dir,
    make_train_command,
    needs_lp_branch,
    parse_semicolon_list,
)


def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep (Lightning AI, parallel)",
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
    parser.add_argument("--debug", action="store_true", help="Smoke-test run (6 steps)")
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without launching")
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip combos whose output dir already exists",
    )
    parser.add_argument(
        "--machine", default="T4_SMALL", help="Lightning Machine type, e.g. T4_SMALL, A10G, L4",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=30, help="Seconds between job-status polls",
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
        "--keep_checkpoints", action="store_true",
        help="Keep *.ckpt after evaluation (needed for zero-shot predictions and fine-tuning)",
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

    if args.skip_existing:
        combos = [c for c in combos if not make_output_dir(*c[:4], losses, c[4]).exists()]
        print(f"After skipping existing: {len(combos)} remaining")

    # Identical for every job (depends only on data_dir, not the combo) — computed once,
    # but still has to run inside each job since it acts on that job's own filesystem.
    extract_cmd = make_extract_command()

    jobs_spec = []
    for csv_file, backbone, train_frames_n, seed, temperature in combos:
        output_dir = make_output_dir(csv_file, backbone, train_frames_n, seed, losses, temperature)
        name       = make_job_name(csv_file, backbone, train_frames_n, seed, losses, temperature)
        train_cmd  = make_train_command(
            csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug,
            temperature, config_file=args.config_file, extra_overrides=extra,
        )
        eval_cmd   = make_eval_command(output_dir, csv_file)
        if args.keep_checkpoints:
            eval_cmd.append("--keep_checkpoints")
        # shlex.join: overrides such as sampling_temperature='inf' carry shell quotes
        full_cmd   = extract_cmd + " && " + shlex.join(train_cmd) + " && " + shlex.join(eval_cmd)
        jobs_spec.append((name, output_dir, full_cmd))

    if args.dry_run:
        print("\n--- Job list ---")
        for name, _output_dir, cmd in jobs_spec:
            print(f"\n{name}:\n  {cmd}")
        print(f"\n(dry run — {len(jobs_spec)} jobs printed, nothing launched)")
        return

    # the jobs run on this Studio's lightning-pose
    if any(needs_lp_branch(c[4], args.config_file, extra) for c in combos):
        check_lightning_pose()

    from lightning_sdk import Job, Machine, Studio

    machine = getattr(Machine, args.machine)
    studio  = Studio()

    jobs = {}
    for name, output_dir, cmd in jobs_spec:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Launching: {name}")
        job = Job.run(command=cmd, name=name, machine=machine, studio=studio)
        jobs[name] = job
        time.sleep(2)

    print(f"\nMonitoring {len(jobs)} jobs...")
    while True:
        statuses: dict[str, int] = {}
        for j in jobs.values():
            s = str(j.status)
            statuses[s] = statuses.get(s, 0) + 1
        print(f"  {statuses}")
        if not any(s in ("Running", "Pending") for s in statuses):
            break
        time.sleep(args.poll_interval)

    failed = [n for n, j in jobs.items() if str(j.status) == "Failed"]
    print(f"\nComplete: {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print("Failed jobs:")
        for n in failed:
            print(f"  {n}")


if __name__ == "__main__":
    main()
