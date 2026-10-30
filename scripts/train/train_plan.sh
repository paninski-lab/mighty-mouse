#!/bin/bash
# Training plan for the Mighty Mouse recipe on the current corpus (paths.yaml data_dir): which models
# exist, which are missing, and training the missing ones (see skills/train/train-plan/SKILL.md).
#
# Needs lightning-pose branch post_sub_lp_delta (paninski-lab/lightning-pose): temperature sampling,
# per-dataset zoom, the nonlinear head and resumable runs are not in stock lightning-pose, and
# scripts/train/train_sweep.py stops before training without them. Drop this note once the branch is merged.
#
# Recipe: ViT-S DINOv3, configs/model_zoominout.yaml, sampling temperature 2, nonlinear head
# (+model.head_hidden_channels=256), a rolling checkpoint every 500 steps (a run that is cut off
# continues where it stopped when the plan is run again), best checkpoint kept.
#   trunks (all, loo): 14k steps, backbone unfrozen at 3k, LR halved at 6k / 8k / 10k, 384 x 384 input
#   dedicated:         the config's schedule (12k steps, unfrozen at 1k) at 256 x 256
#
#   scripts/train/train_plan.sh plan [stages] [--seeds "0;1"]     # table: stage / tag / status per seed
#   scripts/train/train_plan.sh dry  [stages]                     # print the train_sweep commands
#   scripts/train/train_plan.sh show [stages]                     # print the script `run` would launch
#   scripts/train/train_plan.sh run  [stages] [--parallel N]      # train the missing runs (detached)
#   scripts/train/train_plan.sh status                            # tail the latest plan log
#
# Stages: all = the all-data trunk; dedicated = one model per dataset (its own frames only);
# loo = one leave-one-out trunk per dataset; dedicated:<ds> / loo:<ds> for one dataset.
# Default: all dedicated loo. Datasets: ALL_DATASETS; trunk tags: mighty_mouse.datasets.tag_of (a
# dedicated model trains on the dataset's own converted CSV). Each trunk's merged CSV must exist
# (`plan` shows NO CSV otherwise): build it with
#   python scripts/preprocess/build_dataset.py --tag <tag> --datasets <datasets> --n_frames -1
# Output: <results_dir>/trunks/<tag>_train/supervised/sampling-T2/tf1/vits_dinov3/seed<N>/ and
# <results_dir>/dedicated/<ds>_train/... After each loo run, scripts/evaluation/zeroshot_predict.py writes the
# left-out dataset's train-frame predictions to <run>/zeroshot/ (test frames: <run>/eval/<ds>/).
# --parallel N: at most N training runs on this machine at once, counting every
# scripts/train/train_sweep.py process (default 1). Log: <results_dir>/_logs/train_plan_<stamp>_<stages>.log.
set -u
cd "$(dirname "$0")/../.."
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
DATA=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['data_dir'])")
mapfile -t DATASETS < <(python -c "from mighty_mouse.datasets import ALL_DATASETS; print('\n'.join(ALL_DATASETS))")
SEEDS="0"; PARALLEL=1
cmd="${1:-plan}"; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --seeds)    SEEDS="$2"; shift 2;;
  --parallel) PARALLEL="$2"; shift 2;;
  *) STAGES="${STAGES:-} $1"; shift;; esac; done
STAGES="${STAGES:-all dedicated loo}"

LEAF="supervised/sampling-T2/tf1/vits_dinov3"
SWEEP="--config_file configs/model_zoominout.yaml --train_frames 1 --backbones vits_dinov3 --sampling_temperatures 2 --keep_checkpoints --skip_existing"
HEAD_OV="+model.head_hidden_channels=256;+training.resume_every_n_steps=500"
TRUNK_OV="training.unfreezing_step=3000;training.min_steps=14000;training.max_steps=14000;training.lr_scheduler_params.multisteplr.milestone_steps=[6000,8000,10000];data.image_resize_dims.height=384;data.image_resize_dims.width=384"

tag_of() { python -c "import sys; from mighty_mouse.datasets import tag_of; print(tag_of(sys.argv[1:]))" "$@"; }
all_tag=$(tag_of "${DATASETS[@]}") || exit 1
loo_tag() { local keep=(); for d in "${DATASETS[@]}"; do [ "$d" != "$1" ] && keep+=("$d"); done; tag_of "${keep[@]}"; }
# trained = training finished (best checkpoint) but no evaluation, e.g. a run cut off while evaluating
trained_only() { grep -q -E '"status": "(COMPLETED|EVALUATING)"' "$1/train_status.json" 2>/dev/null &&
  ls "$1"/tb_logs/*/version_*/checkpoints/*-best.ckpt >/dev/null 2>&1; }
status_of() { local d=$1; if ls "$d"/eval/*/predictions.csv >/dev/null 2>&1; then echo done
  elif [ -f "$d/resume/resume.ckpt" ]; then echo resumable
  elif trained_only "$d"; then echo trained
  elif [ -d "$d" ]; then echo PARTIAL; else echo missing; fi; }

# rows: stage area tag left_out ("-" unless loo)
rows() {
  for s in $STAGES; do case "$s" in
    all)         echo "all trunks $all_tag -";;
    dedicated)   for d in "${DATASETS[@]}"; do echo "dedicated dedicated $d -"; done;;   # its own CSV
    dedicated:*) echo "dedicated dedicated ${s#dedicated:} -";;
    loo)         for d in "${DATASETS[@]}"; do echo "loo trunks $(loo_tag "$d") $d"; done;;
    loo:*)       echo "loo trunks $(loo_tag "${s#loo:}") ${s#loo:}";;
    *) echo "unknown stage $s" >&2; exit 2;; esac; done
}

case "$cmd" in
  plan)
    echo "corpus: $(basename "$DATA")  results: $RESULTS  seeds: $SEEDS"
    printf "%-10s %-62s" stage tag; for s in ${SEEDS//;/ }; do printf " seed%-6s" "$s"; done; echo
    rows | while read -r stage area tag out; do
      printf "%-10s %-62s" "$stage" "$tag"
      if [ ! -f "$DATA/CollectedData_${tag}_train.csv" ]; then echo " NO CSV (build the tag first)"; continue; fi
      for s in ${SEEDS//;/ }; do printf " %-10s" "$(status_of "$RESULTS/$area/${tag}_train/$LEAF/seed$s")"; done; echo
    done ;;
  dry|run|show)
    # lightning-pose asserts <data_dir>/videos exists even for labeled-frame training
    [ -e "$DATA/videos" ] || { mkdir -p "$DATA/videos"; echo "created empty $DATA/videos"; }
    mkdir -p "$RESULTS/_logs"; LOG="$RESULTS/_logs/train_plan_$(date -u +%Y%m%d-%H%M%S)_$(echo $STAGES | tr -s " :" "-").log"
    script=$(mktemp)
    {
      echo "#!/bin/bash"
      echo "cd $(pwd)"
      echo 'running() { ps -eo args | grep -c "^[^ ]*python[0-9.]* scripts/train/train_sweep.py"; }'
    } > "$script"
    while read -r stage area tag out; do for s in ${SEEDS//;/ }; do
      dir="$RESULTS/$area/${tag}_train/$LEAF/seed$s"; st=$(status_of "$dir")
      [ -f "$DATA/CollectedData_${tag}_train.csv" ] || { echo "skip $stage $tag: NO CSV (build the tag first)" >&2; continue; }
      # done: nothing to do; PARTIAL: crashed without a rolling checkpoint, delete it to retrain
      case "$st" in done|PARTIAL) echo "skip $stage $tag seed$s: $st" >&2; continue;; esac
      ov="$HEAD_OV"; [ "$area" = trunks ] && ov="$ov;$TRUNK_OV"
      sweep="python scripts/train/train_sweep.py $SWEEP --csv_files CollectedData_${tag}_train.csv --seeds $s --output_root $RESULTS/$area --extra_overrides '$ov'"
      [ "$st" = trained ] && sweep="$sweep --eval_only"
      if [ "$cmd" = dry ]; then echo "$sweep --dry_run" >> "$script"; continue; fi
      post=""; [ "$out" != - ] && post="; python scripts/evaluation/zeroshot_predict.py --run $dir --dataset $out --data_dir $DATA"
      echo "while [ \"\$(running)\" -ge $PARALLEL ]; do sleep 60; done" >> "$script"
      echo "( echo \"[\$(date -u +%H:%M)] start $stage $tag seed$s\"; $sweep$post; echo \"[\$(date -u +%H:%M)] end $stage $tag seed$s\" ) & sleep 120" >> "$script"
    done; done < <(rows)
    [ "$cmd" != dry ] && echo "wait" >> "$script"
    echo 'echo "=== TRAIN PLAN COMPLETE ==="' >> "$script"
    if [ "$cmd" = dry ]; then bash "$script"
    elif [ "$cmd" = show ]; then cat "$script"
    else
      setsid nohup bash "$script" > "$LOG" 2>&1 < /dev/null &
      echo "launched pid $!; log $LOG"
    fi ;;
  status) tail -n 15 "$(ls -t "$RESULTS"/_logs/train_plan_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "no plan log";;
  *) echo "usage: $0 plan|dry|show|run|status [stages...] [--seeds \"0;1\"] [--parallel N]"; exit 2;;
esac
