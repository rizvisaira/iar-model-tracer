#!/usr/bin/env bash
# Stage 5 pilot: fine-tune var24-only D^-1s and score the VAR generated holdout (scripts/score_stage5_pilot.py).
#   var24_5ep, var24_20ep   stage2/gen var24 (2,560 images, 2,460 train), 5 and 20 epochs
#   var24_full              stage2/gen var24 + 7,680 new var24 images (seed 10, stage5/gen_extra), assembled with
#                           symlinks into stage5/gen (10,240 rows); first 9,940 used = 9,840 train + 100 holdout
# then scores E, shared, single5, single20 and singlefull, checking the holdout against stage2/gen and stage5/gen.
# All fine-tuning settings other than --data, --models, --max-per-model, --epochs and --out are those of the shared
# inverse (/workspace/cache/stage2/inv/var/config.json). Nothing is written to stage2/gen or stage3/gen. Steps whose
# outputs exist are skipped, and every step resumes, so rerunning after an interruption continues where it stopped.
#
#   tmux new -s stage5 'bash scripts/run_stage5_pilot.sh'
#   SMOKE=1 bash scripts/run_stage5_pilot.sh    # 1 epoch on 32 images, 1 new shard, scoring --limit 16, in stage5_smoke
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/workspace/venv/bin/python
STAGE2_DATA=/workspace/cache/stage2/gen
EXTRA_SEED=10  # derived seeds checked against stage2/gen (seed 0) and stage3/gen (seed 1) before generating
# as in stage2/inv/var/config.json
COMMON=(--family var --models var24 --batch-size 16 --lr 5e-5 --step-size 2 --gamma 0.9 --seed 0)

if [[ "${SMOKE:-0}" == 1 ]]; then
  OUT=/workspace/cache/stage5_smoke
  EPOCHS_A=1 EPOCHS_B=1 EPOCHS_FULL=1
  # 40 per model with the last 8 held out = 32 training images
  DATA_ARGS=(--max-per-model 40 --holdout-per-model 8)
  FULL_ARGS=("${DATA_ARGS[@]}")
  EXTRA_PER_MODEL=512
  SCORE_ARGS=(--limit 16)
else
  OUT=/workspace/cache/stage5
  EPOCHS_A=5 EPOCHS_B=20 EPOCHS_FULL=5
  DATA_ARGS=(--holdout-per-model 100)
  # 2,560 + 7,680 rows; the first 9,940 with the last 100 held out = 9,840 training images, as for shared
  FULL_ARGS=(--max-per-model 9940 --holdout-per-model 100)
  EXTRA_PER_MODEL=7680
  SCORE_ARGS=()
fi
GEN_EXTRA="$OUT/gen_extra"
N_EXTRA=$((EXTRA_PER_MODEL / 512))
mkdir -p "$OUT/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo "[$(ts)] $*"; }

train() {  # train NAME EPOCHS DATA [finetune_inverse.py args...]
  local name=$1 epochs=$2 data=$3 dir="$OUT/inv/$1"
  shift 3
  if [[ -f "$dir/final.pt" ]]; then
    step "skip $name: $dir/final.pt exists"
    return
  fi
  step "start $name: finetune_inverse.py --data $data --epochs $epochs -> $dir"
  "$PY" scripts/finetune_inverse.py "${COMMON[@]}" --data "$data" "$@" --epochs "$epochs" --out "$dir" \
    2>&1 | tee -a "$OUT/logs/finetune_$name.log"
  step "done $name"
}

train var24_5ep "$EPOCHS_A" "$STAGE2_DATA" "${DATA_ARGS[@]}"
train var24_20ep "$EPOCHS_B" "$STAGE2_DATA" "${DATA_ARGS[@]}"

last_extra="$GEN_EXTRA/var/var24/shard_$(printf %04d $((N_EXTRA - 1))).npz"
if [[ -f "$last_extra" ]]; then
  step "skip generation: $last_extra exists"
else
  step "start generation: var24 seed $EXTRA_SEED, $EXTRA_PER_MODEL images -> $GEN_EXTRA"
  "$PY" scripts/assemble_stage5_gen.py --seed-check-only --extra-seed "$EXTRA_SEED" --expect-extra "$N_EXTRA" \
    2>&1 | tee -a "$OUT/logs/generate_extra.log"
  "$PY" scripts/generate_finetune_data.py --family var --models var24 --seed "$EXTRA_SEED" \
    --per-model "$EXTRA_PER_MODEL" --batch-size 64 --out "$GEN_EXTRA" 2>&1 | tee -a "$OUT/logs/generate_extra.log"
  step "done generation"
fi

if [[ -f "$OUT/gen/var/config.json" ]]; then
  step "skip assembly: $OUT/gen/var/config.json exists"
else
  step "start assembly -> $OUT/gen"
  "$PY" scripts/assemble_stage5_gen.py --extra "$GEN_EXTRA" --expect-extra "$N_EXTRA" --out "$OUT/gen" \
    2>&1 | tee -a "$OUT/logs/assemble.log"
  step "done assembly"
fi

train var24_full "$EPOCHS_FULL" "$OUT/gen" "${FULL_ARGS[@]}"

SCORES="$OUT/scores.csv"
INVERSES=(E shared single5 single20 singlefull)
if "$PY" - "$SCORES" "${INVERSES[@]}" <<'EOF'
import json, sys
from pathlib import Path
scores = Path(sys.argv[1])
cfg = scores.with_name(scores.stem + ".config.json")
done = scores.is_file() and cfg.is_file() and set(sys.argv[2:]) <= set(json.loads(cfg.read_text())["inverses"])
sys.exit(0 if done else 1)
EOF
then
  step "skip scoring: $SCORES has ${INVERSES[*]}"
else
  step "start scoring -> $SCORES"
  "$PY" scripts/score_stage5_pilot.py "${SCORE_ARGS[@]}" --out "$SCORES" \
    --single "single5=$OUT/inv/var24_5ep/final.pt" --single "single20=$OUT/inv/var24_20ep/final.pt" \
    --single "singlefull=$OUT/inv/var24_full/final.pt" \
    --overlap-data "$STAGE2_DATA" --overlap-data "$OUT/gen" \
    2>&1 | tee -a "$OUT/logs/score.log"
  step "done scoring"
fi
step "stage 5 pilot done: $OUT"
