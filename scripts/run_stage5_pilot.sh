#!/usr/bin/env bash
# Stage 5 pilot: fine-tune a var24-only D^-1 for 5 and for 20 epochs, then score the VAR generated holdout under
# E, shared, single5 and single20 (scripts/score_stage5_pilot.py). All settings other than --models, --epochs and
# --out are those of the shared inverse (/workspace/cache/stage2/inv/var/config.json). Steps whose outputs exist are
# skipped, and every step resumes, so rerunning after an interruption continues where it stopped.
#
#   tmux new -s stage5 'bash scripts/run_stage5_pilot.sh'
#   SMOKE=1 bash scripts/run_stage5_pilot.sh    # 1 epoch on 32 images, scoring --limit 16, in stage5_smoke
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/workspace/venv/bin/python
DATA=/workspace/cache/stage2/gen
# as in stage2/inv/var/config.json
COMMON=(--family var --data "$DATA" --models var24 --batch-size 16 --lr 5e-5 --step-size 2 --gamma 0.9 --seed 0)

if [[ "${SMOKE:-0}" == 1 ]]; then
  OUT=/workspace/cache/stage5_smoke
  EPOCHS_A=1 EPOCHS_B=1
  # 40 per model with the last 8 held out = 32 training images
  DATA_ARGS=(--max-per-model 40 --holdout-per-model 8)
  SCORE_ARGS=(--limit 16)
else
  OUT=/workspace/cache/stage5
  EPOCHS_A=5 EPOCHS_B=20
  DATA_ARGS=(--holdout-per-model 100)
  SCORE_ARGS=()
fi
mkdir -p "$OUT/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo "[$(ts)] $*"; }

train() {  # train NAME EPOCHS
  local dir="$OUT/inv/$1"
  if [[ -f "$dir/final.pt" ]]; then
    step "skip $1: $dir/final.pt exists"
    return
  fi
  step "start $1: finetune_inverse.py --epochs $2 -> $dir"
  "$PY" scripts/finetune_inverse.py "${COMMON[@]}" "${DATA_ARGS[@]}" --epochs "$2" --out "$dir" \
    2>&1 | tee -a "$OUT/logs/finetune_$1.log"
  step "done $1"
}

train var24_5ep "$EPOCHS_A"
train var24_20ep "$EPOCHS_B"

SCORES="$OUT/scores.csv"
INVERSES=(E shared single5 single20)
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
    2>&1 | tee -a "$OUT/logs/score.log"
  step "done scoring"
fi
step "stage 5 pilot done: $OUT"
