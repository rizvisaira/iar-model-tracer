#!/usr/bin/env bash
# Stage 2 pipeline, both families: generate fine-tuning data -> fine-tune D^-1 -> extract features with D^-1.
# Defaults are the shortened recipe; set environment variables to scale up (paper: PER_MODEL=12800, RAR_EPOCHS=50,
# VAR_EPOCHS=10, with a new OUT). Every step resumes, so rerunning after an interruption continues where it stopped.
#
#   tmux new -s stage2 'bash scripts/run_stage2.sh'
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/workspace/venv/bin/python}
OUT=${OUT:-/workspace/cache/stage2}
PER_MODEL=${PER_MODEL:-2560}  # generated images per model size (multiple of the 512-image shard size)
RAR_EPOCHS=${RAR_EPOCHS:-10}
VAR_EPOCHS=${VAR_EPOCHS:-5}
mkdir -p "$OUT/logs"

for fam in rar var; do
  "$PY" scripts/generate_finetune_data.py --family "$fam" --per-model "$PER_MODEL" --out "$OUT/gen" \
    2>&1 | tee -a "$OUT/logs/generate_$fam.log"
done

"$PY" scripts/finetune_inverse.py --family rar --epochs "$RAR_EPOCHS" --data "$OUT/gen" --out "$OUT/inv/rar" \
  2>&1 | tee -a "$OUT/logs/finetune_rar.log"
"$PY" scripts/finetune_inverse.py --family var --epochs "$VAR_EPOCHS" --data "$OUT/gen" --out "$OUT/inv/var" \
  2>&1 | tee -a "$OUT/logs/finetune_var.log"

for fam in rar var; do
  "$PY" scripts/extract_features.py --family "$fam" --var-iters 0 --inv-ckpt "$OUT/inv/$fam/final.pt" \
    --out "$OUT/features" 2>&1 | tee -a "$OUT/logs/features_$fam.log"
done
echo "stage 2 done: $OUT"
