#!/usr/bin/env bash
# Stage 5 full: Stage 2 with one inverse decoder D^-1 per model (8) instead of one per family.
#   gen_extra/<family>/<model>      7,680 new images per model except var24 (seed 10, this GPU)
#   gen/<family>/<model>            symlinks: stage2/gen (2,560, A40) + gen_extra (7,680) = 10,240 rows
#                                   (assemble_stage5_gen.py); var24's second source is the pilot's stage5/gen_extra
#   inv/<model>                     finetune_inverse.py --models <model> --max-per-model 9940 (9,840 train + 100
#                                   held out) with Stage 2's settings (stage2/inv/<family>/config.json); var24 reuses
#                                   stage5/inv/var24_full, trained with the same recipe
#   features/<model>/<family>.csv   extract_features.py --var-iters 0 with the model's own D^-1, all task images
# Before anything is generated, the derived seeds of all 7 new sets are checked (per family) against stage2/gen,
# stage3/gen and stage5/gen_extra. VAR runs first (fast), then RAR. Steps whose outputs exist are skipped, and every
# step resumes, so rerunning after an interruption continues where it stopped. Nothing is written outside $OUT.
#
#   SMOKE=1 bash scripts/run_stage5_full.sh   # 1 new shard per model, 1 epoch on 32 images, features --limit 64
#   tmux new -s stage5full 'bash scripts/run_stage5_full.sh'
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/workspace/venv/bin/python
CACHE=/workspace/cache
STAGE2=$CACHE/stage2
PILOT=$CACHE/stage5  # read only
PILOT_INV=$PILOT/inv/var24_full
PILOT_EXTRA_SHARDS=15
EXTRA_SEED=10
USED=("$STAGE2/gen" "$CACHE/stage3/gen" "$PILOT/gen_extra")
declare -A NEW=([var]="var16 var20 var30" [rar]="rarb rarl rarxl rarxxl")
declare -A ALL=([var]="var16 var20 var24 var30" [rar]="rarb rarl rarxl rarxxl")

if [[ "${SMOKE:-0}" == 1 ]]; then
  OUT=$CACHE/stage5full_smoke
  EXTRA_PER_MODEL=512  # one shard; shards must be 512 like stage2/gen's
  DATA_ARGS=(--max-per-model 40 --holdout-per-model 8)  # 32 training images
  EPOCHS=1
  FEAT_ARGS=(--limit 64)
else
  OUT=$CACHE/stage5full
  EXTRA_PER_MODEL=7680
  DATA_ARGS=(--max-per-model 9940 --holdout-per-model 100)  # 9,840 training images, as for the shared inverse
  EPOCHS=  # Stage 2's: RAR 10, VAR 5
  FEAT_ARGS=()
fi
N_EXTRA=$((EXTRA_PER_MODEL / 512))
mkdir -p "$OUT/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo "[$(ts)] $*"; }
family_of() { [[ $1 == var* ]] && echo var || echo rar; }
stage2_cfg() {  # stage2_cfg FAMILY KEY: one setting of Stage 2's shared inverse
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$STAGE2/inv/$1/config.json" "$2"
}
inv_ckpt() { [[ $1 == var24 ]] && echo "$PILOT_INV/final.pt" || echo "$OUT/inv/$1/final.pt"; }

preflight() {  # var24's pilot inverse must use the Stage 5 full recipe
  step "preflight: $PILOT_INV"
  "$PY" - "$PILOT_INV" "$PILOT/gen" "$STAGE2/inv/var/config.json" <<'EOF'
import json, sys
from pathlib import Path
inv, data, shared = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
cfg = json.loads((inv / "config.json").read_text())
assert (inv / "final.pt").is_file(), f"missing {inv / 'final.pt'}"
want = {"data": data, "models": ["var24"], "max_per_model": 9940, "holdout_per_model": 100,
        **{k: shared[k] for k in ("epochs", "batch_size", "lr", "step_size", "gamma", "seed", "no_task_train_eval")}}
bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
assert not bad, f"{inv}: (have, want) {bad}"
n = len(list((Path(data) / "var" / "var24").glob("shard_*.npz")))
assert n == 20, f"{data}/var/var24: {n} shards, expected 20"
print(f"{inv}: recipe matches Stage 5 full; {data}/var/var24 has {n} shards")
EOF
}

seed_check() {  # seed_check FAMILY
  step "seed check $1: ${NEW[$1]} with seed $EXTRA_SEED, $N_EXTRA shards each"
  # shellcheck disable=SC2086
  "$PY" scripts/assemble_stage5_gen.py --seed-check-only --family "$1" --labels ${NEW[$1]} --extra-seed "$EXTRA_SEED" \
    --expect-extra "$N_EXTRA" --seed-scope family --used "${USED[@]}" 2>&1 | tee -a "$OUT/logs/seed_check.log"
}

generate() {  # generate FAMILY: N_EXTRA new shards for each new model of FAMILY
  local fam=$1 m missing=()
  for m in ${NEW[$fam]}; do
    [[ -f "$OUT/gen_extra/$fam/$m/shard_$(printf %04d $((N_EXTRA - 1))).npz" ]] || missing+=("$m")
  done
  if (( ${#missing[@]} == 0 )); then
    step "skip generation $fam: last shards exist"
    return
  fi
  step "start generation $fam: ${missing[*]}, seed $EXTRA_SEED, $EXTRA_PER_MODEL images each -> $OUT/gen_extra"
  "$PY" scripts/generate_finetune_data.py --family "$fam" --models "${missing[@]}" --seed "$EXTRA_SEED" \
    --per-model "$EXTRA_PER_MODEL" --batch-size 64 --out "$OUT/gen_extra" 2>&1 | tee -a "$OUT/logs/generate_$fam.log"
  step "done generation $fam"
}

assemble() {  # assemble FAMILY: all 4 models of FAMILY into $OUT/gen
  local fam=$1 over=()
  [[ $fam == var ]] && over=(--extra-for var24 "$PILOT/gen_extra" "$PILOT_EXTRA_SHARDS")
  if [[ -f "$OUT/gen/$fam/config.json" ]]; then
    step "skip assembly $fam: $OUT/gen/$fam/config.json exists"
  else
    step "start assembly $fam -> $OUT/gen"
    # shellcheck disable=SC2086
    "$PY" scripts/assemble_stage5_gen.py --family "$fam" --labels ${ALL[$fam]} --extra "$OUT/gen_extra" \
      --expect-extra "$N_EXTRA" "${over[@]}" --seed-scope family --used "${USED[@]}" --out "$OUT/gen" \
      2>&1 | tee -a "$OUT/logs/assemble_$fam.log"
    step "done assembly $fam"
  fi
  if [[ $fam == var ]]; then  # var24 must be exactly the pilot's training set
    "$PY" - "$OUT/gen/var/var24" "$PILOT/gen/var/var24" <<'EOF'
import os, sys
from pathlib import Path
a, b = (Path(p) for p in sys.argv[1:])
la, lb = ({p.name: os.readlink(p) for p in d.iterdir()} for d in (a, b))
assert la == lb, f"{a} differs from {b}"
print(f"{a}: same {len(la)} shard links as {b}")
EOF
  fi
}

train() {  # train MODEL: its own D^-1 on $OUT/gen with Stage 2's settings
  local m=$1 fam dir="$OUT/inv/$1" epochs
  fam=$(family_of "$m")
  if [[ -f "$dir/final.pt" ]]; then
    step "skip train $m: $dir/final.pt exists"
    return
  fi
  epochs=${EPOCHS:-$(stage2_cfg "$fam" epochs)}
  step "start train $m: $epochs epochs -> $dir"
  "$PY" scripts/finetune_inverse.py --family "$fam" --models "$m" --data "$OUT/gen" "${DATA_ARGS[@]}" \
    --epochs "$epochs" --batch-size "$(stage2_cfg "$fam" batch_size)" --lr "$(stage2_cfg "$fam" lr)" \
    --step-size "$(stage2_cfg "$fam" step_size)" --gamma "$(stage2_cfg "$fam" gamma)" \
    --seed "$(stage2_cfg "$fam" seed)" --out "$dir" 2>&1 | tee -a "$OUT/logs/finetune_$m.log"
  step "done train $m"
}

features() {  # features MODEL: all task images under the model's own D^-1
  local m=$1 fam dir="$OUT/features/$1" ckpt
  fam=$(family_of "$m")
  ckpt=$(inv_ckpt "$m")
  if [[ -f "$dir/$fam.csv" ]]; then
    step "skip features $m: $dir/$fam.csv exists"
    return
  fi
  step "start features $m: $ckpt -> $dir"
  "$PY" scripts/extract_features.py --family "$fam" --var-iters 0 --inv-ckpt "$ckpt" "${FEAT_ARGS[@]}" --out "$dir" \
    2>&1 | tee -a "$OUT/logs/features_$m.log"
  step "done features $m"
}

preflight 2>&1 | tee -a "$OUT/logs/preflight.log"
seed_check var
seed_check rar
for fam in var rar; do
  generate "$fam"
  assemble "$fam"
  for m in ${NEW[$fam]}; do train "$m"; done
  for m in ${ALL[$fam]}; do features "$m"; done
done
step "stage 5 full done: $OUT"
