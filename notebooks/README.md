# notebooks

Analysis notebooks. Long jobs live in [scripts/](../scripts) and are run in tmux; the notebooks only read cached outputs, apart from small sanity checks.

**Kernel:** use the project environment, `/workspace/venv/bin/python` (Python 3.12, torch 2.6). If an import fails, install into the kernel with `%pip install <pkg>` (not `!pip`, which can target a different Python), then restart the kernel.

| notebook | purpose | needs | writes |
|---|---|---|---|
| [00_explore.ipynb](00_explore.ipynb) | data index, sample grid per label, and a round trip of one image through each autoencoder | data, weights, external repos, GPU | nothing |
| [001_stage_1.ipynb](001_stage_1.ipynb) | Stage 1 baseline: score selection, the nearest-family + threshold rule, val metrics, a test submission | `/workspace/cache/stage1/{rar,var}.csv` from `scripts/extract_features.py` | `/workspace/cache/stage1/submission_stage1.csv`, `metrics.json` |
| [002_stage_2.ipynb](002_stage_2.ipynb) | Stage 2: validates the generated data and the $D^{-1}$ fine-tuning, then reruns Stage 1's rule on the new features and compares | the outputs of `scripts/run_stage2.sh` in `/workspace/cache/stage2/`, plus Stage 1's | `/workspace/cache/stage2/submission_stage2.csv`, `metrics.json` |
| [003_stage_3.ipynb](003_stage_3.ipynb) | Stage 3: token habits. Can per-image statistics of the recovered tokens identify the size within a family? Pre-registered oracle / realistic / transfer checks, then the Stage 2 rule with the size step replaced where they pass | `/workspace/cache/stage3/gen/` and `tokens/` (see below), plus Stage 2's features, `inv/*/log.csv` and `metrics.json` | `/workspace/cache/stage3/gen_split.csv`; `submission_stage3.csv` only if val beats Stage 2 |

## 00_explore

A self-contained exploration notebook. It defines its own early versions of the loaders; `tracer/common.py` has the maintained versions, which the scripts use. Run top to bottom.

## 001_stage_1

1. Generate the features first (see [scripts/README.md](../scripts/README.md)):
   ```bash
   /workspace/venv/bin/python scripts/extract_features.py --family rar
   /workspace/venv/bin/python scripts/extract_features.py --family var --var-iters 0
   ```
2. Run all cells. The notebook only reads the cached CSVs and needs no GPU.

The markdown cells give the method and the maths at each step, and the last cell summarises the findings. Everything is fit on train (no outliers); val is used only for evaluation, and q = 0.95 is fixed in advance.

## 002_stage_2

Stage 1 with the original encoder replaced by a fine-tuned inverse decoder $D^{-1}$ (Zhao et al., Eq. 6).

1. Run the pipeline (generation → fine-tuning → features, about 2.5 h; see [scripts/README.md](../scripts/README.md#stage-2-fine-tuning-the-inverse-decoder)):
   ```bash
   tmux new -s stage2 'bash scripts/run_stage2.sh'
   ```
2. Run all cells. Sections 1–3 check the run, and need a GPU with about 15 GB free:
   - **Generated data:** counts, sample images next to task images, stored images = $D(Q^{-1}(t_Z))$, and VAR's token capture.
   - **Fine-tuning curves** from `log.csv`.
   - **Original $E$ vs fine-tuned $D^{-1}$** on held-out generated images, per model size.

   Sections 4–8 repeat Stage 1 on the Stage 2 features (CPU only) and compare with it.

## 003_stage_3

Tests whether the sizes of a family can be told apart from per-image "token habits" (token histogram, entropy, distinct tokens, same-token neighbours, codebook rarity) of the tokens $Q(D^{-1}(x))$ recovered with the Stage 2 $D^{-1}$.

1. Generate a fresh set (seed 1, because $D^{-1}$ was trained on `stage2/gen`) and recover the tokens (see [scripts/README.md](../scripts/README.md#stage-3-token-habits)):
   ```bash
   /workspace/venv/bin/python scripts/generate_finetune_data.py --family rar --per-model 2560 --seed 1 --out /workspace/cache/stage3/gen
   /workspace/venv/bin/python scripts/generate_finetune_data.py --family var --per-model 2560 --seed 1 --out /workspace/cache/stage3/gen
   /workspace/venv/bin/python scripts/extract_tokens.py --family rar
   /workspace/venv/bin/python scripts/extract_tokens.py --family var
   ```
2. Run all cells (CPU only, about 4 minutes). Classifiers are trained on generated data only, and RAR and VAR are analysed separately.
   - **1. Data checks:** counts and class coverage, no overlap with `stage2/gen`, token recovery accuracy per model (VAR also per scale), and the 80/20 generated train / holdout split.
   - **2. Oracle:** habit features and token histograms from the **true** tokens; multinomial logistic regressions; holdout accuracy, confusion matrices and pairwise AUCs.
   - **3. Realistic:** the same from the **recovered** tokens, side by side with the oracle.
   - **4. Transfer:** the realistic classifiers on the task's labelled train / val images (true family given), plus a separate task-train → task-val comparison.
   - **5. End to end:** the Stage 2 rule with the size step replaced for families that passed all three checks; a test submission only if val improves.
   - **6. Findings:** every pre-registered criterion per family, and the conclusion.
