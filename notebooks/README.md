# notebooks

Analysis notebooks. Long jobs live in [scripts/](../scripts) and are run in tmux; the notebooks only read cached outputs, apart from small sanity checks.

**Kernel:** use the project environment, `/workspace/venv/bin/python` (Python 3.12, torch 2.6). If an import fails, install into the kernel with `%pip install <pkg>` (not `!pip`, which can target a different Python), then restart the kernel.

| notebook | purpose | needs | writes |
|---|---|---|---|
| [00_explore.ipynb](00_explore.ipynb) | data index, sample grid per label, and a round trip of one image through each autoencoder | data, weights, external repos, GPU | nothing |
| [001_stage_1.ipynb](001_stage_1.ipynb) | Stage 1 baseline: score selection, the nearest-family + threshold rule, val metrics, a test submission | `/workspace/cache/stage1/{rar,var}.csv` from `scripts/extract_features.py` | `/workspace/cache/stage1/submission_stage1.csv`, `metrics.json` |
| [002_stage_2.ipynb](002_stage_2.ipynb) | Stage 2: validates the generated data and the $D^{-1}$ fine-tuning, then reruns Stage 1's rule on the new features and compares | the outputs of `scripts/run_stage2.sh` in `/workspace/cache/stage2/`, plus Stage 1's | `/workspace/cache/stage2/submission_stage2.csv`, `metrics.json` |
| [003_stage_3.ipynb](003_stage_3.ipynb) | Stage 3: token habits. Can per-image statistics of the recovered tokens identify the size within a family? Pre-registered oracle / realistic / transfer checks, then the Stage 2 rule with the size step replaced where they pass | `/workspace/cache/stage3/gen/` and `tokens/` (see below), plus Stage 2's features, `inv/*/log.csv` and `metrics.json` | `/workspace/cache/stage3/gen_split.csv`; `submission_stage3.csv` only if val beats Stage 2 |
| [004_stage_4.ipynb](004_stage_4.ipynb) | Stage 4: generator likelihood. Does each size's transformer give the highest likelihood, under its own guided sampling distribution, to the tokens it generated? Pre-registered oracle / realistic / transfer checks, then the Stage 2 rule with the size step replaced where they pass | `/workspace/cache/stage4/scores/`, `classes/` and `tests/` (see below), plus Stage 3's `gen_split.csv` and tokens, and Stage 2's features and `metrics.json` | `submission_stage4.csv` only if val beats Stage 2 and the test images have been scored |

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

## 004_stage_4

Tests whether the sizes of a family can be told apart by the likelihood of an image's tokens under each size's transformer, with each repo's guided sampling distribution rebuilt in `tracer/likelihood.py`. The tokens are the true tokens (oracle) or the Stage 3 recovered tokens $Q(D^{-1}(x))$, and the class is the true one or the top-5 from a pretrained ImageNet classifier.

1. Check the likelihood code, estimate the classes, then score (see [scripts/README.md](../scripts/README.md#stage-4-generator-likelihood)). Needs Stage 3's `gen/` and `tokens/`:
   ```bash
   /workspace/venv/bin/python -m tracer.test_likelihood
   /workspace/venv/bin/python scripts/estimate_classes.py
   for f in rar var; do
     /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens true --class-mode true            # A
     /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens recovered --class-mode true       # B1
     /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens recovered --class-mode topk       # B2
     /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source task --tokens recovered --class-mode topk      # C, D
   done
   ```
2. Run all cells (CPU only, about 10 minutes). Classifiers (the own-sample rule and a logistic regression) are trained on generated train only, and RAR and VAR are analysed separately.
   - **1. Data and score checks:** provenance of the score files, the consistency tests of `tracer/likelihood.py`, and the generating × scoring model table.
   - **2. Oracle (A):** true tokens, true class; holdout accuracy, confusion matrices, pairwise AUCs.
   - **3. Recovered tokens (B1) and estimated classes (B2):** the same, side by side with A, for two ways of combining the top-5 classes.
   - **4. Transfer (C):** the B2 classifiers on the task's labelled train / val images (true family given), plus a separate task-train → task-val comparison and a sampling-settings diagnostic.
   - **5. End to end (D):** the Stage 2 rule with the size step replaced for families that passed all three checks. The test submission needs the test images scored first, into a separate directory:
     ```bash
     /workspace/venv/bin/python scripts/score_likelihood.py --family <family> --source task --tokens recovered \
         --class-mode topk --splits test --out /workspace/cache/stage4/scores_test
     ```
   - **6. Findings:** every pre-registered criterion per family and classifier, the A → B1 → B2 losses, and the conclusion.
