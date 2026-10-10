# IAR Model Tracer

Attribute each image to the image autoregressive (IAR) model that generated it, or flag it as an outlier. The 9 labels are RAR-B / L / XL / XXL (`rarb rarl rarxl rarxxl`), VAR-d16 / d20 / d24 / d30 (`var16 var20 var24 var30`) and `outlier`. The approach builds on the autoencoder-based provenance signals of Zhao et al., *Data Provenance for Image Auto-Regressive Generation* (ICLR 2026).

## Repository layout

| directory | contents | docs |
|---|---|---|
| [tracer/](tracer) | Python package: data index, autoencoder and generator loaders, provenance signals, inverse-decoder fine-tuning, token recovery and token-habit features, generator likelihood, decision rule, metrics | [tracer/README.md](tracer/README.md) |
| [scripts/](scripts) | long-running jobs (feature extraction, fine-tuning data generation, inverse-decoder fine-tuning, token extraction, class estimation, likelihood scoring), run in tmux | [scripts/README.md](scripts/README.md) |
| [notebooks/](notebooks) | exploration and per-stage analysis | [notebooks/README.md](notebooks/README.md) |

Data, model weights, extracted features and predictions live outside the repo and are never committed.

## Setup

The paths are fixed in [tracer/common.py](tracer/common.py):

| path | contents | source |
|---|---|---|
| `/workspace/data` | `train/<label>/*.png` (800), `val/<label>/*.png` (450, including `outlier/`), `test/*.png` (9,800) | `huggingface.co/datasets/maitri01/model_tracer` (`Dataset.zip`) |
| `/workspace/external/1d-tokenizer` | RAR tokenizer code | `github.com/bytedance/1d-tokenizer` |
| `/workspace/external/VAR` | VAR VQVAE code | `github.com/FoundationVision/VAR` |
| `/workspace/weights/rar/maskgit-vqgan-imagenet-f16-256.bin` | RAR tokenizer (shared by all RAR sizes) | HF `fun-research/TiTok` |
| `/workspace/weights/rar/rar_{b,l,xl,xxl}.bin` | RAR generators | HF `yucornetto/RAR` |
| `/workspace/weights/var/vae_ch160v4096z32.pth` | VAR VQVAE (shared by all VAR depths) | HF `FoundationVision/var` |
| `/workspace/weights/var/var_d{16,20,24,30}.pth` | VAR generators | HF `FoundationVision/var` |
| `/workspace/cache` | features, predictions, metrics (created by the scripts) | |
| `/workspace/hf_cache` | `HF_HOME` | |

Stage 1 uses only the two tokenizers. Stage 2 also samples the eight generators, to create fine-tuning data. Stage 3 samples them again with a new seed and reuses Stage 2's $D^{-1}$. Stage 4 scores tokens under the eight generators and also uses a torchvision ImageNet classifier (`convnext_large`, downloaded once to `/workspace/weights/torchvision/`).

Python environment: `/workspace/venv` (Python 3.12) with `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, plus the dependencies of the two external repos. Use it for both scripts and notebook kernels.

## Quick start (Stage 1)

```bash
cd /workspace/iar-model-tracer
# 1. Features, once per tokenizer family (GPU; about 7 and 12.5 minutes on an A40)
/workspace/venv/bin/python scripts/extract_features.py --family rar
/workspace/venv/bin/python scripts/extract_features.py --family var --var-iters 0
# 2. Decision rule, val metrics, test predictions: run notebooks/001_stage_1.ipynb
```

This writes `/workspace/cache/stage1/{rar,var}.csv`, `submission_stage1.csv` (`image_name,label`, one row per test image) and `metrics.json`.

## Stage 2: fine-tuned inverse decoder

Stage 1, with each family's encoder replaced by an inverse decoder $D^{-1}$ fine-tuned on the family's own generated images (Zhao et al., Eq. 6).

```bash
# Generate (tokens, image) pairs -> fine-tune D^-1 -> features with D^-1, both families (about 2.5 h, shortened recipe)
tmux new -s stage2 'bash scripts/run_stage2.sh'
# Then validate the run and evaluate it: notebooks/002_stage_2.ipynb
```

Outputs go to `/workspace/cache/stage2/`: `gen/`, `inv/<family>/` (`log.csv`, `final.pt`), `features/`, `logs/`, `submission_stage2.csv` and `metrics.json`.

## Stage 3: token habits

Tests whether the sizes of a family can be told apart by per-image statistics of the tokens they write, using the tokens $Q(D^{-1}(x))$ recovered with Stage 2's $D^{-1}$. The criteria were registered before the analysis (oracle, realistic, transfer); the size step of the Stage 2 rule is replaced only for a family that passes all three.

```bash
# Fresh generated set (seed 1; D^-1 was trained on stage2/gen), then tokens of it and of every task image (about 70 min)
for f in rar var; do
  /workspace/venv/bin/python scripts/generate_finetune_data.py --family $f --per-model 2560 --seed 1 --out /workspace/cache/stage3/gen
  /workspace/venv/bin/python scripts/extract_tokens.py --family $f
done
# Then evaluate: notebooks/003_stage_3.ipynb (CPU only)
```

Outputs go to `/workspace/cache/stage3/`: `gen/`, `tokens/<family>/{gen,task}.npz`, `gen_split.csv`, `logs/`, and `submission_stage3.csv` only if val beats Stage 2. Result: negative. Token habits do not identify the size, even from the true tokens, so the Stage 2 predictions stand (see the notebook's Findings).

## Stage 4: generator likelihood

Tests whether each size's transformer gives the highest likelihood, under its own guided sampling distribution (rebuilt from each repo's sampler in [tracer/likelihood.py](tracer/likelihood.py)), to the tokens it generated. The images are Stage 3's generated set and task images, with Stage 3's recovered tokens. The class comes from a pretrained ImageNet classifier (top 5). The criteria were registered before the analysis (oracle, realistic, transfer); the size step of the Stage 2 rule is replaced only for a family that passes all three.

```bash
# Consistency checks of the likelihood code (GPU, about 5 min), then top-10 classes of task and generated images
/workspace/venv/bin/python -m tracer.test_likelihood
/workspace/venv/bin/python scripts/estimate_classes.py
# Scores under all 4 models of each family (about 16 h on an A40 in total; see scripts/README.md)
for f in rar var; do
  /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens true --class-mode true        # oracle
  /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens recovered --class-mode true   # recovered tokens, true class
  /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source gen --tokens recovered --class-mode topk   # realistic
  /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source task --tokens recovered --class-mode topk  # task train + val
done
# Then evaluate: notebooks/004_stage_4.ipynb (CPU only)
# For a submission, also score the test images (about 10 h), then rerun the notebook
for f in rar var; do
  /workspace/venv/bin/python scripts/score_likelihood.py --family $f --source task --tokens recovered --class-mode topk \
      --splits test --out /workspace/cache/stage4/scores_test
done
```

Outputs go to `/workspace/cache/stage4/`: `tests/`, `classes/{gen,task}.npz`, `scores/<family>/<source>_<tokens>_<class-mode>/<model>.npz`, `scores_test/`, `logs/`, and `submission_stage4.csv` only if val beats Stage 2. Result: positive. Every criterion passes for both families, and a logistic regression on the four models' likelihood summaries replaces the size step of both. Val accuracy rises from 31.6% (Stage 2) to 94.0%, and size accuracy given the correct family from 25.3% to 97.7%. The family and outlier decisions are Stage 2's, and now cause 18 of the 27 val errors (see the notebook's Findings). The test submission is written once the test images are scored.

## Evaluation

The task metric is 9-class accuracy. [tracer/metrics.py](tracer/metrics.py) also reports family accuracy (RAR / VAR / outlier), outlier recall and precision, per-family recall, and size accuracy given the correct family. These show which part of the problem a method actually solves.
