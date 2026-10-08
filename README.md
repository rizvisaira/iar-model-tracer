# IAR Model Tracer

Attribute each image to the image autoregressive (IAR) model that generated it, or flag it as an outlier. The 9 labels are RAR-B / L / XL / XXL (`rarb rarl rarxl rarxxl`), VAR-d16 / d20 / d24 / d30 (`var16 var20 var24 var30`) and `outlier`. The approach builds on the autoencoder-based provenance signals of Zhao et al., *Data Provenance for Image Auto-Regressive Generation* (ICLR 2026).

## Repository layout

| directory | contents | docs |
|---|---|---|
| [tracer/](tracer) | Python package: data index, autoencoder loaders, provenance signals, decision rule, metrics | [tracer/README.md](tracer/README.md) |
| [scripts/](scripts) | long-running jobs (feature extraction), run in tmux | [scripts/README.md](scripts/README.md) |
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

The generator weights are not used by the current code; only the two tokenizers are.

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

## Evaluation

The task metric is 9-class accuracy. [tracer/metrics.py](tracer/metrics.py) also reports family accuracy (RAR / VAR / outlier), outlier recall and precision, per-family recall, and size accuracy given the correct family. These show which part of the problem a method actually solves.
