# notebooks

Analysis notebooks. Long jobs live in [scripts/](../scripts) and are run in tmux; the notebooks only read cached outputs, apart from small sanity checks.

**Kernel:** use the project environment, `/workspace/venv/bin/python` (Python 3.12, torch 2.6). If an import fails, install into the kernel with `%pip install <pkg>` (not `!pip`, which can target a different Python), then restart the kernel.

| notebook | purpose | needs | writes |
|---|---|---|---|
| [00_explore.ipynb](00_explore.ipynb) | data index, sample grid per label, and a round trip of one image through each autoencoder | data, weights, external repos, GPU | nothing |
| [001_stage_1.ipynb](001_stage_1.ipynb) | Stage 1 baseline: score selection, the nearest-family + threshold rule, val metrics, a test submission | `/workspace/cache/stage1/{rar,var}.csv` from `scripts/extract_features.py` | `/workspace/cache/stage1/submission_stage1.csv`, `metrics.json` |

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
