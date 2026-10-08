"""Paths, label mapping, data index and autoencoder loaders shared by scripts and notebooks.

Importing this module sets HF_HOME (if unset) and selects DEVICE ("cuda" if available, else "cpu").
The autoencoder loaders import code from the third-party repos in EXTERNAL and weights from WEIGHTS.
"""
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np
import pandas as pd
import torch
from PIL import Image

DATA = Path("/workspace/data")  # {train,val}/<label>/<n>.png and test/<n>.png
WEIGHTS = Path("/workspace/weights")  # rar/*.bin, var/*.pth
EXTERNAL = Path("/workspace/external")  # third-party code: VAR, 1d-tokenizer (RAR)
CACHE = Path("/workspace/cache")  # cached features and predictions, outside the repo

# Order fixed by the task template; label <-> index always goes through these dicts.
LABELS = ["var16", "var20", "var24", "var30", "rarb", "rarl", "rarxl", "rarxxl", "outlier"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
# label -> family, from the label prefix: "rarxl" -> "rar", "var16" -> "var", "outlier" -> "outlier"
FAMILY_OF = {l: ("var" if l.startswith("var") else "rar" if l.startswith("rar") else "outlier") for l in LABELS}
FAMILIES = ["rar", "var", "outlier"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_index() -> pd.DataFrame:
    """One row per image: split, label (None for test), family, name, path, key. Labels come from folder names.

    Row order is fixed: train, then val (sorted paths), then test (sorted by numeric file name), so cached
    feature files can be checked against it row by row. key is "split/label/name" ("test//<n>.png" for test).
    """
    rows = []
    for split in ["train", "val"]:
        for p in sorted((DATA / split).glob("*/*.png")):
            label = p.parent.name
            assert label in LABEL2IDX, f"unknown label folder {label}"
            rows.append((split, label, p.name, str(p)))
    for p in sorted((DATA / "test").glob("*.png"), key=lambda q: int(q.stem)):
        rows.append(("test", None, p.name, str(p)))
    df = pd.DataFrame(rows, columns=["split", "label", "name", "path"])
    df["family"] = df.label.map(FAMILY_OF)
    # train/val file names repeat across class folders, so the unique key includes split and label
    df["key"] = df.split + "/" + df.label.fillna("") + "/" + df.name
    assert df.key.is_unique
    return df


def load_batch(paths) -> torch.Tensor:
    """List of PNG paths -> [B,3,H,W] float in [0,1] on DEVICE."""
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255).to(DEVICE)


# Both repos expose top-level packages named `models`, `utils`, `data`, so only one can be imported at a time.
CLASHING = ("models", "utils", "data", "modeling", "dist")


def _switch_repo(path):
    """Make `path` the only external repo on sys.path and drop cached modules whose names clash."""
    for p in [str(EXTERNAL / "VAR"), str(EXTERNAL / "1d-tokenizer")]:
        while p in sys.path:
            sys.path.remove(p)
    for m in [k for k in sys.modules if k.split(".")[0] in CLASHING]:
        del sys.modules[m]
    sys.path.insert(0, str(path))


def free():
    """Release GPU memory after `del`-ing a model, e.g. before loading the other family's autoencoder."""
    gc.collect()
    torch.cuda.empty_cache()


def load_rar_tokenizer():
    """MaskGIT-VQGAN tokenizer shared by all RAR sizes. Inputs/outputs in [0,1].

    256x256 image -> 16x16 grid of 256-dim features -> indices into a 1024-entry codebook.
    Modules used by tracer.signals: .encoder, .quantize (returns quantized features, indices, loss), .decoder.
    """
    _switch_repo(EXTERNAL / "1d-tokenizer")
    from modeling.titok import PretrainedTokenizer
    return PretrainedTokenizer(str(WEIGHTS / "rar/maskgit-vqgan-imagenet-f16-256.bin")).to(DEVICE).eval()


def load_var_vae():
    """Multi-scale VQVAE shared by all VAR depths. Inputs/outputs in [-1,1].

    256x256 image -> 16x16 grid of 32-dim features, quantized into token maps at 10 scales
    (1x1 ... 16x16, 680 tokens) from a 4096-entry codebook. Built directly (no VAR transformer) and frozen.
    """
    _switch_repo(EXTERNAL / "VAR")
    from models.vqvae import VQVAE
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4,
                v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).to(DEVICE)
    vae.load_state_dict(torch.load(WEIGHTS / "var/vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    return vae.eval().requires_grad_(False)
