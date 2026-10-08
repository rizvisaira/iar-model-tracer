"""Inverse decoder D^-1 (Stage 2): fine-tune each family's encoder so that it inverts the frozen decoder.

Zhao et al., Eq. 6: L_inv = ||f_Z - D^-1(D(f_Z))||^2, with D^-1 initialised from the original encoder E, and the
codebook and decoder D frozen. Training pairs (t_Z, x_Z = D(Q^-1(t_Z))) come from the family's generators
(tracer/generate.py, scripts/generate_finetune_data.py), so the target f_Z = Q^-1(t_Z) is known exactly.

D^-1 has the same architecture as E, so a fine-tuned D^-1 is used by loading its weights into the tokenizer
(load_inverse); tracer/signals.py then computes every Stage 1 signal with D^-1 in place of E.
    rar: D^-1 = tok.encoder                          x in [0,1]  -> f [B,256,16,16]
    var: D^-1 = vae.quant_conv(vae.encoder(2x - 1))  x in [0,1]  -> f [B,32,16,16]
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DEVICE, build_index, load_batch, load_rar_tokenizer, load_var_vae
from .generate import split_var_tokens
from .signals import _var_idx_to_fhat, rar_signals, var_signals

LOAD_TOKENIZER = {"rar": load_rar_tokenizer, "var": load_var_vae}


# ---------------------------------------------------------------- D^-1, targets and quantization, per family

def inverse_module(tok, family: str) -> nn.Module:
    """The trainable part of the tokenizer: the modules that make up D^-1 (shared with tok, not a copy)."""
    return tok.encoder if family == "rar" else nn.ModuleDict({"encoder": tok.encoder, "quant_conv": tok.quant_conv})


def encode(tok, family: str, x01: torch.Tensor) -> torch.Tensor:
    """D^-1(x): [B,3,256,256] in [0,1] -> continuous features f [B,C,16,16]."""
    return tok.encoder(x01) if family == "rar" else tok.quant_conv(tok.encoder(x01 * 2 - 1))


def target(tok, family: str, tokens: torch.Tensor) -> torch.Tensor:
    """f_Z = Q^-1(t_Z) for generated tokens [B,L]: codebook vectors (RAR) or the multi-scale sum f_hat (VAR)."""
    if family == "rar":
        return tok.quantize.get_codebook_entry(tokens)
    return _var_idx_to_fhat(tok.quantize, split_var_tokens(tokens))


def quantize(tok, family: str, f: torch.Tensor):
    """Q then Q^-1 with the original quantizer: f -> (tokens [B,L], f_Z [B,C,16,16]). VAR uses its greedy Alg. 2."""
    if family == "rar":
        zq, idx, _ = tok.quantize(f)
        return idx, zq
    idx = tok.quantize.f_to_idxBl_or_fhat(f, to_fhat=False)
    return torch.cat(idx, 1), _var_idx_to_fhat(tok.quantize, idx)


def load_inverse(tok, family: str, path):
    """Load fine-tuned D^-1 weights (a checkpoint from finetune) into tok, in place. Returns tok."""
    ckpt = torch.load(path, map_location="cpu")
    inverse_module(tok, family).load_state_dict(ckpt["inverse"] if "inverse" in ckpt else ckpt)
    return tok


# ---------------------------------------------------------------- generated data

def load_generated(root, family: str, models=None, max_per_model=None) -> dict:
    """Load generated shards from <root>/<family>/<label>/shard_*.npz into memory.

    Returns numpy arrays tokens [N,L] (int64), images [N,3,256,256] (uint8), classes [N], label [N] (str) and
    index [N] (position within its model's sample sequence). max_per_model keeps the first n samples of each model.
    """
    parts = []
    for d in sorted((Path(root) / family).iterdir()):
        if not d.is_dir() or (models and d.name not in models):
            continue
        shards = [np.load(p) for p in sorted(d.glob("shard_*.npz"))]
        if not shards:
            continue
        tok = np.concatenate([s["tokens"] for s in shards]).astype(np.int64)
        img = np.concatenate([s["images"] for s in shards])
        cls = np.concatenate([s["classes"] for s in shards])
        n = len(tok) if max_per_model is None else min(len(tok), max_per_model)
        parts.append(dict(tokens=tok[:n], images=img[:n], classes=cls[:n], label=np.full(n, d.name),
                          index=np.arange(n)))
    assert parts, f"no generated shards under {Path(root) / family}"
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def split_holdout(data: dict, n_per_model: int):
    """The last n_per_model samples of each model are held out for evaluation; returns (train, holdout)."""
    last = pd.Series(data["index"]).groupby(data["label"]).transform("max").values
    hold = data["index"] > last - n_per_model
    return ({k: v[~hold] for k, v in data.items()}, {k: v[hold] for k, v in data.items()})


def _batches(data: dict, batch_size: int, order=None):
    """Yield (x01 [B,3,256,256] float on DEVICE, tokens [B,L] on DEVICE)."""
    order = np.arange(len(data["tokens"])) if order is None else order
    for i in range(0, len(order), batch_size):
        j = order[i:i + batch_size]
        x = torch.from_numpy(data["images"][j]).to(DEVICE, non_blocking=True).float().div_(255)
        yield x, torch.from_numpy(data["tokens"][j]).to(DEVICE, non_blocking=True)


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate_generated(tok, family: str, data: dict, batch_size: int = 64) -> dict:
    """How well D^-1 inverts D on generated images (higher token accuracy / lower losses = better inverse).

    l_inv      ||f_Z - D^-1(x_Z)||^2, the training loss (Eq. 6)
    tok_acc    fraction of tokens recovered exactly: Q(D^-1(x_Z)) == t_Z
    quant      QuantLoss ||D^-1(x) - Q^-1(Q(D^-1(x)))||^2 (Eq. 5), as Stage 1 measures it on real images
    """
    sums, n = {"l_inv": 0.0, "tok_acc": 0.0, "quant": 0.0}, 0
    for x, t in _batches(data, batch_size):
        f = encode(tok, family, x)
        idx, fz = quantize(tok, family, f)
        sums["l_inv"] += (f - target(tok, family, t)).pow(2).flatten(1).mean(1).sum().item()
        sums["tok_acc"] += (idx == t).float().mean(1).sum().item()
        sums["quant"] += (f - fz).pow(2).flatten(1).mean(1).sum().item()
        n += len(t)
    return {k: v / n for k, v in sums.items()}


_TRAIN_IMAGES = {}


def _task_train_images():
    """The task's train images as uint8 [800,3,256,256] on the CPU, plus their index rows (loaded once)."""
    if not _TRAIN_IMAGES:
        idx = build_index()
        idx = idx[idx.split == "train"].reset_index(drop=True)
        imgs = [(load_batch(idx.path.iloc[i:i + 64].tolist()) * 255).round().to(torch.uint8).cpu()
                for i in range(0, len(idx), 64)]
        _TRAIN_IMAGES.update(idx=idx, images=torch.cat(imgs))
    return _TRAIN_IMAGES["idx"], _TRAIN_IMAGES["images"]


@torch.no_grad()
def evaluate_task_train(tok, family: str, batch_size: int = 64, signals=("quant", "cal", "comb", "tok_match")) -> dict:
    """AUC of the family's signals on the task's 800 train images, own family vs the other family (as in Stage 1).

    Uses train only (never val), so it can be logged during fine-tuning without touching held-out data.
    """
    from sklearn.metrics import roc_auc_score

    idx, images = _task_train_images()
    fn = (lambda x: rar_signals(tok, x)) if family == "rar" else (lambda x: var_signals(tok, x, n_iters=0))
    out = {s: [] for s in signals}
    for i in range(0, len(idx), batch_size):
        res = fn(images[i:i + batch_size].to(DEVICE).float().div_(255))
        for s in signals:
            out[s].append(res[s].float().cpu().numpy())
    own = (idx.family == family).values
    auc = {}
    for s in signals:
        v = np.concatenate(out[s])
        auc[f"train_auc_{s}"] = roc_auc_score(own, v if s == "tok_match" else -np.log(v))  # higher = own family
    return auc


# ---------------------------------------------------------------- training

def finetune(family: str, train: dict, holdout: dict, out_dir, epochs: int, batch_size: int, lr: float,
             step_size: int = 2, gamma: float = 0.9, seed: int = 0, eval_task_train: bool = True, log=print):
    """Fine-tune D^-1 for one family with L_inv (Eq. 6), Adam and StepLR (stepped once per epoch).

    Writes to out_dir:
        log.csv      one row per epoch (epoch 0 = original encoder): mean train L_inv, the lr used in that epoch,
                     held-out metrics (evaluate_generated) and, if eval_task_train, task-train AUCs
        last.pt      D^-1 weights plus optimizer / scheduler state after the latest epoch (used to resume)
        final.pt     D^-1 weights after the last epoch (load with load_inverse)
    Rerunning with the same out_dir resumes from last.pt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = LOAD_TOKENIZER[family]()
    inv = inverse_module(tok, family).requires_grad_(True)
    opt = torch.optim.Adam(inv.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    log_path, last_path = out_dir / "log.csv", out_dir / "last.pt"
    rows, start = [], 1

    def evaluate(epoch, train_loss, lr):
        inv.eval()
        row = {"epoch": epoch, "train_l_inv": train_loss, "lr": lr}
        row.update({f"holdout_{k}": v for k, v in evaluate_generated(tok, family, holdout).items()})
        if eval_task_train:
            row.update(evaluate_task_train(tok, family))
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    if last_path.exists():  # resume
        ckpt = torch.load(last_path, map_location="cpu")
        inv.load_state_dict(ckpt["inverse"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rows, start = pd.read_csv(log_path).to_dict("records"), ckpt["epoch"] + 1
        log(f"resumed after epoch {ckpt['epoch']}")
    else:
        evaluate(0, float("nan"), float("nan"))  # baseline: the original encoder

    rng = np.random.default_rng(seed)
    for _ in range(1, start):  # replay the shuffles of finished epochs so a resumed run sees the same order
        rng.permutation(len(train["tokens"]))
    for epoch in range(start, epochs + 1):
        inv.train()
        t0, total, n = time.time(), 0.0, 0
        for x, t in _batches(train, batch_size, rng.permutation(len(train["tokens"]))):
            with torch.no_grad():
                fz = target(tok, family, t)
            loss = F.mse_loss(encode(tok, family, x), fz)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(t), n + len(t)
        lr = opt.param_groups[0]["lr"]  # the rate used during this epoch
        sched.step()
        log(f"epoch {epoch}/{epochs} done in {time.time() - t0:.0f}s")
        evaluate(epoch, total / n, lr)
        torch.save({"inverse": inv.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch}, last_path.with_suffix(".tmp"))
        last_path.with_suffix(".tmp").rename(last_path)
    torch.save({"inverse": inv.state_dict(), "epoch": epochs}, out_dir / "final.pt")
    return tok


def write_config(out_dir, cfg: dict):
    """Write cfg to out_dir/config.json, or exit if a different config is already there (no mixed resumes)."""
    path = Path(out_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise SystemExit(f"{path} differs from the current arguments; use a new --out or delete it to restart")
    path.write_text(json.dumps(cfg, indent=2))
