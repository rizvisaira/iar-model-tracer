"""Token extraction (Stage 3): recover the tokens Q(D^-1(x)) of every image with one family's Stage 2 D^-1.

The tokenizer is loaded with the fine-tuned D^-1 (scripts/finetune_inverse.py) in place of the encoder, and each
image's tokens are recovered with tracer.tokens.recover_tokens. Two sources:
    task   every image from build_index() (train, val, test; all families and outliers), in index order
    gen    the fresh generated set (scripts/generate_finetune_data.py --seed 1), with its true tokens

    python scripts/extract_tokens.py --family rar
    python scripts/extract_tokens.py --family var --sources gen

Output, per source: resumable shards <out>/<family>/<source>/shard_XXXX.npz, merged into <out>/<family>/<source>.npz.
Token arrays are int16 [N,L] (L = 256 RAR, 680 VAR; layout as in tracer/tokens.py).
    task   key, split, label, family, name (strings; "" for test labels / families), tokens (recovered)
    gen    label, class, index, true_tokens (sampled by the generator), tokens (recovered)
Rerunning skips finished shards; if the arguments (other than --batch-size) differ from
<out>/<family>/<source>/config.json, the script exits instead of mixing configurations.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import CACHE, DEVICE, build_index, free, load_batch  # noqa: E402
from tracer.inverse import LOAD_TOKENIZER, load_generated, load_inverse  # noqa: E402
from tracer.tokens import recover_tokens  # noqa: E402

TASK_META = ["key", "split", "label", "family", "name"]


def task_source(limit):
    """Metadata of every task image in build_index() order, and a function giving a batch of its images."""
    df = build_index()
    if limit:
        df = df.head(limit)
    meta = {c: df[c].fillna("").to_numpy(dtype=str) for c in TASK_META}
    return meta, lambda i, j: load_batch(df.path.iloc[i:j].tolist())


def gen_source(root, family, limit):
    """Metadata (with true tokens) of the generated set, and a function giving a batch of its images.

    limit keeps the first N samples of each model, so a debug run still covers every size.
    """
    data = load_generated(root, family, max_per_model=limit)
    meta = {"label": data["label"].astype(str), "class": data["classes"].astype(np.int64),
            "index": data["index"].astype(np.int64), "true_tokens": data["tokens"].astype(np.int16)}
    images = data["images"]
    return meta, lambda i, j: torch.from_numpy(images[i:j]).to(DEVICE).float().div_(255)


def check_config(shard_dir: Path, cfg: dict):
    """Record the configuration; refuse to resume shards written with different settings."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != cfg:
        sys.exit(f"{path} differs from current args; delete {shard_dir} to recompute")
    path.write_text(json.dumps(cfg, indent=2))


def extract(tok, args, source: str):
    """Recover the tokens of every image of one source and write shards + the merged <source>.npz."""
    if source == "task":
        meta, get_images = task_source(args.limit)
    else:
        meta, get_images = gen_source(args.gen_root, args.family, args.limit)
    n = len(next(iter(meta.values())))
    shard_dir = args.out / args.family / source
    cfg = {"family": args.family, "source": source, "inv_ckpt": str(args.inv_ckpt), "shard_size": args.shard_size,
           "limit": args.limit}
    if source == "gen":
        cfg["gen_root"] = str(args.gen_root)
    check_config(shard_dir, cfg)

    n_shards = (n + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    for s in range(n_shards):
        path = shard_dir / f"shard_{s:04d}.npz"
        if path.exists():  # finished in an earlier run
            continue
        a, b = s * args.shard_size, min((s + 1) * args.shard_size, n)
        tokens = torch.cat([recover_tokens(tok, args.family, get_images(i, min(i + args.batch_size, b))).cpu()
                            for i in range(a, b, args.batch_size)]).numpy().astype(np.int16)
        res = {k: v[a:b] for k, v in meta.items()}
        res["tokens"] = tokens
        # write to .tmp and rename, so an interrupted run never leaves a partial shard
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        acc = f", token acc {np.mean(tokens == res['true_tokens']):.3f}" if source == "gen" else ""
        print(f"[{args.family}/{source}] shard {s + 1}/{n_shards} done{acc}, {time.time() - t0:.0f}s elapsed",
              flush=True)

    # Merge all shards and check the result matches the source row by row.
    shards = [np.load(shard_dir / f"shard_{s:04d}.npz") for s in range(n_shards)]
    merged = {k: np.concatenate([sh[k] for sh in shards]) for k in shards[0].files}
    for k, v in meta.items():
        assert np.array_equal(merged[k], v), f"merged {k} does not match the source"
    out = args.out / args.family / f"{source}.npz"
    np.savez(out, **merged)
    print(f"wrote {out} ({len(merged['tokens'])} images)", flush=True)
    if source == "gen":
        acc = (merged["tokens"] == merged["true_tokens"]).mean(axis=1)
        print("recovered-vs-true token accuracy per model:\n"
              + pd.Series(acc).groupby(merged["label"]).agg(["mean", "count"]).to_string(float_format="%.3f")
              + f"\nall: {acc.mean():.3f}", flush=True)


def main():
    """Parse arguments, load the tokenizer with D^-1, and extract tokens for each selected source."""
    ap = argparse.ArgumentParser(description="Recover Q(D^-1(x)) tokens with one family's fine-tuned D^-1.")
    ap.add_argument("--family", choices=["rar", "var"], required=True, help="which family's tokenizer to use")
    ap.add_argument("--sources", nargs="+", choices=["task", "gen"], default=["task", "gen"],
                    help="task: build_index() images; gen: the generated set at --gen-root")
    ap.add_argument("--inv-ckpt", type=Path, default=None,
                    help="fine-tuned D^-1 checkpoint; default: <cache>/stage2/inv/<family>/final.pt")
    ap.add_argument("--gen-root", type=Path, default=CACHE / "stage3" / "gen", help="generated set (load_generated)")
    ap.add_argument("--batch-size", type=int, default=64, help="images per GPU batch (may change on resume)")
    ap.add_argument("--shard-size", type=int, default=512, help="images per resumable shard file")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N task images / the first N generated images per model")
    ap.add_argument("--out", type=Path, default=CACHE / "stage3" / "tokens", help="output directory")
    args = ap.parse_args()
    args.inv_ckpt = args.inv_ckpt or CACHE / "stage2" / "inv" / args.family / "final.pt"

    tok = load_inverse(LOAD_TOKENIZER[args.family](), args.family, args.inv_ckpt)
    print(f"loaded D^-1 from {args.inv_ckpt}", flush=True)
    for source in args.sources:
        extract(tok, args, source)
    del tok
    free()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # fixed input size, so let cuDNN pick the fastest kernels
    main()
