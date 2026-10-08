"""Feature extraction (Stages 1 and 2): per-family provenance signals for every image.

Stage 1 uses the original encoders as D^-1; Stage 2 loads a fine-tuned D^-1 with --inv-ckpt (see
scripts/finetune_inverse.py). The signal code is the same in both stages.
Each family's tokenizer is shared by all its sizes, so signals are computed once per family, not per model.
Every image (all families and outliers) is scored with the chosen family's tokenizer; see tracer/signals.py.
Writes resumable shards to <out>/<family>/shard_XXXX.csv and a merged <out>/<family>.csv.

    python scripts/extract_features.py --family rar
    python scripts/extract_features.py --family var --var-iters 0
    python scripts/extract_features.py --family rar --inv-ckpt /workspace/cache/stage2/inv/rar/final.pt \\
        --out /workspace/cache/stage2/features

Output: one row per image in build_index() order, with columns key, split, label, family, name and one
<family>_<signal> column per signal. Rerunning skips finished shards; if the arguments (other than
--batch-size) differ from <out>/<family>/config.json, the script exits instead of mixing configurations.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import CACHE, build_index, free, load_batch, load_rar_tokenizer, load_var_vae  # noqa: E402
from tracer.inverse import load_inverse  # noqa: E402
from tracer.signals import rar_signals, var_signals  # noqa: E402


def main():
    """Parse arguments, score every selected image with one family's tokenizer, and write shards + merged CSV."""
    ap = argparse.ArgumentParser(description="Compute provenance signals for one tokenizer family.")
    ap.add_argument("--family", choices=["rar", "var"], required=True, help="which family's tokenizer to use")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="data splits to score")
    ap.add_argument("--batch-size", type=int, default=64, help="images per GPU batch (may change on resume)")
    ap.add_argument("--shard-size", type=int, default=512, help="images per resumable shard file")
    ap.add_argument("--var-iters", type=int, default=0, help="Algorithm 3 token-search iterations (0 = greedy only)")
    ap.add_argument("--var-lr", type=float, default=0.1, help="Algorithm 3 Adam learning rate")
    ap.add_argument("--var-init-logit", type=float, default=10.0, help="Algorithm 3 initial logit on greedy tokens")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N images")
    ap.add_argument("--inv-ckpt", type=Path, default=None,
                    help="fine-tuned D^-1 checkpoint (Stage 2); default: the original encoder (Stage 1)")
    ap.add_argument("--out", type=Path, default=CACHE / "stage1", help="output directory")
    args = ap.parse_args()

    # Images to score, in build_index() order.
    df = build_index()
    df = df[df.split.isin(args.splits)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    # Record the configuration; refuse to resume shards written with different settings.
    # inv_ckpt is recorded only when set, so Stage 1 configs written before it existed still match.
    shard_dir = args.out / args.family
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
           if k != "batch_size" and not (k == "inv_ckpt" and v is None)}
    cfg_path = shard_dir / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        sys.exit(f"{cfg_path} differs from current args; delete {shard_dir} to recompute")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # fn: [B,3,256,256] batch in [0,1] -> dict of per-image signal tensors.
    if args.family == "rar":
        model = load_rar_tokenizer()
        fn = lambda x: rar_signals(model, x)  # noqa: E731
    else:
        model = load_var_vae()
        fn = lambda x: var_signals(model, x, n_iters=args.var_iters, lr=args.var_lr, init_logit=args.var_init_logit)  # noqa: E731
    if args.inv_ckpt:
        load_inverse(model, args.family, args.inv_ckpt)  # replace the encoder by the fine-tuned D^-1
        print(f"loaded D^-1 from {args.inv_ckpt}", flush=True)

    n_shards = (len(df) + args.shard_size - 1) // args.shard_size
    t0 = time.time()
    for s in range(n_shards):
        path = shard_dir / f"shard_{s:04d}.csv"
        if path.exists():  # finished in an earlier run
            continue
        part = df.iloc[s * args.shard_size:(s + 1) * args.shard_size]
        feats = []
        for i in range(0, len(part), args.batch_size):
            x = load_batch(part.path.iloc[i:i + args.batch_size].tolist())
            out = fn(x)
            # prefix each signal with the family, e.g. cal -> rar_cal
            feats.append(pd.DataFrame({f"{args.family}_{k}": v.float().cpu().numpy() for k, v in out.items()}))
        res = pd.concat([part[["key", "split", "label", "family", "name"]].reset_index(drop=True),
                         pd.concat(feats, ignore_index=True)], axis=1)
        assert res.notna().drop(columns=["label", "family"]).all().all(), "NaN in features"
        # write to .tmp and rename, so an interrupted run never leaves a partial shard
        res.to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)
        print(f"[{args.family}] shard {s + 1}/{n_shards} done, {time.time() - t0:.0f}s elapsed", flush=True)

    # Merge all shards and check the result matches the index row by row.
    merged = pd.concat([pd.read_csv(shard_dir / f"shard_{s:04d}.csv") for s in range(n_shards)], ignore_index=True)
    assert len(merged) == len(df) and merged.key.tolist() == df.key.tolist()
    merged.to_csv(args.out / f"{args.family}.csv", index=False)
    print(f"wrote {args.out / f'{args.family}.csv'} ({len(merged)} rows)")
    del model
    free()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # fixed input size, so let cuDNN pick the fastest kernels
    main()
