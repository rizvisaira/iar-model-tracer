"""Generate inverse-decoder fine-tuning data (Stage 2): samples from every generator of one family, with tokens.

All sizes of a family share the tokenizer, and the task's images come from all of them, so one D^-1 per family is
fine-tuned on an equal mix of every size's samples. Classes cycle through the 1000 ImageNet classes in a fixed
random order per model. Shards are resumable and deterministic (seeded per model and shard; bit-identical for the
same --seed and --batch-size), and always hold --shard-size samples, so a larger --per-model later only adds shards.

    python scripts/generate_finetune_data.py --family rar --per-model 2500
    python scripts/generate_finetune_data.py --family var --per-model 2500

Writes <out>/<family>/<label>/shard_XXXX.npz with tokens [n,L] (int16), images [n,3,256,256] (uint8, = the tokenizer
decoder's output for those tokens) and classes [n] (int16), plus <out>/<family>/config.json.
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import CACHE, free, load_rar_tokenizer  # noqa: E402
from tracer.generate import MODELS, load_rar_generator, load_var_generator, sample_rar, sample_var  # noqa: E402


def main():
    """Parse arguments, then sample --per-model images from each of the family's generators, shard by shard."""
    ap = argparse.ArgumentParser(description="Sample (tokens, image) pairs from one family's generators.")
    ap.add_argument("--family", choices=["rar", "var"], required=True, help="which generator family to sample")
    ap.add_argument("--per-model", type=int, required=True,
                    help="samples per model size, rounded up to a whole number of shards")
    ap.add_argument("--models", nargs="+", default=None, help="subset of model labels (default: all 4 sizes)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="images per generation batch; part of the seeding, so fixed per output directory")
    ap.add_argument("--shard-size", type=int, default=512, help="images per resumable shard file")
    ap.add_argument("--seed", type=int, default=0, help="base random seed")
    ap.add_argument("--out", type=Path, default=CACHE / "stage2" / "gen", help="output directory")
    args = ap.parse_args()
    models = args.models or MODELS[args.family]
    assert set(models) <= set(MODELS[args.family]), f"--models must be from {MODELS[args.family]}"

    # Record the configuration; refuse to resume shards written with different settings.
    # --per-model may grow between runs (shards are a deterministic prefix), so it is not part of the check.
    # --batch-size is: the samples drawn depend on it (same seed and batch size -> bit-identical shards).
    fam_dir = args.out / args.family
    fam_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"family": args.family, "shard_size": args.shard_size, "batch_size": args.batch_size, "seed": args.seed}
    cfg_path = fam_dir / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        sys.exit(f"{cfg_path} differs from current args; delete {fam_dir} to regenerate")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    tok = load_rar_tokenizer() if args.family == "rar" else None
    t0 = time.time()
    for m_i, label in enumerate(MODELS[args.family]):
        if label not in models:
            continue
        out_dir = fam_dir / label
        out_dir.mkdir(exist_ok=True)
        n_shards = (args.per_model + args.shard_size - 1) // args.shard_size
        todo = [s for s in range(n_shards) if not (out_dir / f"shard_{s:04d}.npz").exists()]
        if not todo:
            continue
        # The same classes for every --per-model: a fixed shuffle of 0..999, repeated.
        classes_all = np.resize(np.random.default_rng(args.seed + m_i).permutation(1000), n_shards * args.shard_size)

        # fn: [B] classes -> (tokens, uint8 images) from this model
        if args.family == "rar":
            gen = load_rar_generator(label)
            fn = partial(sample_rar, gen, tok, label)
        else:
            vae, gen = load_var_generator(label)
            fn = partial(sample_var, vae, gen)

        for s in todo:
            torch.manual_seed(args.seed * 1_000_003 + m_i * 10_007 + s)  # deterministic per (model, shard)
            classes = torch.from_numpy(classes_all[s * args.shard_size:(s + 1) * args.shard_size].astype(np.int64))
            toks, imgs = [], []
            for i in range(0, len(classes), args.batch_size):
                t, im = fn(classes[i:i + args.batch_size])
                toks.append(t.cpu().numpy().astype(np.int16))  # codebooks have <= 4096 entries
                imgs.append(im.cpu().numpy())
            path = out_dir / f"shard_{s:04d}.npz"
            # write to .tmp and rename, so an interrupted run never leaves a partial shard
            with open(path.with_suffix(".tmp"), "wb") as fh:
                np.savez(fh, tokens=np.concatenate(toks), images=np.concatenate(imgs),
                         classes=classes.numpy().astype(np.int16))
            path.with_suffix(".tmp").rename(path)
            print(f"[{label}] shard {s + 1}/{n_shards} done, {time.time() - t0:.0f}s elapsed", flush=True)
        del gen, fn
        if args.family == "var":
            del vae
        free()
    print(f"done: {fam_dir}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # as in the repos' sampling scripts
    torch.backends.cudnn.allow_tf32 = True
    main()
