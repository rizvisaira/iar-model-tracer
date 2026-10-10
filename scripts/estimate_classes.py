"""Class estimation (Stage 4): top-10 ImageNet classes of every image, from a pretrained ImageNet classifier.

The generators are class-conditional, and the guided likelihood (tracer/likelihood.py) needs the class. Generated
images come with their true class; task images do not, so their class is estimated here. The classifier is
torchvision's convnext_large with IMAGENET1K_V1 weights and their own preprocessing (resize 232, centre crop 224,
ImageNet normalisation), run on the [0,1] images. Two sources:
    task   every image from build_index() (train, val, test; all families and outliers), in index order
    gen    the generated set at --gen-root (both families), with its true classes

    python scripts/estimate_classes.py
    python scripts/estimate_classes.py --sources gen --limit 64 --out /tmp/classes_dryrun

Output, per source: resumable shards <out>/<source>/shard_XXXX.npz, merged into <out>/<source>.npz, with
top10_class [N,10] int16 and top10_prob [N,10] float32 (softmax probabilities, most likely first) and
    task   key, split, label, family, name (strings; "" for test labels / families)
    gen    family, label, index, class (the true class)
Rerunning skips finished shards; if the arguments (other than --batch-size) differ from <out>/<source>/config.json,
the script exits instead of mixing configurations.
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
from tracer.common import CACHE, DEVICE, WEIGHTS, build_index, free, load_batch  # noqa: E402
from tracer.inverse import load_generated  # noqa: E402

TASK_META = ["key", "split", "label", "family", "name"]
TOP_K = 10


def task_source(limit):
    """Metadata of every task image in build_index() order, and a function giving a batch of its images."""
    df = build_index()
    if limit:
        df = df.head(limit)
    meta = {c: df[c].fillna("").to_numpy(dtype=str) for c in TASK_META}
    return meta, lambda i, j: load_batch(df.path.iloc[i:j].tolist())


def gen_source(root, limit):
    """Metadata (with true classes) of the generated set of both families, and a function giving a batch of images.

    limit keeps the first N samples of each model, so a debug run still covers every model.
    """
    parts = [dict(load_generated(root, fam, max_per_model=limit), family=fam) for fam in ("rar", "var")]
    meta = {"family": np.concatenate([np.full(len(p["label"]), p["family"]) for p in parts]),
            "label": np.concatenate([p["label"] for p in parts]).astype(str),
            "index": np.concatenate([p["index"] for p in parts]).astype(np.int64),
            "class": np.concatenate([p["classes"] for p in parts]).astype(np.int64)}
    images = np.concatenate([p["images"] for p in parts])
    return meta, lambda i, j: torch.from_numpy(images[i:j]).to(DEVICE).float().div_(255)


def check_config(shard_dir: Path, cfg: dict):
    """Record the configuration; refuse to resume shards written with different settings."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != cfg:
        sys.exit(f"{path} differs from current args; delete {shard_dir} to recompute")
    path.write_text(json.dumps(cfg, indent=2))


def load_classifier(name: str):
    """torchvision ImageNet classifier and its preprocessing; weights are cached under WEIGHTS/torchvision."""
    import torchvision

    torch.hub.set_dir(str(WEIGHTS / "torchvision"))
    arch, weights = name.split(".")
    w = torchvision.models.get_model_weights(arch)[weights]
    model = torchvision.models.get_model(arch, weights=w).to(DEVICE).eval().requires_grad_(False)
    return model, w.transforms()


@torch.no_grad()
def extract(model, preprocess, args, source: str):
    """Top-10 classes of every image of one source; writes shards + the merged <source>.npz."""
    meta, get_images = task_source(args.limit) if source == "task" else gen_source(args.gen_root, args.limit)
    n = len(next(iter(meta.values())))
    shard_dir = args.out / source
    cfg = {"source": source, "classifier": args.classifier, "top_k": TOP_K, "shard_size": args.shard_size,
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
        cls, prob = [], []
        for i in range(a, b, args.batch_size):
            p = model(preprocess(get_images(i, min(i + args.batch_size, b)))).float().softmax(-1)
            top = p.topk(TOP_K, dim=-1)
            cls.append(top.indices.cpu()), prob.append(top.values.cpu())
        res = {k: v[a:b] for k, v in meta.items()}
        res["top10_class"] = torch.cat(cls).numpy().astype(np.int16)
        res["top10_prob"] = torch.cat(prob).numpy().astype(np.float32)
        # write to .tmp and rename, so an interrupted run never leaves a partial shard
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        print(f"[{source}] shard {s + 1}/{n_shards} done, {time.time() - t0:.0f}s elapsed", flush=True)

    # Merge all shards and check the result matches the source row by row.
    shards = [np.load(shard_dir / f"shard_{s:04d}.npz") for s in range(n_shards)]
    merged = {k: np.concatenate([sh[k] for sh in shards]) for k in shards[0].files}
    for k, v in meta.items():
        assert np.array_equal(merged[k], v), f"merged {k} does not match the source"
    out = args.out / f"{source}.npz"
    np.savez(out, **merged)
    print(f"wrote {out} ({len(merged['top10_class'])} images)", flush=True)
    if source == "gen":
        hit = merged["top10_class"] == merged["class"][:, None]
        acc = pd.DataFrame({f"top-{k}": hit[:, :k].any(1) for k in (1, 5, 10)}).groupby(merged["label"]).mean()
        acc.loc["all"] = [hit[:, :k].any(1).mean() for k in (1, 5, 10)]
        print("accuracy against the true generated classes:\n" + acc.to_string(float_format="%.3f"), flush=True)


def main():
    """Parse arguments, load the classifier, and estimate classes for each selected source."""
    ap = argparse.ArgumentParser(description="Top-10 ImageNet classes of task and generated images.")
    ap.add_argument("--sources", nargs="+", choices=["task", "gen"], default=["task", "gen"],
                    help="task: build_index() images; gen: the generated set at --gen-root")
    ap.add_argument("--classifier", default="convnext_large.IMAGENET1K_V1",
                    help="torchvision <architecture>.<weights>; preprocessing comes from the weights")
    ap.add_argument("--gen-root", type=Path, default=CACHE / "stage3" / "gen", help="generated set (load_generated)")
    ap.add_argument("--batch-size", type=int, default=64, help="images per GPU batch (may change on resume)")
    ap.add_argument("--shard-size", type=int, default=512, help="images per resumable shard file")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N task images / the first N generated images per model")
    ap.add_argument("--out", type=Path, default=CACHE / "stage4" / "classes", help="output directory")
    args = ap.parse_args()

    model, preprocess = load_classifier(args.classifier)
    print(f"loaded {args.classifier}; preprocessing: {preprocess}", flush=True)
    for source in args.sources:
        extract(model, preprocess, args, source)
    del model
    free()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # fixed input size, so let cuDNN pick the fastest kernels
    main()
