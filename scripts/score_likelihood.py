"""Generator likelihood scores (Stage 4): per-token log-probabilities of each image's tokens under each of the
family's 4 transformers, with tracer.likelihood.score_tokens.

For one --family and one input configuration, every image is scored by every model of the family (one model on the
GPU at a time). The input configuration is
    --source     gen    the fresh generated set (stage3/tokens/<family>/gen.npz, rows as in that file)
                 task   the task images of --splits (stage3/tokens/<family>/task.npz, rows as in that file)
    --tokens     true       the generator's own tokens (gen only)
                 recovered  Q(D^-1(x)) from scripts/extract_tokens.py
    --class-mode true   the generated image's true class (gen only)
                 topk   each of the top --topk classes from scripts/estimate_classes.py, scored separately

    python scripts/score_likelihood.py --family var --source gen --tokens true --class-mode true
    python scripts/score_likelihood.py --family rar --source task --tokens recovered --class-mode topk --topk 5

Output: <out>/<family>/<source>_<tokens>_<class-mode>/<model>/shard_XXXX.npz (resumable), merged into
<out>/<family>/<source>_<tokens>_<class-mode>/<model>.npz. K = 1 for --class-mode true, --topk otherwise.
    ids          gen: row, label, index, class (true);  task: row, key, split, label, family, name
                 (row = position in the stage3 token file; strings "" where missing)
    classes      [N,K] int16, the class scored in each slot;  class_prob [N,K] float32 (classifier probability,
                 1 for --class-mode true)
    per token    logp_cond, logp_uncond, logp_guided [N,K,L] float16; rank_guided [N,K,L] int16;
                 in_support [N,K,L] int8 (VAR only)
    per image    computed in float32 before rounding, [N,K] float32:
                 sum_<q>, mean_<q> for every per-token quantity q;
                 min20_logp_guided    mean of the lowest 20% of the image's logp_guided (Min-K% with K = 20%)
                 mean_cond_minus_uncond   mean of logp_cond - logp_uncond
                 in_support_rate      fraction of tokens in the top-k / top-p support (VAR only)
Rerunning skips finished shards; if the arguments (other than --batch-size) differ from
<model>/config.json, the script exits instead of mixing configurations.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import CACHE, free  # noqa: E402
from tracer.generate import MODELS, load_rar_generator, load_var_generator  # noqa: E402
from tracer.likelihood import score_tokens  # noqa: E402

TOKENS = CACHE / "stage3" / "tokens"
MIN_FRACTION = 0.2  # Min-K% of logp_guided
FLOAT_KEYS = ["logp_cond", "logp_uncond", "logp_guided"]


def load_inputs(args) -> dict:
    """Ids, tokens [N,L], classes [N,K] and class_prob [N,K] of every image to score, in token-file row order."""
    with np.load(TOKENS / args.family / f"{args.source}.npz") as z:
        tok = {k: z[k] for k in z.files}
    n_all = len(tok["tokens"])
    if args.source == "gen":
        keep = np.ones(n_all, bool) if args.limit is None else tok["index"] < args.limit  # first N per model
        ids = {"row": np.arange(n_all), "label": tok["label"], "index": tok["index"], "class": tok["class"]}
    else:
        keep = np.isin(tok["split"], args.splits)
        if args.limit is not None:
            keep &= np.cumsum(keep) <= args.limit  # first N of the selected splits
        ids = {"row": np.arange(n_all), **{k: tok[k] for k in ("key", "split", "label", "family", "name")}}
    ids = {k: v[keep] for k, v in ids.items()}
    tokens = (tok["true_tokens"] if args.tokens == "true" else tok["tokens"])[keep].astype(np.int64)

    if args.class_mode == "true":
        classes, class_prob = ids["class"][:, None], np.ones((len(tokens), 1), np.float32)
    else:
        with np.load(args.classes or CACHE / "stage4" / "classes" / f"{args.source}.npz") as z:
            cl = {k: z[k] for k in z.files}
        # look up each image's row in the class file by its identity (gen: family/label/index; task: key)
        if args.source == "gen":
            fam = cl["family"] == args.family
            lookup = {(lab, int(i)): r for r, lab, i in zip(np.flatnonzero(fam), cl["label"][fam], cl["index"][fam])}
            rows = np.array([lookup.get((lab, int(i)), -1) for lab, i in zip(ids["label"], ids["index"])])
        else:
            lookup = {k: r for r, k in enumerate(cl["key"])}
            rows = np.array([lookup.get(k, -1) for k in ids["key"]])
        assert (rows >= 0).all(), f"{(rows < 0).sum()} images are missing from the class file"
        if args.source == "gen":
            assert (cl["class"][rows] == ids["class"]).all(), "class file disagrees with the token file"
        classes = cl["top10_class"][rows, :args.topk].astype(np.int64)
        class_prob = cl["top10_prob"][rows, :args.topk].astype(np.float32)
    return {"ids": ids, "tokens": tokens, "classes": classes, "class_prob": class_prob}


def summarise(res: dict, family: str) -> dict:
    """Per-image summaries [N,K] float32 of the float32 per-token arrays [N,K,L]."""
    out = {}
    for q, v in res.items():
        out[f"sum_{q}"] = v.sum(-1)
        out[f"mean_{q}"] = v.mean(-1)
    lp = np.sort(res["logp_guided"], axis=-1)
    out["min20_logp_guided"] = lp[..., :max(1, int(round(MIN_FRACTION * lp.shape[-1])))].mean(-1)
    out["mean_cond_minus_uncond"] = (res["logp_cond"] - res["logp_uncond"]).mean(-1)
    if family == "var":
        out["in_support_rate"] = res["in_support"].mean(-1)
    return {k: v.astype(np.float32) for k, v in out.items()}


def check_config(shard_dir: Path, cfg: dict):
    """Record the configuration; refuse to resume shards written with different settings."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / "config.json"
    if path.exists() and json.loads(path.read_text()) != cfg:
        sys.exit(f"{path} differs from current args; delete {shard_dir} to recompute")
    path.write_text(json.dumps(cfg, indent=2))


def score_model(label: str, inputs: dict, args, run_dir: Path):
    """Score every image with one model; writes shards + the merged <model>.npz. Loads the model only if needed."""
    shard_dir = run_dir / label
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
           if k not in ("batch_size", "models", "out")}
    cfg["model"] = label
    check_config(shard_dir, cfg)
    n, (_, K) = len(inputs["tokens"]), inputs["classes"].shape
    n_shards = (n + args.shard_size - 1) // args.shard_size
    todo = [s for s in range(n_shards) if not (shard_dir / f"shard_{s:04d}.npz").exists()]

    model, t0, n_done = None, time.time(), 0
    if todo:
        model = load_rar_generator(label) if args.family == "rar" else load_var_generator(label)
        torch.cuda.synchronize()
        print(f"[{label}] loaded in {time.time() - t0:.0f}s; {len(todo)}/{n_shards} shards to score", flush=True)
    t0 = time.time()
    for s in todo:
        a, b = s * args.shard_size, min((s + 1) * args.shard_size, n)
        tokens = np.repeat(inputs["tokens"][a:b], K, axis=0)  # image-major: rows i*K .. i*K+K-1 are image i
        classes = inputs["classes"][a:b].reshape(-1)
        sc = score_tokens(model, args.family, label, tokens, classes, batch_size=args.batch_size)
        sc = {k: v.reshape(b - a, K, -1) for k, v in sc.items()}
        res = {k: v[a:b] for k, v in inputs["ids"].items()}
        res.update(classes=inputs["classes"][a:b].astype(np.int16), class_prob=inputs["class_prob"][a:b])
        res.update(summarise(sc, args.family))
        res.update({k: sc[k].astype(np.float16) for k in FLOAT_KEYS})
        res["rank_guided"] = sc["rank_guided"].astype(np.int16)
        if args.family == "var":
            res["in_support"] = sc["in_support"].astype(np.int8)
        # write to .tmp and rename, so an interrupted run never leaves a partial shard
        path = shard_dir / f"shard_{s:04d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        n_done += b - a
        el = time.time() - t0
        print(f"[{label}] shard {s + 1}/{n_shards} done, {el:.0f}s elapsed, {1000 * el / n_done:.1f} ms/image "
              f"({1000 * el / (n_done * K):.1f} ms per image and class)", flush=True)
    if model is not None:
        del model
        free()

    # Merge all shards and check the ids match the inputs row by row.
    shards = [np.load(shard_dir / f"shard_{s:04d}.npz") for s in range(n_shards)]
    merged = {k: np.concatenate([sh[k] for sh in shards]) for k in shards[0].files}
    for k, v in inputs["ids"].items():
        assert np.array_equal(merged[k], v), f"merged {k} does not match the inputs"
    out = run_dir / f"{label}.npz"
    np.savez(out, **merged)
    print(f"wrote {out} ({len(merged['row'])} images x {K} classes)", flush=True)


def main():
    """Parse and validate arguments, load the inputs, and score them with each selected model in turn."""
    ap = argparse.ArgumentParser(description="Per-token generator likelihoods under each model of one family.")
    ap.add_argument("--family", choices=["rar", "var"], required=True)
    ap.add_argument("--source", choices=["gen", "task"], required=True)
    ap.add_argument("--tokens", choices=["true", "recovered"], required=True, help="true: gen only")
    ap.add_argument("--class-mode", choices=["true", "topk"], required=True, help="true: gen only")
    ap.add_argument("--topk", type=int, default=5, help="classes per image for --class-mode topk (at most 10)")
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"],
                    help="task images to score (task only)")
    ap.add_argument("--models", nargs="+", default=None, help="subset of the family's models (default: all 4)")
    ap.add_argument("--classes", type=Path, default=None,
                    help="class file for --class-mode topk; default <cache>/stage4/classes/<source>.npz")
    ap.add_argument("--batch-size", type=int, default=32, help="(image, class) pairs per GPU pass")
    ap.add_argument("--shard-size", type=int, default=512, help="images per resumable shard file")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: gen, the first N images per generating model; task, the first N images")
    ap.add_argument("--out", type=Path, default=CACHE / "stage4" / "scores", help="output directory")
    args = ap.parse_args()
    if args.source == "task" and "true" in (args.tokens, args.class_mode):
        ap.error("--tokens true and --class-mode true need --source gen (task images have neither)")
    if args.class_mode == "topk" and not 1 <= args.topk <= 10:
        ap.error("--topk must be between 1 and 10")
    models = args.models or MODELS[args.family]
    assert set(models) <= set(MODELS[args.family]), f"--models must be among {MODELS[args.family]}"
    if args.class_mode == "true":
        args.topk = None  # unused; kept out of the config
    if args.source == "gen":
        args.splits = None

    inputs = load_inputs(args)
    run_dir = args.out / args.family / f"{args.source}_{args.tokens}_{args.class_mode}"
    print(f"{run_dir.name}: {len(inputs['tokens'])} images x {inputs['classes'].shape[1]} classes, "
          f"models {models}", flush=True)
    for label in models:
        score_model(label, inputs, args, run_dir)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 scoring, as in tracer/test_likelihood.py
    torch.backends.cudnn.allow_tf32 = False
    main()
