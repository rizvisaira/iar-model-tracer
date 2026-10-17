"""Stage 5 pilot: score the VAR holdout of the fresh generated set under several inverse decoders D^-1.

Images are the holdout rows of stage3/gen_split.csv for all four VAR depths (512 each), read from stage3/gen. True
tokens come from stage3/tokens/var/gen.npz, matched by (label, index). Inverses, in this order:
    E        the original encoder (load_inverse is not called)
    shared   the Stage 2 D^-1 trained on all four depths (stage2/inv/var/final.pt)
    NAME     each --single NAME=PATH, e.g. a D^-1 trained on var24 only (scripts/finetune_inverse.py --models var24)

Per image and inverse:
    quant, enc1, enc2, cal, comb, tok_match   tracer.signals.var_signals with n_iters=0 (greedy quantization)
    tok_recovery   fraction of the 680 greedy tokens Q(D^-1(x)) equal to the true tokens
    inv_mse        mean (D^-1(x) - f_Z)^2, f_Z = _var_idx_to_fhat(true tokens): the Eq. 6 training loss per image

Before scoring, the var24 holdout true tokens are checked for exact overlap with stage2/gen/var/var24 (the data a
var24-only D^-1 is trained on); the script stops if any sequence appears in both.

    python scripts/score_stage5_pilot.py --single single5=/workspace/cache/stage5/inv/var24_5ep/final.pt
    python scripts/score_stage5_pilot.py --limit 16 --out /tmp/stage5_dry/scores.csv     # quick check

Writes the long-format CSV --out (label, index, inverse, quant, enc1, enc2, cal, comb, tok_match, tok_recovery,
inv_mse; one row per image and inverse) and <out stem>.config.json (per inverse: checkpoint, limit, row count,
git commit, command line). Rerunning skips inverses already scored with the same checkpoint and --limit; it exits
if a name is reused with a different checkpoint or --limit.
Convolutions run in full fp32 (TF32 off) with deterministic cuDNN: reruns at the same --batch-size are identical,
and other batch sizes change about 1 recovered token in 1e5. Recovered tokens can differ from stage3/tokens
(extracted with TF32) by a few percent; mean token recovery agrees.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tracer.common import CACHE, DEVICE, free, load_var_vae  # noqa: E402
from tracer.inverse import encode, load_generated, load_inverse, quantize, target  # noqa: E402
from tracer.signals import var_signals  # noqa: E402

FAMILY = "var"
SHARED_CKPT = CACHE / "stage2" / "inv" / "var" / "final.pt"
SIGNALS = ["quant", "enc1", "enc2", "cal", "comb", "tok_match"]
COLUMNS = ["label", "index", "inverse"] + SIGNALS + ["tok_recovery", "inv_mse"]
OVERLAP_MODEL = "var24"


def parse_single(s: str):
    """'NAME=PATH' -> (NAME, Path)."""
    name, sep, path = s.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {s!r}")
    return name, Path(path)


def load_holdout(limit, log):
    """Holdout images, labels, indices and true tokens, in gen.npz row order (label, then index).

    limit keeps the first N holdout rows (by index) of each depth. Returns a dict of numpy arrays and the full
    (unlimited) var24 holdout true tokens for the overlap check.
    """
    split = pd.read_csv(CACHE / "stage3" / "gen_split.csv")
    hold = split[(split.family == FAMILY) & (split.split == "holdout")][["label", "index"]]
    counts = hold.label.value_counts().sort_index()
    log(f"holdout rows per depth: {counts.to_dict()}")
    assert len(counts) == 4 and (counts == 512).all(), "expected 512 holdout rows for each of the 4 VAR depths"
    hold = hold.sort_values(["label", "index"]).reset_index(drop=True)

    g = np.load(CACHE / "stage3" / "tokens" / FAMILY / "gen.npz")
    gpos = pd.DataFrame({"label": g["label"], "index": g["index"], "gpos": np.arange(len(g["label"]))})
    hold = hold.merge(gpos, on=["label", "index"], how="left", validate="1:1")
    assert hold.gpos.notna().all(), "holdout rows missing from gen.npz"
    full_v24 = g["true_tokens"][hold.gpos[hold.label == OVERLAP_MODEL].astype(int).to_numpy()]

    if limit:
        hold = hold.groupby("label", sort=False).head(limit).reset_index(drop=True)
    gp = hold.gpos.astype(int).to_numpy()

    data = load_generated(CACHE / "stage3" / "gen", FAMILY)
    dpos = pd.DataFrame({"label": data["label"], "index": data["index"], "dpos": np.arange(len(data["label"]))})
    hold = hold.merge(dpos, on=["label", "index"], how="left", validate="1:1")
    assert hold.dpos.notna().all(), "holdout rows missing from stage3/gen"
    dp = hold.dpos.astype(int).to_numpy()

    true_tokens = g["true_tokens"][gp].astype(np.int64)
    assert np.array_equal(true_tokens, data["tokens"][dp]), "gen.npz true_tokens differ from stage3/gen tokens"
    assert np.array_equal(g["label"][gp], data["label"][dp]) and np.array_equal(g["index"][gp], data["index"][dp])
    return {"label": hold.label.to_numpy(str), "index": hold["index"].to_numpy(np.int64),
            "images": data["images"][dp], "true_tokens": true_tokens,
            "stored_tokens": g["tokens"][gp].astype(np.int64)}, full_v24


def check_overlap(holdout_v24: np.ndarray, log):
    """Assert no var24 holdout true-token sequence of stage3 appears in stage2/gen/var/var24."""
    s2 = load_generated(CACHE / "stage2" / "gen", FAMILY, models=[OVERLAP_MODEL])["tokens"]
    s2_keys = {r.tobytes() for r in s2.astype(np.int16)}
    n = sum(r.tobytes() in s2_keys for r in holdout_v24.astype(np.int16))
    log(f"overlap check: {n} of {len(holdout_v24)} stage3 {OVERLAP_MODEL} holdout token sequences appear among "
        f"{len(s2)} stage2/gen {OVERLAP_MODEL} sequences")
    assert n == 0, f"{n} stage3 {OVERLAP_MODEL} holdout sequences also appear in stage2/gen"


def git_state():
    """(commit hash, dirty flag) of the repo, or (None, None) if git is unavailable."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


@torch.no_grad()
def score(tok, hold: dict, batch_size: int, log):
    """Per-image scores of one inverse -> (DataFrame with SIGNALS + tok_recovery + inv_mse, recovered tokens)."""
    tok.eval()
    n = len(hold["label"])
    out = {k: [] for k in SIGNALS + ["tok_recovery", "inv_mse"]}
    rec = []
    t0, n_batches = time.time(), (n + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, n, batch_size)):
        x = torch.from_numpy(hold["images"][i:i + batch_size]).to(DEVICE).float().div_(255)
        t = torch.from_numpy(hold["true_tokens"][i:i + batch_size]).to(DEVICE)
        s = var_signals(tok, x, n_iters=0)
        f = encode(tok, FAMILY, x)  # D^-1(x)
        idx, _ = quantize(tok, FAMILY, f)  # greedy tokens [B,680], as recover_tokens
        for k in SIGNALS:
            out[k].append(s[k].float().cpu())
        out["tok_recovery"].append((idx == t).float().mean(1).cpu())
        out["inv_mse"].append((f - target(tok, FAMILY, t)).pow(2).flatten(1).mean(1).cpu())
        rec.append(idx.cpu())
        if (b + 1) % 8 == 0 or b + 1 == n_batches:
            log(f"  batch {b + 1}/{n_batches}, {time.time() - t0:.0f}s")
    df = pd.DataFrame({k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()})
    return df, torch.cat(rec).numpy()


def write_atomic(path: Path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.rename(path)


def main():
    """Parse arguments, check the data, and score every inverse not yet in --out."""
    ap = argparse.ArgumentParser(description="Stage 5 pilot: score the VAR generated holdout under several D^-1.")
    ap.add_argument("--single", type=parse_single, action="append", default=[], metavar="NAME=PATH",
                    help="a fine-tuned D^-1 checkpoint to score as inverse NAME (repeatable)")
    ap.add_argument("--batch-size", type=int, default=64, help="images per GPU batch")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N holdout images of each depth")
    ap.add_argument("--out", type=Path, default=CACHE / "stage5" / "scores.csv", help="output CSV")
    args = ap.parse_args()

    def log(s):
        print(s, flush=True)

    inverses = [("E", None), ("shared", SHARED_CKPT)] + args.single
    names = [n for n, _ in inverses]
    if len(set(names)) != len(names):
        ap.error(f"inverse names must be unique (E and shared are reserved): {names}")
    for name, path in inverses:
        if path is not None and not path.is_file():
            ap.error(f"{name}: checkpoint {path} not found")

    cfg_path = args.out.with_name(args.out.stem + ".config.json")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"limit": args.limit, "inverses": {}}
    if cfg["limit"] != args.limit:
        sys.exit(f"{cfg_path} was written with --limit {cfg['limit']}; use another --out or delete it")
    scores = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(columns=COLUMNS)
    unknown = set(scores.inverse) - set(cfg["inverses"])
    if unknown:
        sys.exit(f"{args.out} has inverses {sorted(unknown)} missing from {cfg_path}; delete both to recompute")

    todo = []
    for name, path in inverses:
        ckpt = None if path is None else str(path)
        prev = cfg["inverses"].get(name)
        if prev is not None and prev["ckpt"] != ckpt:
            sys.exit(f"inverse {name} was scored with {prev['ckpt']}, now {ckpt}; use another name or --out")
        if prev is not None and (scores.inverse == name).sum() == prev["n_rows"]:
            log(f"{name}: already scored ({prev['n_rows']} rows), skipping")
        else:
            todo.append((name, path))
    if not todo:
        log("nothing to do")
        return

    hold, full_v24 = load_holdout(args.limit, log)
    check_overlap(full_v24, log)
    log(f"scoring {len(hold['label'])} images under {[n for n, _ in todo]} on {DEVICE}")
    commit, dirty = git_state()
    command = shlex.join([sys.executable] + sys.argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for name, path in todo:
        t0 = time.time()
        tok = load_var_vae()
        if path is not None:
            load_inverse(tok, FAMILY, path)
        log(f"{name}: {'original encoder' if path is None else path}")
        df, rec = score(tok, hold, args.batch_size, log)
        del tok
        free()
        df.insert(0, "label", hold["label"])
        df.insert(1, "index", hold["index"])
        df.insert(2, "inverse", name)

        cfg["inverses"][name] = {"ckpt": None if path is None else str(path), "limit": args.limit, "n_rows": len(df),
                                 "git_commit": commit, "git_dirty": dirty, "command": command}
        # config first: if the run stops before the CSV is written, the inverse is simply rescored next time
        write_atomic(cfg_path, lambda p: p.write_text(json.dumps(cfg, indent=2)))
        kept = scores[scores.inverse != name]
        scores = pd.concat([kept, df[COLUMNS]] if len(kept) else [df[COLUMNS]], ignore_index=True)
        write_atomic(args.out, lambda p: scores.to_csv(p, index=False))

        log(f"{name}: done in {time.time() - t0:.0f}s; means per depth:\n"
            + df.groupby("label")[["tok_recovery", "inv_mse", "quant", "cal"]].mean().to_string(float_format="%.4g"))
        if name == "shared":
            same = rec == hold["stored_tokens"]
            stored_rec = (hold["stored_tokens"] == hold["true_tokens"]).mean(1).mean()
            log(f"sanity: shared tokens vs stored gen.npz recovered tokens (TF32, other batches): {same.mean():.4f} "
                f"of tokens and {same.all(1).mean():.4f} of images identical; mean tok_recovery {stored_rec:.4f} "
                f"stored vs {df.tok_recovery.mean():.4f} now")
    log(f"wrote {args.out} ({len(scores)} rows) and {cfg_path}")


if __name__ == "__main__":
    # TF32 convolutions (PyTorch default) flip 2-4% of greedy VAR tokens depending on batch size; full fp32 with fixed
    # cuDNN algorithms makes reruns at the same --batch-size identical and leaves ~1 token in 1e5 batch-dependent
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
