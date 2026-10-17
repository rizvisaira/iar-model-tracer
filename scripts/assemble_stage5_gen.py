"""Assemble the Stage 5 var24 training set from symlinks: the var24 shards of stage2/gen, then newly generated ones.

    stage5/gen/var/var24/shard_0000-0004  -> stage2/gen/var/var24/shard_0000-0004      (seed 0, A40)
    stage5/gen/var/var24/shard_0005-0019  -> stage5/gen_extra/var/var24/shard_0000-0014 (seed 10, this GPU)

so load_generated(<out>, "var") returns 10,240 var24 rows with index 0..10239. Nothing is copied or written
outside --out. Checks: both sources were generated with the same shard and batch size, every shard holds
--shard-size samples, each source's classes follow the generator's class cycling, and none of the extra source's
derived RNG seeds (class shuffle and per-shard sampling) was used for any model or shard of --used.
Writes <out>/var/config.json with the composition. A rerun with the same composition does nothing; a different one
exits.

    python scripts/assemble_stage5_gen.py --seed-check-only --extra-seed 10 --expect-extra 15   # before generating
    python scripts/assemble_stage5_gen.py --expect-extra 15
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tracer.common import CACHE  # noqa: E402
from tracer.generate import MODELS  # noqa: E402

FAMILY = "var"
BASE_GPU = "NVIDIA A40 (from README.md; not recorded at generation)"


# Derived seeds, exactly as in scripts/generate_finetune_data.py (m_i = position of the model in MODELS[family]).
def class_seed(seed, m_i):
    return seed + m_i


def shard_seed(seed, m_i, s):
    return seed * 1_000_003 + m_i * 10_007 + s


def class_cycle(seed, m_i, n):
    return np.resize(np.random.default_rng(class_seed(seed, m_i)).permutation(1000), n)


def shard_paths(d: Path):
    """shard_*.npz files of d in order; asserts they are shard_0000 ... shard_{n-1} without gaps."""
    paths = sorted(d.glob("shard_*.npz"))
    assert [p.name for p in paths] == [f"shard_{s:04d}.npz" for s in range(len(paths))], f"gaps in {d}"
    return paths


def used_seeds(roots):
    """{class seeds} and {shard seeds} of every model and existing shard under roots (all families)."""
    cls, shard = set(), set()
    for root in roots:
        for fam, labels in MODELS.items():
            cfg_path = Path(root) / fam / "config.json"
            if not cfg_path.exists():
                continue
            seed = json.loads(cfg_path.read_text())["seed"]
            for m_i, label in enumerate(labels):
                n = len(list((Path(root) / fam / label).glob("shard_*.npz")))
                if n:
                    cls.add(class_seed(seed, m_i))
                    shard.update(shard_seed(seed, m_i, s) for s in range(n))
    return cls, shard


def seed_check(seed, m_i, n_shards, roots):
    """Assert the extra source's derived seeds are new; return a summary for config.json."""
    used_cls, used_shard = used_seeds(roots)
    cs, ss = class_seed(seed, m_i), [shard_seed(seed, m_i, s) for s in range(n_shards)]
    clash_cls, clash_shard = cs in used_cls, sorted(set(ss) & used_shard)
    res = {"roots": [str(r) for r in roots], "class_seed": cs, "shard_seeds": [ss[0], ss[-1]],
           "used_class_seeds": sorted(used_cls), "n_used_shard_seeds": len(used_shard),
           "used_shard_seed_range": [min(used_shard), max(used_shard)],
           "class_seed_collision": clash_cls, "shard_seed_collisions": len(clash_shard)}
    print(f"seed check: class seed {cs} vs used {sorted(used_cls)}; shard seeds {ss[0]}..{ss[-1]} vs "
          f"{len(used_shard)} used in {min(used_shard)}..{max(used_shard)}; collisions: class {clash_cls}, "
          f"shard {len(clash_shard)}", flush=True)
    assert not clash_cls and not clash_shard, "derived seeds collide with an existing generated set"
    return res


def check_source(d: Path, cfg: dict, m_i: int, n_expect=None):
    """Shard paths of one source, with shard sizes and class cycling checked against its config."""
    paths = shard_paths(d)
    assert paths, f"no shards in {d}"
    assert n_expect is None or len(paths) == n_expect, f"{d}: {len(paths)} shards, expected {n_expect}"
    classes = []
    for p in paths:
        z = np.load(p)
        assert len(z["classes"]) == cfg["shard_size"] and z["tokens"].shape[0] == cfg["shard_size"], f"{p}: size"
        classes.append(z["classes"])
    classes = np.concatenate(classes).astype(np.int64)
    assert np.array_equal(classes, class_cycle(cfg["seed"], m_i, len(classes))), f"{d}: classes differ from seed"
    return paths


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True).stdout.strip().splitlines()[0] + " (detected at assembly)"
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


def main():
    """Check both sources and the seeds, then create the symlinked shard directory and its config.json."""
    ap = argparse.ArgumentParser(description="Assemble stage2 var24 shards + new var24 shards into one directory.")
    ap.add_argument("--label", default="var24", choices=MODELS[FAMILY])
    ap.add_argument("--base", type=Path, default=CACHE / "stage2" / "gen", help="first source (its shards come first)")
    ap.add_argument("--extra", type=Path, default=CACHE / "stage5" / "gen_extra", help="generate_finetune_data.py --out")
    ap.add_argument("--expect-extra", type=int, required=True, help="number of shards expected in --extra")
    ap.add_argument("--used", type=Path, nargs="+", default=[CACHE / "stage2" / "gen", CACHE / "stage3" / "gen"],
                    help="generated sets whose derived seeds the extra source must not reuse")
    ap.add_argument("--seed-check-only", action="store_true", help="only check --extra-seed against --used, then exit")
    ap.add_argument("--extra-seed", type=int, default=None, help="with --seed-check-only: the planned --seed")
    ap.add_argument("--out", type=Path, default=CACHE / "stage5" / "gen")
    args = ap.parse_args()
    m_i = MODELS[FAMILY].index(args.label)

    if args.seed_check_only:
        if args.extra_seed is None:
            ap.error("--seed-check-only needs --extra-seed")
        seed_check(args.extra_seed, m_i, args.expect_extra, args.used)
        return

    base_cfg = json.loads((args.base / FAMILY / "config.json").read_text())
    extra_cfg = json.loads((args.extra / FAMILY / "config.json").read_text())
    for k in ("family", "shard_size", "batch_size"):
        assert base_cfg[k] == extra_cfg[k], f"{k} differs: base {base_cfg[k]}, extra {extra_cfg[k]}"
    seeds = seed_check(extra_cfg["seed"], m_i, args.expect_extra, args.used)
    base = check_source(args.base / FAMILY / args.label, base_cfg, m_i)
    extra = check_source(args.extra / FAMILY / args.label, extra_cfg, m_i, args.expect_extra)
    links = {f"shard_{s:04d}.npz": str(p.resolve()) for s, p in enumerate(base + extra)}

    dst = args.out / FAMILY / args.label
    cfg_path = args.out / FAMILY / "config.json"
    if dst.exists():
        have = {p.name: os.readlink(p) for p in dst.iterdir()}
        if have == links and cfg_path.exists():
            print(f"{dst} already assembled ({len(links)} shards), nothing to do")
            return
        sys.exit(f"{dst} exists with a different composition; delete {args.out / FAMILY} to reassemble")

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, target in links.items():
        (tmp / name).symlink_to(target)
    ss = base_cfg["shard_size"]

    def source(cfg, d, paths, first, gpu):
        return {"dir": str((d / FAMILY / args.label).resolve()), "seed": cfg["seed"], "batch_size": cfg["batch_size"],
                "shard_size": ss, "n_shards": len(paths), "n_rows": len(paths) * ss, "gpu": gpu,
                "out_shards": [f"shard_{first:04d}.npz", f"shard_{first + len(paths) - 1:04d}.npz"],
                "out_index": [first * ss, (first + len(paths)) * ss - 1]}

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"family": FAMILY, "label": args.label, "shard_size": ss, "n_rows": len(links) * ss,
           "sources": [source(base_cfg, args.base, base, 0, BASE_GPU),
                       source(extra_cfg, args.extra, extra, len(base), gpu_name())],
           "seed_check": seeds, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "git_commit": commit or None, "command": " ".join([sys.executable] + sys.argv)}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    tmp.rename(dst)
    print(f"assembled {dst}: {len(base)} + {len(extra)} shards = {len(links) * ss} rows; wrote {cfg_path}")


if __name__ == "__main__":
    main()
