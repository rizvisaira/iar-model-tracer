"""Assemble Stage 5 training sets from symlinks: a model's shards of stage2/gen, then newly generated ones.

    stage5/gen/var/var24/shard_0000-0004  -> stage2/gen/var/var24/shard_0000-0004      (seed 0, A40)
    stage5/gen/var/var24/shard_0005-0019  -> stage5/gen_extra/var/var24/shard_0000-0014 (seed 10, this GPU)

so load_generated(<out>, "var") returns 10,240 var24 rows with index 0..10239. The defaults assemble the pilot's
var24 set; Stage 5 full assembles all four models of a family in one call (--family, --labels), with --extra-for
giving a model another extra source (var24: the pilot's stage5/gen_extra). Nothing is copied or written outside
--out. Checks: both sources were generated with the same shard and batch size, every shard holds --shard-size
samples, each source's classes follow the generator's class cycling, and none of an extra source's derived RNG seeds
(class shuffle and per-shard sampling) was used for any model and shard of --used (every family, or with
--seed-scope family only --family; a --used set that is the source itself is skipped) or by another extra source of
the same call. Writes <out>/<family>/config.json with the composition once every model folder is in place. A rerun
with the same composition does nothing; a different one exits.

    python scripts/assemble_stage5_gen.py --seed-check-only --extra-seed 10 --expect-extra 15   # before generating
    python scripts/assemble_stage5_gen.py --expect-extra 15
    # Stage 5 full, VAR (scripts/run_stage5_full.sh)
    python scripts/assemble_stage5_gen.py --family var --labels var16 var20 var24 var30 --seed-scope family \\
        --extra /workspace/cache/stage5full/gen_extra --expect-extra 15 \\
        --extra-for var24 /workspace/cache/stage5/gen_extra 15 \\
        --used /workspace/cache/stage2/gen /workspace/cache/stage3/gen /workspace/cache/stage5/gen_extra \\
        --out /workspace/cache/stage5full/gen
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


def used_seeds(roots, family=None):
    """{class seeds} and {shard seeds} of every model and existing shard under roots (all families, or only family)."""
    cls, shard = set(), set()
    for root in roots:
        for fam, labels in MODELS.items():
            cfg_path = Path(root) / fam / "config.json"
            if (family is not None and fam != family) or not cfg_path.exists():
                continue
            seed = json.loads(cfg_path.read_text())["seed"]
            for m_i, label in enumerate(labels):
                n = len(list((Path(root) / fam / label).glob("shard_*.npz")))
                if n:
                    cls.add(class_seed(seed, m_i))
                    shard.update(shard_seed(seed, m_i, s) for s in range(n))
    return cls, shard


def seed_check(seed, m_i, n_shards, roots, family=None):
    """Assert the extra source's derived seeds are new; return a summary for config.json.

    family=None compares with every family under roots (the pilot's check); a family name only with that family.
    """
    used_cls, used_shard = used_seeds(roots, family)
    cs, ss = class_seed(seed, m_i), [shard_seed(seed, m_i, s) for s in range(n_shards)]
    clash_cls, clash_shard = cs in used_cls, sorted(set(ss) & used_shard)
    rng = [min(used_shard), max(used_shard)] if used_shard else None
    res = {"roots": [str(r) for r in roots], "class_seed": cs, "shard_seeds": [ss[0], ss[-1]],
           "used_class_seeds": sorted(used_cls), "n_used_shard_seeds": len(used_shard),
           "used_shard_seed_range": rng, "class_seed_collision": clash_cls, "shard_seed_collisions": len(clash_shard)}
    if family is not None:
        res["scope"] = family
    where = f"{rng[0]}..{rng[1]}" if rng else "none"
    print(f"seed check{f' ({family})' if family else ''}: class seed {cs} vs used {sorted(used_cls)}; shard seeds "
          f"{ss[0]}..{ss[-1]} vs {len(used_shard)} used in {where}; collisions: class {clash_cls}, "
          f"shard {len(clash_shard)}", flush=True)
    assert not clash_cls and not clash_shard, "derived seeds collide with an existing generated set"
    return res


def check_disjoint(sources):
    """Assert no two extra sources of one call share a derived seed; sources: [(label, seed, m_i, n_shards)]."""
    cls, shard = {}, {}
    for label, seed, m_i, n in sources:
        cs = class_seed(seed, m_i)
        assert cs not in cls, f"{label}: class seed {cs} is also used by {cls[cs]}"
        cls[cs] = label
        for s in range(n):
            ss = shard_seed(seed, m_i, s)
            assert ss not in shard, f"{label}: shard seed {ss} is also used by {shard[ss]}"
            shard[ss] = label


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
    """Check the sources and the seeds, then create the symlinked shard directories and their config.json."""
    ap = argparse.ArgumentParser(description="Assemble stage2 shards + newly generated shards into one directory.")
    ap.add_argument("--family", default="var", choices=list(MODELS))
    ap.add_argument("--label", default="var24", help="model folder to assemble (one of --family's models)")
    ap.add_argument("--labels", nargs="+", default=None, help="several model folders of --family (default: --label)")
    ap.add_argument("--base", type=Path, default=CACHE / "stage2" / "gen", help="first source (its shards come first)")
    ap.add_argument("--extra", type=Path, default=CACHE / "stage5" / "gen_extra", help="generate_finetune_data.py --out")
    ap.add_argument("--expect-extra", type=int, required=True, help="number of shards expected in --extra")
    ap.add_argument("--extra-for", nargs=3, action="append", default=[], metavar=("LABEL", "DIR", "N_SHARDS"),
                    help="use DIR (with N_SHARDS shards) instead of --extra as LABEL's second source (repeatable)")
    ap.add_argument("--used", type=Path, nargs="+", default=[CACHE / "stage2" / "gen", CACHE / "stage3" / "gen"],
                    help="generated sets whose derived seeds the extra sources must not reuse")
    ap.add_argument("--seed-scope", choices=["all", "family"], default="all",
                    help="compare seeds with every family of --used (all) or only with --family (family)")
    ap.add_argument("--seed-check-only", action="store_true", help="only check --extra-seed against --used, then exit")
    ap.add_argument("--extra-seed", type=int, default=None, help="with --seed-check-only: the planned --seed")
    ap.add_argument("--out", type=Path, default=CACHE / "stage5" / "gen")
    args = ap.parse_args()
    fam = args.family
    labels = args.labels or [args.label]
    if not set(labels) <= set(MODELS[fam]) or len(set(labels)) != len(labels):
        ap.error(f"labels must be distinct models of {fam}: {MODELS[fam]}")
    extra_for = {}
    for label, d, n in args.extra_for:
        if label not in labels:
            ap.error(f"--extra-for {label}: not in {labels}")
        extra_for[label] = (Path(d), int(n))
    scope = fam if args.seed_scope == "family" else None

    if args.seed_check_only:
        if args.extra_seed is None:
            ap.error("--seed-check-only needs --extra-seed")
        if extra_for:
            ap.error("--seed-check-only checks planned sources only; drop --extra-for")
        for label in labels:
            seed_check(args.extra_seed, MODELS[fam].index(label), args.expect_extra, args.used, scope)
        check_disjoint([(l, args.extra_seed, MODELS[fam].index(l), args.expect_extra) for l in labels])
        return

    base_cfg = json.loads((args.base / fam / "config.json").read_text())
    ss = base_cfg["shard_size"]
    gpu = gpu_name()
    records, links, planned = {}, {}, []
    for label in labels:
        m_i = MODELS[fam].index(label)
        extra_dir, n_extra = extra_for.get(label, (args.extra, args.expect_extra))
        extra_cfg = json.loads((extra_dir / fam / "config.json").read_text())
        for k in ("family", "shard_size", "batch_size"):
            assert base_cfg[k] == extra_cfg[k], f"{label}: {k} differs: base {base_cfg[k]}, extra {extra_cfg[k]}"
        used = [r for r in args.used if r.resolve() != extra_dir.resolve()]  # a reused source is not its own clash
        seeds = seed_check(extra_cfg["seed"], m_i, n_extra, used, scope)
        planned.append((label, extra_cfg["seed"], m_i, n_extra))
        base = check_source(args.base / fam / label, base_cfg, m_i)
        extra = check_source(extra_dir / fam / label, extra_cfg, m_i, n_extra)
        links[label] = {f"shard_{s:04d}.npz": str(p.resolve()) for s, p in enumerate(base + extra)}

        def source(cfg, d, paths, first, gpu):
            return {"dir": str((d / fam / label).resolve()), "seed": cfg["seed"], "batch_size": cfg["batch_size"],
                    "shard_size": ss, "n_shards": len(paths), "n_rows": len(paths) * ss, "gpu": gpu,
                    "out_shards": [f"shard_{first:04d}.npz", f"shard_{first + len(paths) - 1:04d}.npz"],
                    "out_index": [first * ss, (first + len(paths)) * ss - 1]}

        records[label] = {"family": fam, "label": label, "shard_size": ss, "n_rows": len(links[label]) * ss,
                          "sources": [source(base_cfg, args.base, base, 0, BASE_GPU),
                                      source(extra_cfg, extra_dir, extra, len(base), gpu)],
                          "seed_check": seeds}
    check_disjoint(planned)

    # Existing output: same labels and links -> nothing to do (or only config.json missing); anything else exits.
    cfg_path = args.out / fam / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get("labels", [old.get("label")]) != labels:
            sys.exit(f"{cfg_path} lists other models; delete {args.out / fam} to reassemble")
    todo = []
    for label in labels:
        dst = args.out / fam / label
        if not dst.exists():
            todo.append(label)
        elif {p.name: os.readlink(p) for p in dst.iterdir()} != links[label]:
            sys.exit(f"{dst} exists with a different composition; delete {args.out / fam} to reassemble")
    if not todo and cfg_path.exists():
        print(f"{args.out / fam} already assembled ({', '.join(labels)}), nothing to do")
        return

    for label in todo:
        dst = args.out / fam / label
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, target in links[label].items():
            (tmp / name).symlink_to(target)
        tmp.rename(dst)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit or None,
            "command": " ".join([sys.executable] + sys.argv)}
    if len(labels) == 1:
        cfg = {**records[labels[0]], **meta}  # the pilot's schema
    else:
        cfg = {"family": fam, "labels": labels, "seed_scope": args.seed_scope, "models": records, **meta}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    for label in labels:
        r = records[label]
        print(f"assembled {args.out / fam / label}: " + " + ".join(str(s["n_shards"]) for s in r["sources"])
              + f" shards = {r['n_rows']} rows", flush=True)
    print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
