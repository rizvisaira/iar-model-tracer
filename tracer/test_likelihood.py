"""Consistency checks for tracer/likelihood.py (GPU, about 5 minutes):

    python -m tracer.test_likelihood

1. Teacher forcing equals the samplers. For 4 generated samples of RAR-L and of VAR-d16, the teacher-forced guided
   logits are compared with the ones the repo's sampler computes when it is forced to emit the same tokens:
     rar  (a) a step-by-step loop without kv cache, written exactly like RAR.generate (fp32);
          (b) RAR.generate itself (kv cache), with torch.multinomial replaced by the known token; compares the
              sampling probabilities it passes in.
     var  VAR.autoregressive_infer_cfg itself (kv cache, fp16 autocast as in generate.sample_var), with
          sample_with_top_k_top_p_ replaced by the known tokens; compares the guided logits it receives and its own
          top-k / top-p mask, against teacher forcing in fp32 and in fp16.
2. Own-model support. 64 generated samples per VAR depth (true tokens and classes, generated TRAIN split only),
   scored by all 4 depths: the fraction of tokens in the top-k / top-p support of the scoring model's guided
   distribution must be ~1.0 when the scoring model generated them.
Samples are read from stage3/gen and taken from the generated train split (stage3/gen_split.csv), so the
Stage 4 holdout stays untouched. Results are also saved to <cache>/stage4/tests/test_likelihood.json (check 1) and
test_likelihood_support.csv (check 2), for notebooks/004_stage_4.ipynb.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import CACHE, DEVICE, free
from .generate import RAR_SAMPLING, RAR_SIZES, VAR_DEPTHS, VAR_SAMPLING, load_rar_generator, load_var_generator
from .likelihood import guided_logits, score_tokens, top_k_top_p_keep

GEN3 = CACHE / "stage3" / "gen"
SPLIT = CACHE / "stage3" / "gen_split.csv"
OUT = CACHE / "stage4" / "tests"


def load_train_samples(family: str, label: str, n: int):
    """First n generated-train samples of one model from stage3/gen: tokens [n,L] and classes [n], long on DEVICE."""
    shards = sorted((GEN3 / family / label).glob("shard_*.npz"))
    tok, cls = [], []
    for p in shards:
        with np.load(p) as z:
            tok.append(z["tokens"].astype(np.int64)), cls.append(z["classes"].astype(np.int64))
    tok, cls = np.concatenate(tok), np.concatenate(cls)  # row i = sample index i, as in inverse.load_generated
    split = pd.read_csv(SPLIT)
    idx = np.sort(split.query("family == @family and label == @label and split == 'train'")["index"].values)[:n]
    assert len(idx) == n
    return torch.from_numpy(tok[idx]).to(DEVICE), torch.from_numpy(cls[idx]).to(DEVICE)


def _logp_at(logits, tokens):
    return F.log_softmax(logits.float(), -1).gather(-1, tokens[..., None])[..., 0]


# ---------------------------------------------------------------- check 1

@torch.no_grad()
def check_rar_teacher_forcing(label: str = "rarl", n: int = 4) -> dict:
    gen = load_rar_generator(label)
    tokens, classes = load_train_samples("rar", label, n)
    w, pw, temp = RAR_SAMPLING[RAR_SIZES[label]]
    tf = guided_logits(gen, "rar", label, tokens, classes)["guided"]
    scores = score_tokens(gen, "rar", label, tokens, classes)

    # (a) step by step, no kv cache, exactly the arithmetic of RAR.generate
    cond = gen.preprocess_condition(classes, cond_drop_prob=0.0)
    ids, steps = tokens[:, :0], []
    for step in range(gen.image_seq_len):
        scale_pow = torch.ones((1), device=DEVICE) * pw
        scale_step = (1 - torch.cos(((step / gen.image_seq_len) ** scale_pow) * torch.pi)) * 1 / 2
        cfg_scale = (w - 1) * scale_step + 1
        logits = gen.forward_fn(torch.cat([ids, ids], dim=0), torch.cat([cond, gen.get_none_condition(cond)], dim=0),
                                orders=None, is_sampling=True)
        cond_logits, uncond_logits = logits[:n], logits[n:]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        steps.append(logits[:, -1] / temp)
        ids = torch.cat((ids, tokens[:, step:step + 1]), dim=-1)
    step_logits = torch.stack(steps, 1)

    # (b) RAR.generate itself, kv cache on, with the sampled token replaced by the known one
    probs, orig = [], torch.multinomial

    def forced(p, num_samples, *args, **kwargs):
        i = len(probs)
        probs.append(p.clone())
        return tokens[:, i:i + 1]

    torch.multinomial = forced
    try:
        out = gen.generate(condition=classes, guidance_scale=w, randomize_temperature=temp, guidance_scale_pow=pw)
    finally:
        torch.multinomial = orig
    assert (out == tokens).all()
    gen_probs = torch.stack(probs, 1)
    tf_probs = F.softmax(tf, -1)
    true_p = gen_probs.gather(-1, tokens[..., None])[..., 0]

    scores_t = torch.from_numpy(scores["logp_guided"]).to(DEVICE)
    res = {"family": "rar", "label": label, "logits": tuple(tf.shape), "|guided logit| max": tf.abs().max().item(),
           "(a) max |dlogit| step-by-step": (tf - step_logits).abs().max().item(),
           "(a) max |dlogp(t)| step-by-step": (scores_t - _logp_at(step_logits, tokens)).abs().max().item(),
           "(b) max |dprob| generate()": (tf_probs - gen_probs).abs().max().item(),
           "(b) max |dlogp(t)| generate()": (scores_t - true_p.log()).abs().max().item(),
           "rank 0 share (argmax = token)": float((scores["rank_guided"] == 0).mean())}
    del gen
    free()
    return res


@torch.no_grad()
def check_var_teacher_forcing(label: str = "var16", n: int = 4) -> dict:
    vae, var = load_var_generator(label)
    tokens, classes = load_train_samples("var", label, n)
    import models.var as var_module  # VAR repo is on sys.path after load_var_generator

    logits_seen, masks_seen = [], []
    orig, rng = var_module.sample_with_top_k_top_p_, torch.Generator(device=DEVICE)

    def forced(logits_BlV, top_k=0, top_p=0.0, rng_=None, num_samples=1, **kwargs):
        si = len(logits_seen)
        logits_seen.append(logits_BlV.clone())
        masked = logits_BlV.clone()
        orig(masked, top_k=top_k, top_p=top_p, rng=rng, num_samples=1)  # repo's own in-place filter
        masks_seen.append(~torch.isneginf(masked))
        a, b = var.begin_ends[si]
        return tokens[:, a:b, None]

    var_module.sample_with_top_k_top_p_ = forced
    try:
        with torch.autocast("cuda", dtype=torch.float16):  # as generate.sample_var
            var.autoregressive_infer_cfg(B=n, label_B=classes, **VAR_SAMPLING)
    finally:
        var_module.sample_with_top_k_top_p_ = orig
    samp = torch.cat(logits_seen, 1)
    samp_mask = torch.cat(masks_seen, 1)
    samp_in = samp_mask.gather(-1, tokens[..., None])[..., 0]

    res = {"family": "var", "label": label, "logits": tuple(samp.shape), "|guided logit| max": samp.abs().max().item(),
           "own top-k/top-p mask == repo mask (all codes)": bool((top_k_top_p_keep(samp) == samp_mask).all()),
           "sampler in_support at t": samp_in.float().mean().item()}
    for name, amp in [("fp32", False), ("fp16", True)]:
        tf = guided_logits((vae, var), "var", label, tokens, classes, amp=amp)["guided"]
        sc = score_tokens((vae, var), "var", label, tokens, classes, amp=amp)
        d = (tf - samp).abs()
        res.update({f"{name}: max |dlogit|": d.max().item(), f"{name}: mean |dlogit|": d.mean().item(),
                    f"{name}: max |dlogp(t)|": (torch.from_numpy(sc["logp_guided"]).to(DEVICE)
                                               - _logp_at(samp, tokens)).abs().max().item(),
                    f"{name}: in_support agrees with sampler": float((sc["in_support"] == samp_in.cpu().numpy()).mean()),
                    f"{name}: in_support": float(sc["in_support"].mean())})
        del tf
    del vae, var
    free()
    return res


# ---------------------------------------------------------------- check 2

@torch.no_grad()
def check_var_own_support(n: int = 64) -> pd.DataFrame:
    samples = {lab: load_train_samples("var", lab, n) for lab in VAR_DEPTHS}
    rows = []
    for scorer in VAR_DEPTHS:
        vae, var = load_var_generator(scorer)
        for gen_label, (tokens, classes) in samples.items():
            sc = score_tokens((vae, var), "var", scorer, tokens, classes)
            row = {"scoring model": scorer, "generated by": gen_label, "in_support": float(sc["in_support"].mean()),
                   "images fully in support": float((sc["in_support"].min(1) == 1).mean())}
            if scorer == gen_label:
                sc16 = score_tokens((vae, var), "var", scorer, tokens, classes, amp=True)
                row["in_support (fp16)"] = float(sc16["in_support"].mean())
            rows.append(row)
        del vae, var
        free()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pd.set_option("display.width", 200)

    print("== 1. teacher forcing vs the samplers")
    r = check_rar_teacher_forcing()
    for k, v in r.items():
        print(f"  {k:45s} {v}")
    assert r["(a) max |dlogp(t)| step-by-step"] < 1e-3 and r["(b) max |dlogp(t)| generate()"] < 1e-3, "RAR mismatch"
    v = check_var_teacher_forcing()
    for k, val in v.items():
        print(f"  {k:45s} {val}")
    assert v["own top-k/top-p mask == repo mask (all codes)"], "top-k / top-p reimplementation differs"
    assert v["fp32: max |dlogp(t)|"] < 0.1 and v["fp16: max |dlogp(t)|"] < 0.1, "VAR mismatch beyond fp16 noise"

    print("\n== 2. VAR own-model support (64 generated-train samples per depth, true tokens and classes)")
    s = check_var_own_support()
    print(s.pivot(index="scoring model", columns="generated by", values="in_support").round(4).to_string())
    diag = s[s["scoring model"] == s["generated by"]].set_index("scoring model")
    print(diag[["in_support", "in_support (fp16)", "images fully in support"]].round(4).to_string())
    assert (diag["in_support"] > 0.99).all(), "own-model support is not ~1"

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", "tf32": False}
    (OUT / "test_likelihood.json").write_text(json.dumps({"meta": meta, "rar": r, "var": v}, indent=2, default=str))
    s.to_csv(OUT / "test_likelihood_support.csv", index=False)
    print(f"\nall checks passed; results in {OUT}")
