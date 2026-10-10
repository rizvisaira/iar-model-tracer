"""Generator likelihood (Stage 4): per-token log-probabilities of known tokens under a variant's transformer.

Teacher forcing: the whole known token sequence goes through the transformer once per condition (class c and the
null class), both conditions batched together, and the guided sampling distribution of the repo's sampler is
rebuilt from the two sets of logits l_c, l_u at every position:

    rar   (1d-tokenizer modeling/rar.py, RAR.generate)
          l = (l_u + (l_c - l_u) * g_n) / temperature,  g_n = (w - 1) * (1 - cos(pi * (n / 256) ** pow)) / 2 + 1
          for token n = 0..255 (raster order), (w, pow, temperature) = generate.RAR_SAMPLING per size.
    var   (VAR models/var.py, VAR.autoregressive_infer_cfg)
          l = (1 + t) * l_c - t * l_u,  t = cfg * si / 9 for every token of scale si = 0..9, cfg = 1.5;
          the sampler then keeps the top_k = 900 / top_p = 0.96 tokens (models/helpers.sample_with_top_k_top_p_).
          logp_guided here is log softmax(l) WITHOUT that filter (always finite); in_support says whether the
          token survives it.

Which logit predicts which token:
    rar   forward_fn prepends the condition token (and a cls token, dropped before the head), so with input tokens
          t_0..t_{k-1} it returns k + 1 logit vectors; position j is computed from [condition, t_0..t_{j-1}] and
          predicts t_j (generate() samples token n from position n, the last one). Feeding t_0..t_254 gives
          the 256 positions that predict t_0..t_255. The head has exactly codebook_size = 1024 outputs (the extra
          embedding rows for the mask token, classes and null class are inputs only), so nothing is dropped.
    var   forward returns one logit vector per token, 680 in total; position p of scale si sees the class (sos) and
          the upsampled f_hat of scales < si (block-causal mask) and predicts token p, as in the sampler.

Models: rar  generate.load_rar_generator(label)          -> the RAR module
        var  generate.load_var_generator(label)          -> the (vae, var) tuple
"""
import numpy as np
import torch
import torch.nn.functional as F

from .common import DEVICE
from .generate import RAR_SAMPLING, RAR_SIZES, VAR_DEPTHS, VAR_SAMPLING, split_var_tokens


# ---------------------------------------------------------------- guidance, as in the samplers

def rar_guidance_scales(label: str, seq_len: int = 256, device=DEVICE) -> torch.Tensor:
    """g_n for n = 0..seq_len-1, computed with the same tensor arithmetic as RAR.generate. Returns [seq_len] float."""
    guidance_scale, guidance_scale_pow, _ = RAR_SAMPLING[RAR_SIZES[label]]
    out = []
    for step in range(seq_len):
        scale_pow = torch.ones((1), device=device) * guidance_scale_pow
        scale_step = (1 - torch.cos(((step / seq_len) ** scale_pow) * torch.pi)) * 1 / 2
        out.append((guidance_scale - 1) * scale_step + 1)
    return torch.cat(out)


def top_k_top_p_keep(logits: torch.Tensor, top_k: int = VAR_SAMPLING["top_k"],
                     top_p: float = VAR_SAMPLING["top_p"]) -> torch.Tensor:
    """Bool mask [..., V] of the entries VAR's sample_with_top_k_top_p_ keeps (same operations, on a copy)."""
    logits = logits.clone()
    if top_k > 0:
        idx_to_remove = logits < logits.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove),
                            -torch.inf)
    return ~torch.isneginf(logits)


# ---------------------------------------------------------------- teacher-forced logits

@torch.no_grad()
def _rar_logits(gen, label: str, tokens: torch.Tensor, classes: torch.Tensor) -> dict:
    B, L = tokens.shape
    assert L == gen.image_seq_len, f"RAR tokens must be [B,{gen.image_seq_len}]"
    assert not gen.blocks[0].attn.kv_cache, "disable the kv cache (gen.disable_kv_cache()) before scoring"
    cond = gen.preprocess_condition(classes, cond_drop_prob=0.0)
    inp = tokens[:, :-1]  # position j predicts token j from [condition, t_0..t_{j-1}]
    logits = gen.forward_fn(torch.cat([inp, inp]), torch.cat([cond, gen.get_none_condition(cond)]),
                            orders=None, is_sampling=True)
    assert logits.shape == (2 * B, L, gen.target_codebook_size), logits.shape
    l_c, l_u = logits[:B], logits[B:]
    _, _, temperature = RAR_SAMPLING[RAR_SIZES[label]]
    g = rar_guidance_scales(label, L, tokens.device)[None, :, None]
    return {"cond": l_c, "uncond": l_u, "guided": (l_u + (l_c - l_u) * g) / temperature}


@torch.no_grad()
def _var_logits(vae, var, tokens: torch.Tensor, classes: torch.Tensor, amp: bool) -> dict:
    B, L = tokens.shape
    assert L == var.L, f"VAR tokens must be [B,{var.L}]"
    assert var.prog_si == -1 and vae.quantize.prog_si == -1, "progressive training must be off (prog_si = -1)"
    x = vae.quantize.idxBl_to_var_input(split_var_tokens(tokens))  # [B, L - 1, Cvae], f_hat inputs of scales 1..9
    labels = torch.cat([classes, torch.full_like(classes, var.num_classes)])  # class, then the null class
    drop = var.cond_drop_rate
    var.cond_drop_rate = 0.0  # VAR.forward drops labels at this rate even in eval mode
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and DEVICE == "cuda"):
            logits = var(labels, torch.cat([x, x])).float()
    finally:
        var.cond_drop_rate = drop
    assert logits.shape == (2 * B, L, var.V), logits.shape
    l_c, l_u = logits[:B], logits[B:]
    guided = torch.empty_like(l_c)
    for si, (a, b) in enumerate(var.begin_ends):  # python-float guidance per scale, as autoregressive_infer_cfg
        t = VAR_SAMPLING["cfg"] * (si / var.num_stages_minus_1)
        guided[:, a:b] = (1 + t) * l_c[:, a:b] - t * l_u[:, a:b]
    return {"cond": l_c, "uncond": l_u, "guided": guided}


def _as_long(x) -> torch.Tensor:
    x = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
    return x.to(DEVICE).long()


@torch.no_grad()
def guided_logits(model, family: str, label: str, tokens, classes, amp: bool = False) -> dict:
    """Teacher-forced logits for one batch: {"cond", "uncond", "guided"}, each [B,L,V] float32 on DEVICE.

    model: the RAR module (rar) or the (vae, var) tuple (var). tokens [B,L], classes [B] (ImageNet ids 0..999).
    "guided" is the sampler's pre-softmax logits: for RAR already divided by the temperature, for VAR before the
    top-k / top-p filter. amp (VAR only) runs the transformer under fp16 autocast, like generate.sample_var.
    """
    tokens, classes = _as_long(tokens), _as_long(classes)
    if family == "rar":
        assert label in RAR_SIZES, label
        return _rar_logits(model, label, tokens, classes)
    assert label in VAR_DEPTHS, label
    vae, var = model
    assert var.depth == VAR_DEPTHS[label], f"{label} but the model has depth {var.depth}"
    return _var_logits(vae, var, tokens, classes, amp)


def _at(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return logits.gather(-1, tokens[..., None])[..., 0]


@torch.no_grad()
def score_tokens(model, family: str, label: str, tokens, classes, batch_size: int = 32, amp: bool = False) -> dict:
    """Per-token scores of known tokens under one variant: dict of float32 numpy arrays [B,L].

    logp_cond     log softmax(l_c)[t], class-conditional
    logp_uncond   log softmax(l_u)[t], null class
    logp_guided   log softmax(l)[t] under the guided sampling distribution (VAR: without top-k / top-p)
    rank_guided   number of codes with a strictly larger guided logit than t (0 = most likely)
    in_support    VAR only: 1 if t survives the sampler's top_k = 900 / top_p = 0.96 filter of l, else 0

    model: the RAR module or the (vae, var) tuple, for variant `label`. tokens [B,L] (RAR 256, VAR 680, layout as in
    tracer.generate), classes [B]. Processed in chunks of batch_size images (2 * batch_size sequences per pass).
    amp: VAR only, fp16 autocast as in the sampler (default fp32).
    """
    tokens, classes = _as_long(tokens), _as_long(classes)
    keys = ["logp_cond", "logp_uncond", "logp_guided", "rank_guided"] + (["in_support"] if family == "var" else [])
    out = {k: [] for k in keys}
    for i in range(0, len(tokens), batch_size):
        t = tokens[i:i + batch_size]
        lg = guided_logits(model, family, label, t, classes[i:i + batch_size], amp=amp)
        res = {"logp_cond": _at(F.log_softmax(lg["cond"], -1), t),
               "logp_uncond": _at(F.log_softmax(lg["uncond"], -1), t),
               "logp_guided": _at(F.log_softmax(lg["guided"], -1), t),
               "rank_guided": (lg["guided"] > _at(lg["guided"], t)[..., None]).sum(-1)}
        if family == "var":
            res["in_support"] = _at(top_k_top_p_keep(lg["guided"]), t)
        for k in keys:
            out[k].append(res[k].float().cpu().numpy())
        del lg
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
