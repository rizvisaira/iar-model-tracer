"""Sample images from the RAR / VAR generators, keeping the tokens, for inverse-decoder fine-tuning (Stage 2).

Each sample is (tokens t_Z, image x_Z = D(Q^-1(t_Z))), with the tokenizer's frozen decoder D. Images are stored
as uint8 like the task PNGs; both repos save samples with x.mul(255) -> uint8 (truncation), which is copied here.

Sampling settings are each repo's ImageNet FID settings (RAR: per size, from README_RAR.md; VAR: README.md). The
task's own generation settings are unknown, so these are an assumption.
"""
from contextlib import contextmanager

import torch

from .common import DEVICE, EXTERNAL, WEIGHTS, _switch_repo

RAR_SIZES = {"rarb": "b", "rarl": "l", "rarxl": "xl", "rarxxl": "xxl"}  # label -> weight file suffix
VAR_DEPTHS = {"var16": 16, "var20": 20, "var24": 24, "var30": 30}  # label -> transformer depth
MODELS = {"rar": list(RAR_SIZES), "var": list(VAR_DEPTHS)}
VAR_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

# RAR architecture per size: hidden_size, num_hidden_layers, intermediate_size (README_RAR.md)
RAR_ARCH = {"b": (768, 24, 3072), "l": (1024, 24, 4096), "xl": (1280, 32, 5120), "xxl": (1408, 40, 6144)}
# RAR FID sampling settings per size: guidance_scale, guidance_scale_pow, randomize_temperature (README_RAR.md)
RAR_SAMPLING = {"b": (16.0, 2.75, 1.0), "l": (15.5, 2.5, 1.02), "xl": (6.9, 1.5, 1.02), "xxl": (8.0, 1.2, 1.02)}
# VAR FID sampling settings (README.md), shared by all depths
VAR_SAMPLING = dict(cfg=1.5, top_k=900, top_p=0.96, more_smooth=False)


def to_uint8(x01: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> uint8, truncating like the repos' PNG saving (x * 255 -> uint8)."""
    return (x01.clamp(0, 1) * 255).to(torch.uint8)


def load_rar_generator(label: str):
    """RAR generator for one size, e.g. "rarxl". Frozen, on DEVICE, with random-order training disabled."""
    _switch_repo(EXTERNAL / "1d-tokenizer")
    from omegaconf import OmegaConf
    from modeling.rar import RAR

    size = RAR_SIZES[label]
    cfg = OmegaConf.load(EXTERNAL / "1d-tokenizer/configs/training/generator/rar.yaml")
    g = cfg.model.generator
    g.hidden_size, g.num_hidden_layers, g.intermediate_size = RAR_ARCH[size]
    g.num_attention_heads = 16
    model = RAR(cfg)
    model.load_state_dict(torch.load(WEIGHTS / f"rar/rar_{size}.bin", map_location="cpu"))
    model.set_random_ratio(0)  # sample in raster order, as in the repo's demo_util.get_rar_generator
    return model.to(DEVICE).eval().requires_grad_(False)


def load_var_generator(label: str):
    """(vae, var) for one depth, e.g. "var16". Both frozen, on DEVICE. The VQVAE is the shared tokenizer."""
    _switch_repo(EXTERNAL / "VAR")
    from models import build_vae_var

    vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4, device=DEVICE, patch_nums=VAR_PATCH_NUMS,
                             num_classes=1000, depth=VAR_DEPTHS[label], shared_aln=False)
    vae.load_state_dict(torch.load(WEIGHTS / "var/vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(WEIGHTS / f"var/var_d{VAR_DEPTHS[label]}.pth", map_location="cpu"), strict=True)
    return vae.eval().requires_grad_(False), var.eval().requires_grad_(False)


@torch.no_grad()
def sample_rar(generator, tok, label: str, classes: torch.Tensor):
    """Sample one batch from a RAR generator.

    generator: load_rar_generator(label). tok: load_rar_tokenizer(). classes: [B] ImageNet class ids.
    Returns tokens [B,256] (long) and images [B,3,256,256] (uint8), with images = D(Q^-1(tokens)).
    """
    scale, scale_pow, temp = RAR_SAMPLING[RAR_SIZES[label]]
    tokens = generator.generate(condition=classes.to(DEVICE), guidance_scale=scale, guidance_scale_pow=scale_pow,
                                randomize_temperature=temp).view(len(classes), -1)
    return tokens, to_uint8(tok.decode_tokens(tokens))


@contextmanager
def _capture_var_tokens():
    """Record the token indices VAR samples at each scale (autoregressive_infer_cfg only returns the image)."""
    import models.var as var_module

    orig, captured = var_module.sample_with_top_k_top_p_, []

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)  # [B, pn*pn, 1]
        captured.append(out[:, :, 0].clone())
        return out

    var_module.sample_with_top_k_top_p_ = wrapped
    try:
        yield captured
    finally:
        var_module.sample_with_top_k_top_p_ = orig


@torch.no_grad()
def sample_var(vae, var, classes: torch.Tensor, return_generator_image: bool = False):
    """Sample one batch from a VAR generator.

    The transformer runs under fp16 autocast, as in the repo's demo. The image is decoded again in fp32 from the
    captured tokens, so images = D(f_hat(tokens)) exactly.
    Returns tokens [B,680] (long, all 10 scales concatenated small -> large) and images [B,3,256,256] (uint8);
    with return_generator_image, also the generator's own [0,1] output, for checking the token capture.
    """
    from .signals import _var_idx_to_fhat

    with _capture_var_tokens() as idx, torch.autocast("cuda", dtype=torch.float16):
        gen_img = var.autoregressive_infer_cfg(B=len(classes), label_B=classes.to(DEVICE), **VAR_SAMPLING)
    assert [t.shape[1] for t in idx] == [pn * pn for pn in VAR_PATCH_NUMS]
    img = (vae.fhat_to_img(_var_idx_to_fhat(vae.quantize, idx).float()) + 1) / 2
    out = (torch.cat(idx, 1), to_uint8(img))
    return out + (gen_img.float(),) if return_generator_image else out


def split_var_tokens(tokens: torch.Tensor):
    """[B,680] concatenated VAR tokens -> list of [B, pn*pn] per scale (the format of _var_idx_to_fhat)."""
    return list(tokens.split([pn * pn for pn in VAR_PATCH_NUMS], dim=1))
