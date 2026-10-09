"""Token recovery and per-image "token habit" features (Stage 3).

Within a family all variants share the autoencoder, so they can only differ in which tokens their transformer
writes. Tokens are recovered from an image as t = Q(D^-1(x)), with the Stage 2 D^-1 loaded into the tokenizer
(tracer.inverse.load_inverse), and summarised per image:
    habit_features       a few interpretable statistics (entropy, distinct tokens, repeats, rare-token usage)
    histogram_features   the normalised token histogram, as a sparse matrix

Token layout (as in tracer.generate):
    rar: [B,256]  16x16 grid in raster order, codebook of 1024
    var: [B,680]  10 scales (1x1 ... 16x16) concatenated small -> large, each in raster order, codebook of 4096
Feature functions take token arrays as numpy or torch (any device) and work on the CPU.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .generate import VAR_PATCH_NUMS, split_var_tokens
from .inverse import encode, quantize

CODEBOOK_SIZE = {"rar": 1024, "var": 4096}
SEQ_LEN = {"rar": 256, "var": sum(pn * pn for pn in VAR_PATCH_NUMS)}
RAR_GRID = 16
MIN_ADJ_SCALE = 4  # VAR: adjacency features only on scales >= 4x4


# ---------------------------------------------------------------- token recovery

@torch.no_grad()
def recover_tokens(tok, family: str, x01: torch.Tensor) -> torch.Tensor:
    """Q(D^-1(x)): [B,3,256,256] in [0,1] on DEVICE -> LongTensor [B,L] on DEVICE (L = 256 RAR, 680 VAR).

    tok should hold the fine-tuned D^-1 (inverse.load_inverse); with the original encoder this is Q(E(x)).
    VAR uses its greedy scale-wise quantization (Alg. 2), with the scales concatenated as in generate.sample_var.
    """
    idx, _ = quantize(tok, family, encode(tok, family, x01))
    return idx.reshape(len(x01), -1).long()


# ---------------------------------------------------------------- helpers

def _as_numpy(tokens, family: str) -> np.ndarray:
    """Tokens [B,L] (numpy or torch) -> int64 numpy, with the shape and codebook range checked."""
    t = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    t = t.astype(np.int64, copy=False)
    assert t.ndim == 2 and t.shape[1] == SEQ_LEN[family], f"{family} tokens must be [B,{SEQ_LEN[family]}], got {t.shape}"
    assert t.size == 0 or (t.min() >= 0 and t.max() < CODEBOOK_SIZE[family]), "token index outside the codebook"
    return t


def _entropy_distinct(t: np.ndarray):
    """Per row of t [B,n]: entropy (bits) of the token histogram, and the fraction of distinct tokens."""
    B, n = t.shape
    s = np.sort(t, axis=1)
    new_run = np.ones_like(s, dtype=bool)
    new_run[:, 1:] = s[:, 1:] != s[:, :-1]  # column 0 starts a run, so runs never cross rows
    starts = np.flatnonzero(new_run.ravel())
    p = np.diff(np.append(starts, B * n)) / n  # run length / n = probability of each distinct token
    row = starts // n
    entropy = -np.bincount(row, weights=p * np.log2(p), minlength=B)
    distinct = np.bincount(row, minlength=B) / n
    return entropy, distinct


def _same_adjacent(grid: np.ndarray):
    """grid [B,s,s] -> fraction of horizontally / vertically adjacent position pairs with the same token, each [B]."""
    return ((grid[:, :, 1:] == grid[:, :, :-1]).mean(axis=(1, 2)),
            (grid[:, 1:, :] == grid[:, :-1, :]).mean(axis=(1, 2)))


# ---------------------------------------------------------------- features

def token_frequency(tokens, family: str, alpha: float = 1.0) -> np.ndarray:
    """Codebook-usage frequency [K] from reference tokens [N,L], with add-alpha smoothing so that log() is finite.

    Fit it on training tokens only, and pass it to habit_features for every split, so no test information leaks.
    VAR counts all scales together.
    """
    t = _as_numpy(tokens, family)
    counts = np.bincount(t.ravel(), minlength=CODEBOOK_SIZE[family]).astype(np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


def habit_features(tokens, family: str, freq: np.ndarray | None = None) -> pd.DataFrame:
    """Interpretable per-image token statistics: tokens [B,L] -> DataFrame [B, n_features].

    Columns are prefixed with the family (as in scripts/extract_features.py), e.g. rar_entropy, var_same_h_s8.
        entropy      entropy (bits) of the image's token histogram
        distinct     number of distinct tokens / number of tokens
        same_h/v     fraction of horizontally / vertically adjacent grid positions holding the same token
        logfreq_mean, logfreq_p10
                     mean and 10th percentile, over the image's tokens, of log freq[token]; only if freq is given
                     (token_frequency, fit on training tokens only)
    RAR: entropy, distinct, same_h, same_v on the 16x16 grid.
    VAR: entropy and distinct on the whole sequence (no suffix) and per scale (suffix _s<pn>; the 1x1 scale is
    skipped, as both are constant there), and same_h / same_v on each scale >= 4x4.
    """
    t = _as_numpy(tokens, family)
    p = family
    cols = {}
    if family == "rar":
        cols[f"{p}_entropy"], cols[f"{p}_distinct"] = _entropy_distinct(t)
        cols[f"{p}_same_h"], cols[f"{p}_same_v"] = _same_adjacent(t.reshape(-1, RAR_GRID, RAR_GRID))
    else:
        cols[f"{p}_entropy"], cols[f"{p}_distinct"] = _entropy_distinct(t)
        for pn, ts in zip(VAR_PATCH_NUMS, split_var_tokens(torch.from_numpy(t))):
            ts = ts.numpy()
            if pn > 1:
                cols[f"{p}_entropy_s{pn}"], cols[f"{p}_distinct_s{pn}"] = _entropy_distinct(ts)
            if pn >= MIN_ADJ_SCALE:
                cols[f"{p}_same_h_s{pn}"], cols[f"{p}_same_v_s{pn}"] = _same_adjacent(ts.reshape(-1, pn, pn))
    if freq is not None:
        freq = np.asarray(freq, dtype=np.float64)
        assert freq.shape == (CODEBOOK_SIZE[family],) and (freq > 0).all(), "freq must be positive, one per code"
        logf = np.log(freq)[t]
        cols[f"{p}_logfreq_mean"] = logf.mean(axis=1)
        cols[f"{p}_logfreq_p10"] = np.percentile(logf, 10, axis=1)
    return pd.DataFrame(cols)


def histogram_features(tokens, family: str, codebook_size: int | None = None) -> sp.csr_matrix:
    """Normalised token counts: tokens [B,L] -> CSR [B, codebook_size] float32, each row summing to 1.

    codebook_size defaults to the family's (RAR 1024, VAR 4096); VAR counts are summed over all scales.
    """
    t = _as_numpy(tokens, family)
    K = CODEBOOK_SIZE[family] if codebook_size is None else codebook_size
    assert t.size == 0 or t.max() < K, "token index >= codebook_size"
    B, L = t.shape
    data = np.full(B * L, 1.0 / L, dtype=np.float32)
    m = sp.csr_matrix((data, (np.repeat(np.arange(B), L), t.ravel())), shape=(B, K))
    m.sum_duplicates()
    return m


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B = 64
    for family in ("rar", "var"):
        K, L = CODEBOOK_SIZE[family], SEQ_LEN[family]
        t = rng.integers(0, K, size=(B, L))
        t[0] = 7  # constant image: entropy 0, one distinct token, all neighbours equal
        freq = token_frequency(t[B // 2:], family)  # "train" half only
        df = habit_features(torch.from_numpy(t[:B // 2]), family, freq)  # torch input, "test" half
        h = histogram_features(t, family)

        assert len(df) == B // 2 and np.isfinite(df.values).all()
        ent = df.filter(regex=r"_entropy")
        assert (ent.values >= 0).all() and (df[f"{family}_entropy"] <= np.log2(L) + 1e-9).all()
        frac = df.filter(regex=r"_(distinct|same_h|same_v)")
        assert ((frac.values >= 0) & (frac.values <= 1)).all()
        assert (df.filter(regex=r"_logfreq").values < 0).all()
        assert (df[f"{family}_logfreq_p10"] <= df[f"{family}_logfreq_mean"] + 1e-9).all()
        row0 = df.iloc[0]
        assert (row0.filter(regex=r"_entropy") == 0).all()
        assert (row0.filter(regex=r"_same_") == 1).all()
        assert np.isclose(row0[f"{family}_distinct"], 1 / L)
        assert h.shape == (B, K) and isinstance(h, sp.csr_matrix)
        assert np.allclose(np.asarray(h.sum(axis=1)).ravel(), 1, atol=1e-5) and h.min() >= 0
        assert np.isclose(h[0, 7], 1)
        nonconst = df.iloc[1:].drop(columns=df.filter(regex=r"_logfreq").columns)  # random rows only
        assert np.allclose(nonconst[f"{family}_distinct"], np.count_nonzero(h[1:B // 2].toarray(), axis=1) / L)

        print(f"{family}: tokens {t.shape}, habit_features {df.shape}, histogram_features {h.shape} "
              f"(nnz {h.nnz}), frequency table {freq.shape}")
        print(df.iloc[1:].describe().T[["mean", "min", "max"]].to_string(float_format="%.4f"), "\n")
    print("self-test passed")
