"""Compare each Stage 2 inverse decoder with its family's original encoder.

Task train/val images and held-out generated images are processed with both families.
The generated images used to train D_inv are excluded by split_holdout().
"""
import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import (  # noqa: E402
    DEVICE,
    build_index,
    free,
    load_batch,
    load_rar_tokenizer,
    load_var_vae,
)
from tracer.inverse import (  # noqa: E402
    _batches,
    encode,
    load_generated,
    load_inverse,
    split_holdout,
    target,
)

FAMILIES = ("rar", "var")
INV_ROOT = Path("/workspace/cache/stage2/inv")
GEN_ROOT = Path("/workspace/cache/stage2/gen")
OUT_ROOT = Path("/workspace/features")


def _subset(data: dict, indices) -> dict:
    return {key: value[indices] for key, value in data.items()}


def _heldout_groups(limit: int | None) -> list[dict]:
    """Load only held-out generated samples, optionally selecting across all eight model labels."""
    groups = []
    for family in FAMILIES:
        all_samples = load_generated(GEN_ROOT, family)
        _, heldout = split_holdout(all_samples, n_per_model=100)
        groups.append(heldout)
        del all_samples

    if limit is None:
        return groups

    # Select round-robin across labels so a small diagnostic run covers both families and sizes.
    labels = sorted({label for group in groups for label in np.unique(group["label"])})
    candidates = []
    for label in labels:
        for group_index, group in enumerate(groups):
            matches = np.flatnonzero(group["label"] == label)
            if len(matches):
                candidates.append((group_index, matches))
    indices = [[] for _ in groups]
    cursors = [0] * len(candidates)
    while sum(map(len, indices)) < limit:
        made_progress = False
        for candidate_index, (group_index, matches) in enumerate(candidates):
            if cursors[candidate_index] < len(matches) and sum(map(len, indices)) < limit:
                indices[group_index].append(int(matches[cursors[candidate_index]]))
                cursors[candidate_index] += 1
                made_progress = True
            if sum(map(len, indices)) >= limit:
                break
        if not made_progress:
            break
    return [_subset(group, np.asarray(index, dtype=np.int64)) for group, index in zip(groups, indices)]


def _jpeg_size(image: np.ndarray) -> int:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.tell()


def _metadata(split: str, groups: list[dict], limit: int | None) -> dict:
    if split in ("train", "val"):
        frame = build_index()
        frame = frame[frame.split == split].reset_index(drop=True)
        if limit is not None:
            frame = frame.head(limit)
        names = frame["name"].to_numpy(dtype=str)
        labels = frame["label"].fillna("").to_numpy(dtype=str)
        families = frame["family"].fillna("").to_numpy(dtype=str)
        png_sizes = np.asarray([Path(path).stat().st_size for path in frame["path"]], dtype=np.int64)
        jpeg_sizes = []
        for path in frame["path"]:
            with Image.open(path) as image:
                jpeg_sizes.append(_jpeg_size(np.asarray(image.convert("RGB"))))
        return {
            "name": names,
            "label": labels,
            "family": families,
            "png_size": png_sizes,
            "jpeg_q90_size": np.asarray(jpeg_sizes, dtype=np.int64),
            "task_paths": frame["path"].tolist(),
        }

    names, labels, families, png_sizes, jpeg_sizes = [], [], [], [], []
    for source_family, group in zip(FAMILIES, groups):
        for label, index, image in zip(group["label"], group["index"], group["images"]):
            names.append(f"{label}/{int(index)}.png")
            labels.append(str(label))
            families.append(source_family)
            png_buffer = io.BytesIO()
            Image.fromarray(image.transpose(1, 2, 0)).save(png_buffer, format="PNG")
            png_sizes.append(png_buffer.tell())
            jpeg_sizes.append(_jpeg_size(image.transpose(1, 2, 0)))
    return {
        "name": np.asarray(names, dtype=str),
        "label": np.asarray(labels, dtype=str),
        "family": np.asarray(families, dtype=str),
        "png_size": np.asarray(png_sizes, dtype=np.int64),
        "jpeg_q90_size": np.asarray(jpeg_sizes, dtype=np.int64),
    }


def _iter_batches(split: str, groups: list[dict], metadata: dict, batch_size: int):
    offset = 0
    if split in ("train", "val"):
        paths = metadata["task_paths"]
        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            yield start, end, load_batch(paths[start:end]), None
        return

    for group in groups:
        for x, tokens in _batches(group, batch_size):
            end = offset + len(x)
            yield offset, end, x, tokens
            offset = end


def _extract_family(split: str, groups: list[dict], metadata: dict, results: dict, family: str,
                    batch_size: int, tmp_dir: Path) -> None:
    tokenizer = load_rar_tokenizer() if family == "rar" else load_var_vae()
    count = len(metadata["name"])
    feature_file = tmp_dir / f"{split}_{family}_encoder_features.bin"
    features = None

    for start, end, images, _ in _iter_batches(split, groups, metadata, batch_size):
        with torch.no_grad():
            encoded = encode(tokenizer, family, images)
        if features is None:
            features = np.memmap(feature_file, mode="w+", dtype=np.float32,
                                 shape=(count, *encoded.shape[1:]))
        features[start:end] = encoded.cpu().numpy()
    if features is None:
        del tokenizer
        free()
        return
    features.flush()

    checkpoint = INV_ROOT / family / "final.pt"
    load_inverse(tokenizer, family, checkpoint)
    print(f"{split}: loaded {family} D_inv from {checkpoint}", flush=True)

    for start, end, images, tokens in _iter_batches(split, groups, metadata, batch_size):
        with torch.no_grad():
            f_d = encode(tokenizer, family, images)
            f_e = torch.from_numpy(np.asarray(features[start:end])).to(DEVICE)
            per_position_l2 = (f_e - f_d).square().sum(dim=1)
            cosine_distance = 1 - F.cosine_similarity(f_e, f_d, dim=1)

            results[f"{family}_s_l2"][start:end] = per_position_l2.mean(dim=(1, 2)).cpu().numpy()
            results[f"{family}_s_cos"][start:end] = cosine_distance.mean(dim=(1, 2)).cpu().numpy()
            results[f"{family}_disagreement_map"][start:end] = per_position_l2.cpu().numpy()

            if family == "rar":
                _, idx_e = tokenizer.quantize(f_e)[:2]
                _, idx_d = tokenizer.quantize(f_d)[:2]
                spatial_shape = per_position_l2.shape[-2:]
                idx_e = idx_e.reshape(len(images), *spatial_shape)
                idx_d = idx_d.reshape(len(images), *spatial_shape)
                token_map = (idx_e != idx_d)
                results["rar_tok_disagree_map"][start:end] = token_map.cpu().numpy().astype(np.uint8)
                results["rar_tok_disagree"][start:end] = token_map.float().mean(dim=(1, 2)).cpu().numpy()

            if tokens is not None:
                token_offset = 0
                for group in groups:
                    group_end = token_offset + len(group["tokens"])
                    overlap_start, overlap_end = max(start, token_offset), min(end, group_end)
                    if overlap_start < overlap_end and group["label"][0].startswith(family):
                        batch_start = overlap_start - start
                        batch_end = overlap_end - start
                        z_q = target(tokenizer, family, tokens[batch_start:batch_end])
                        original_f = f_e[batch_start:batch_end]
                        inverse_f = f_d[batch_start:batch_end]
                        results[f"{family}_true_err"][overlap_start:overlap_end] = (
                            (original_f - z_q).square().sum(dim=1).mean(dim=(1, 2)).cpu().numpy()
                        )
                        results[f"{family}_true_err_Dinv"][overlap_start:overlap_end] = (
                            (inverse_f - z_q).square().sum(dim=1).mean(dim=(1, 2)).cpu().numpy()
                        )
                    token_offset = group_end

    del features, tokenizer
    free()


def _empty_results(count: int) -> dict:
    results = {}
    for family in FAMILIES:
        results[f"{family}_s_l2"] = np.full(count, np.nan, dtype=np.float32)
        results[f"{family}_s_cos"] = np.full(count, np.nan, dtype=np.float32)
        results[f"{family}_disagreement_map"] = np.full((count, 16, 16), np.nan, dtype=np.float32)
        results[f"{family}_true_err"] = np.full(count, np.nan, dtype=np.float32)
        results[f"{family}_true_err_Dinv"] = np.full(count, np.nan, dtype=np.float32)
    results["rar_tok_disagree"] = np.full(count, np.nan, dtype=np.float32)
    results["rar_tok_disagree_map"] = np.zeros((count, 16, 16), dtype=np.uint8)
    return results


def _run_split(split: str, limit: int | None, batch_size: int, out_dir: Path) -> None:
    groups = _heldout_groups(limit) if split == "generated_heldout" else []
    metadata = _metadata(split, groups, limit)
    count = len(metadata["name"])
    results = _empty_results(count)
    results.update({
        "image_name": metadata["name"],
        "true_label": metadata["label"],
        "family": metadata["family"],
        "png_size": metadata["png_size"],
        "jpeg_q90_size": metadata["jpeg_q90_size"],
    })
    with tempfile.TemporaryDirectory(prefix=f"disagreement_{split}_") as scratch:
        for family in FAMILIES:
            _extract_family(split, groups, metadata, results, family, batch_size, Path(scratch))
    output = out_dir / f"disagreement_{split}.npz"
    np.savez_compressed(output, **results)
    print(f"wrote {output} ({count} images)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "generated_heldout"],
                        default=["train", "val", "generated_heldout"])
    parser.add_argument("--limit", type=int, default=None, help="maximum images per split (for dry runs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _run_split(split, args.limit, args.batch_size, args.out)


if __name__ == "__main__":
    main()
