"""Fine-tune the inverse decoder D^-1 of one family (Stage 2), following Zhao et al., Eq. 6 and Table A1.

D^-1 starts as the family's original encoder and is trained to map generated images x_Z = D(f_Z) back to their
quantized features f_Z, with the decoder and codebook frozen. Data comes from scripts/generate_finetune_data.py.

    python scripts/finetune_inverse.py --family rar                       # paper recipe (50 epochs, lr 5e-4, batch 8)
    python scripts/finetune_inverse.py --family var                       # paper recipe (10 epochs, lr 5e-5, batch 16)
    python scripts/finetune_inverse.py --family rar --epochs 5 --out /workspace/cache/stage2/inv/rar_short

Writes config.json, log.csv (per-epoch metrics; epoch 0 = original encoder), last.pt (for resuming) and final.pt
(load with tracer.inverse.load_inverse, or pass to scripts/extract_features.py --inv-ckpt). Rerunning the same
command resumes after the last finished epoch.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer.common import CACHE  # noqa: E402
from tracer.inverse import finetune, load_generated, split_holdout, write_config  # noqa: E402

# Table A1 of the paper: epochs, batch size, Adam learning rate (both use StepLR with step 2, gamma 0.9)
PAPER = {"rar": dict(epochs=50, batch_size=8, lr=5e-4), "var": dict(epochs=10, batch_size=16, lr=5e-5)}


def main():
    """Parse arguments, load the generated data, hold out a fixed subset, and fine-tune."""
    ap = argparse.ArgumentParser(description="Fine-tune one family's inverse decoder with L_inv (Eq. 6).")
    ap.add_argument("--family", choices=["rar", "var"], required=True)
    ap.add_argument("--data", type=Path, default=CACHE / "stage2" / "gen", help="generate_finetune_data.py output")
    ap.add_argument("--max-per-model", type=int, default=None, help="use only the first N samples of each model")
    ap.add_argument("--holdout-per-model", type=int, default=100,
                    help="last N samples of each model are held out for evaluation, not trained on")
    ap.add_argument("--epochs", type=int, default=None, help="default: paper (rar 50, var 10)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: paper (rar 8, var 16)")
    ap.add_argument("--lr", type=float, default=None, help="Adam learning rate; default: paper (rar 5e-4, var 5e-5)")
    ap.add_argument("--step-size", type=int, default=2, help="StepLR: decay the lr every N epochs")
    ap.add_argument("--gamma", type=float, default=0.9, help="StepLR: decay factor")
    ap.add_argument("--seed", type=int, default=0, help="shuffling seed")
    ap.add_argument("--no-task-train-eval", action="store_true",
                    help="skip the per-epoch AUC on the task's train images (about 25 s/epoch for rar, 50 s for var)")
    ap.add_argument("--out", type=Path, default=None, help="default: <cache>/stage2/inv/<family>")
    args = ap.parse_args()
    for k, v in PAPER[args.family].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.out = args.out or CACHE / "stage2" / "inv" / args.family
    write_config(args.out, {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    torch.manual_seed(args.seed)
    data = load_generated(args.data, args.family, max_per_model=args.max_per_model)
    train, holdout = split_holdout(data, args.holdout_per_model)
    print(f"{args.family}: {len(train['tokens'])} train / {len(holdout['tokens'])} held-out generated images "
          f"from {sorted(set(data['label']))}", flush=True)
    finetune(args.family, train, holdout, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             step_size=args.step_size, gamma=args.gamma, seed=args.seed,
             eval_task_train=not args.no_task_train_eval, log=lambda s: print(s, flush=True))
    print(f"done: {args.out / 'final.pt'}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # fixed input size
    main()
