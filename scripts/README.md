# scripts

Long-running jobs. Run them in tmux with the project environment, and they write their outputs to `/workspace/cache` (outside the repo, never committed). Every script resumes after an interruption.

| script | stage | does |
|---|---|---|
| [extract_features.py](extract_features.py) | 1, 2 | provenance signals for every image, with the original encoder (Stage 1) or a fine-tuned $D^{-1}$ (`--inv-ckpt`, Stage 2) |
| [generate_finetune_data.py](generate_finetune_data.py) | 2 | sample (tokens, image) pairs from every generator of a family |
| [finetune_inverse.py](finetune_inverse.py) | 2 | fine-tune one family's inverse decoder $D^{-1}$ |
| [run_stage2.sh](run_stage2.sh) | 2 | the whole Stage 2 pipeline for both families |
| [extract_tokens.py](extract_tokens.py) | 3 | recover the tokens $Q(D^{-1}(x))$ of task and generated images with a family's fine-tuned $D^{-1}$ |
| [estimate_classes.py](estimate_classes.py) | 4 | top-10 ImageNet classes of task and generated images, from a pretrained classifier |
| [score_likelihood.py](score_likelihood.py) | 4 | per-token log-likelihoods of each image's tokens under each of its family's 4 transformers |

## extract_features.py

Computes the provenance signals (see [tracer/README.md](../tracer/README.md#signals)) for every image, using one family's tokenizer. All sizes in a family share the tokenizer, so this runs once per family, not once per model. Every image is scored, including the other family's images and outliers.

```bash
cd /workspace/iar-model-tracer
tmux new -s feats
# Stage 1: original encoders
/workspace/venv/bin/python scripts/extract_features.py --family rar
/workspace/venv/bin/python scripts/extract_features.py --family var --var-iters 0
# Stage 2: fine-tuned D^-1
/workspace/venv/bin/python scripts/extract_features.py --family rar --var-iters 0 \
    --inv-ckpt /workspace/cache/stage2/inv/rar/final.pt --out /workspace/cache/stage2/features
```

On the A40 at the default batch size, the full 11,050 images take about 7 minutes for RAR and 12.5 minutes for VAR.

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which tokenizer to use |
| `--splits` | `train val test` | splits to score |
| `--batch-size` | 64 | images per GPU batch; may change between resumed runs |
| `--shard-size` | 512 | images per shard file |
| `--var-iters` | 0 | Algorithm 3 token-search iterations for VAR; 0 = VAR's greedy quantization only |
| `--var-lr`, `--var-init-logit` | 0.1, 10.0 | Algorithm 3 Adam learning rate, and the initial logit on the greedy tokens |
| `--limit N` | none | debug: only the first N images |
| `--inv-ckpt` | none | fine-tuned $D^{-1}$ (`final.pt` from `finetune_inverse.py`) to use instead of the original encoder |
| `--out` | `/workspace/cache/stage1` | output directory |

### Outputs

```
<out>/<family>/config.json       arguments of the run (except --batch-size)
<out>/<family>/shard_XXXX.csv    one per --shard-size images
<out>/<family>.csv               all shards merged
```

Each CSV has one row per image, in `tracer.common.build_index()` order. The columns are `key, split, label, family, name`, followed by one `<family>_<signal>` column per signal:

- RAR: `rar_quant, rar_quant_p90, rar_enc1, rar_enc2, rar_cal, rar_comb, rar_tok_match`
- VAR: `var_quant_greedy, var_quant, var_enc1, var_enc2, var_cal, var_comb_greedy, var_comb, var_tok_match`

Join the two families on `key`, as the stage notebooks do.

### Resuming and recomputing

- Shards are written atomically (`.tmp`, then renamed), so the job can be killed at any time. Rerunning the same command skips finished shards and re-merges.
- If the arguments differ from `<out>/<family>/config.json` (apart from `--batch-size`), the script exits instead of mixing configurations. Delete `<out>/<family>/` to recompute.

### Quick check

```bash
/workspace/venv/bin/python scripts/extract_features.py --family rar --splits val --limit 8 --out /tmp/feats_test
```

## Stage 2: fine-tuning the inverse decoder

Stage 2 follows Zhao et al. (Eq. 6, App. C). The family's generators produce tokens $t_Z$ and images $x_Z = D(Q^{-1}(t_Z))$. $D^{-1}$, initialised from the original encoder, is trained so that $D^{-1}(x_Z) \approx f_Z = Q^{-1}(t_Z)$, with the decoder and codebook frozen. One $D^{-1}$ is trained per family, on an equal mix of all 4 sizes.

### run_stage2.sh

```bash
tmux new -s stage2 'bash scripts/run_stage2.sh'   # shortened recipe, about 2.5 h on the A40
```

This runs generation → fine-tuning → feature extraction for both families, with logs in `/workspace/cache/stage2/logs/`. The defaults are the shortened recipe: `PER_MODEL=2560`, `RAR_EPOCHS=10`, `VAR_EPOCHS=5`, with the paper's learning rate and batch size. To scale up, set the environment variables and a new output directory, e.g. `OUT=/workspace/cache/stage2_full PER_MODEL=12800 RAR_EPOCHS=50 VAR_EPOCHS=10` (the paper's 50k images per family; about 24 h).

### generate_finetune_data.py

```bash
/workspace/venv/bin/python scripts/generate_finetune_data.py --family rar --per-model 2560
```

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which generators to sample |
| `--per-model` | required | samples per model size, rounded up to whole shards |
| `--models` | all 4 | subset of model labels |
| `--batch-size` | 64 | generation batch; part of the seeding, so fixed per output directory |
| `--shard-size` | 512 | samples per shard file |
| `--seed` | 0 | base seed |
| `--out` | `/workspace/cache/stage2/gen` | output directory |

Output: `<out>/<family>/<label>/shard_XXXX.npz` with `tokens` [n,L] int16 (L = 256 for RAR, 680 for VAR), `images` [n,3,256,256] uint8 and `classes` [n], plus `<out>/<family>/config.json`. Classes cycle through the 1000 ImageNet classes in a fixed shuffled order. With the same seed and batch size, shards are bit-identical, so a larger `--per-model` only adds shards.

Speed at batch 64 (ms per image): RAR-B / L / XL / XXL 157 / 198 / 305 / 415; VAR-d16 / 20 / 24 / 30 48 / 58 / 78 / 108. VAR-d30 needs 38 GB of GPU memory.

### finetune_inverse.py

```bash
/workspace/venv/bin/python scripts/finetune_inverse.py --family rar --epochs 10   # paper lr / batch size
```

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which $D^{-1}$ to fine-tune |
| `--data` | `/workspace/cache/stage2/gen` | `generate_finetune_data.py` output |
| `--max-per-model` | all | use only the first N samples of each model |
| `--holdout-per-model` | 100 | last N samples of each model are held out (never trained on) and evaluated each epoch |
| `--epochs` | paper: rar 50, var 10 | |
| `--batch-size` | paper: rar 8, var 16 | |
| `--lr` | paper: rar 5e-4, var 5e-5 | Adam learning rate |
| `--step-size`, `--gamma` | 2, 0.9 | StepLR, stepped once per epoch |
| `--seed` | 0 | shuffling seed |
| `--no-task-train-eval` | off | skip the per-epoch AUC on the task's train images (about 25 s per epoch for RAR, 50 s for VAR) |
| `--out` | `/workspace/cache/stage2/inv/<family>` | output directory |

Outputs, in `--out`:

```
config.json   arguments; a rerun with different arguments exits instead of resuming
log.csv       one row per epoch (epoch 0 = original encoder): train L_inv, lr, held-out L_inv / token accuracy /
              QuantLoss, and AUC of each signal on the task's train images (own family vs other family)
last.pt       D^-1 + optimizer + scheduler after the latest epoch; a rerun resumes from here
final.pt      D^-1 weights after the last epoch -> extract_features.py --inv-ckpt
```

Speed: about 19 ms per image per epoch for RAR (batch 8) and 32 ms for VAR (batch 16); peak memory 18 GB and 25 GB. A resumed run gives the same weights as an uninterrupted one (checked with deterministic cuDNN).

## Stage 3: token habits

### extract_tokens.py

Recovers each image's tokens $t = Q(D^{-1}(x))$ with one family's fine-tuned Stage 2 $D^{-1}$ (`tracer.tokens.recover_tokens`), for the per-image token-habit features in [tracer/tokens.py](../tracer/tokens.py). Like `extract_features.py`, it runs once per family and every image is scored with that family's tokenizer, including the other family's images and outliers.

```bash
cd /workspace/iar-model-tracer
tmux new -s tokens
/workspace/venv/bin/python scripts/extract_tokens.py --family rar
/workspace/venv/bin/python scripts/extract_tokens.py --family var
```

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which tokenizer and $D^{-1}$ to use |
| `--sources` | `task gen` | `task`: every `build_index()` image (train, val, test), in index order; `gen`: the generated set at `--gen-root` |
| `--inv-ckpt` | `/workspace/cache/stage2/inv/<family>/final.pt` | fine-tuned $D^{-1}$ (from `finetune_inverse.py`) |
| `--gen-root` | `/workspace/cache/stage3/gen` | generated set, read with `tracer.inverse.load_generated`. The fresh seed-1 set, not `stage2/gen`, most of which $D^{-1}$ was trained on |
| `--batch-size` | 64 | images per GPU batch; may change between resumed runs |
| `--shard-size` | 512 | images per shard file |
| `--limit N` | none | debug: only the first N task images, and the first N generated images **per model** |
| `--out` | `/workspace/cache/stage3/tokens` | output directory |

#### Outputs

```
<out>/<family>/<source>/config.json      arguments of the run (except --batch-size)
<out>/<family>/<source>/shard_XXXX.npz   one per --shard-size images
<out>/<family>/<source>.npz              all shards merged
```

Token arrays are int16 `[N,L]`, with L = 256 for RAR (16x16 grid, raster order) and 680 for VAR (10 scales concatenated small → large), as in `tracer/generate.py`.

- `task.npz`: `key, split, label, family, name` (strings; `label` and `family` are `""` for test images) and `tokens` (recovered), in `build_index()` order.
- `gen.npz`: `label, class, index` (position in that model's sample sequence), `true_tokens` (sampled by the generator) and `tokens` (recovered). The `gen` run also prints the recovered-vs-true token accuracy per model.

Resuming works as in `extract_features.py`: shards are written atomically, finished shards are skipped, and the script exits if the arguments differ from `config.json`.

#### Quick check

```bash
/workspace/venv/bin/python scripts/extract_tokens.py --family rar --limit 64 --out /tmp/tokens_dryrun
```

## Stage 4: generator likelihood

### estimate_classes.py

The generators are class-conditional, so scoring an image's tokens under a generator (`tracer/likelihood.py`) needs its ImageNet class. Generated images come with their true class; task images do not. This script saves the top-10 classes and their softmax probabilities from a pretrained ImageNet classifier: torchvision `convnext_large` with `IMAGENET1K_V1` weights (84.4% top-1 on ImageNet), using the weights' own preprocessing (resize 232, centre crop 224, ImageNet normalisation) on the $[0,1]$ images. The weights (755 MB) are downloaded once to `/workspace/weights/torchvision/`.

```bash
cd /workspace/iar-model-tracer
tmux new -s classes
/workspace/venv/bin/python scripts/estimate_classes.py
```

| argument | default | meaning |
|---|---|---|
| `--sources` | `task gen` | `task`: every `build_index()` image (train, val, test), in index order; `gen`: the generated set at `--gen-root`, both families |
| `--classifier` | `convnext_large.IMAGENET1K_V1` | torchvision `<architecture>.<weights>` |
| `--gen-root` | `/workspace/cache/stage3/gen` | generated set, read with `tracer.inverse.load_generated` |
| `--batch-size` | 64 | images per GPU batch; may change between resumed runs |
| `--shard-size` | 512 | images per shard file |
| `--limit N` | none | debug: only the first N task images, and the first N generated images **per model** |
| `--out` | `/workspace/cache/stage4/classes` | output directory |

#### Outputs

```
<out>/<source>/config.json      arguments of the run (except --batch-size)
<out>/<source>/shard_XXXX.npz   one per --shard-size images
<out>/<source>.npz              all shards merged
```

Both archives have `top10_class` (int16 `[N,10]`) and `top10_prob` (float32 `[N,10]`, softmax probabilities, most likely first), plus:

- `task.npz`: `key, split, label, family, name` (strings; `label` and `family` are `""` for test images), in `build_index()` order.
- `gen.npz`: `family, label, index` (position in that model's sample sequence) and `class` (the true class). RAR rows come first, then VAR. The `gen` run also prints top-1 / 5 / 10 accuracy against the true classes, per model.

Resuming works as in `extract_features.py`.

#### Quick check

```bash
/workspace/venv/bin/python scripts/estimate_classes.py --sources gen --limit 64 --out /tmp/classes_dryrun
```

On this check (64 images per model, 512 in total), top-1 / 5 / 10 accuracy against the true generated classes is 92.6% / 99.2% / 99.6%.

### score_likelihood.py

Scores every image's tokens under each of the family's 4 transformers with `tracer.likelihood.score_tokens` (teacher forcing, with the guided sampling distribution of each repo's sampler; see [tracer/likelihood.py](../tracer/likelihood.py)). One model is on the GPU at a time and `free()` is called between models. Scoring is fp32, matching `tracer/test_likelihood.py`.

```bash
cd /workspace/iar-model-tracer
tmux new -s scores
# oracle: true tokens, true class (generated set)
/workspace/venv/bin/python scripts/score_likelihood.py --family rar --source gen --tokens true --class-mode true
# realistic: recovered tokens, top-5 estimated classes (needs estimate_classes.py first)
/workspace/venv/bin/python scripts/score_likelihood.py --family rar --source gen --tokens recovered --class-mode topk
/workspace/venv/bin/python scripts/score_likelihood.py --family rar --source task --tokens recovered --class-mode topk
```

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which family's 4 models score the images |
| `--source {gen,task}` | required | `gen`: `stage3/tokens/<family>/gen.npz`; `task`: `stage3/tokens/<family>/task.npz`, images of `--splits` (all families and outliers) |
| `--tokens {true,recovered}` | required | the generator's own tokens (`gen` only), or $Q(D^{-1}(x))$ |
| `--class-mode {true,topk}` | required | the true class (`gen` only), or each of the top `--topk` classes from `estimate_classes.py`, scored separately |
| `--topk` | 5 | classes per image for `topk` (at most 10) |
| `--splits` | `train val` | task splits to score (`task` only) |
| `--models` | all 4 | subset of the family's models |
| `--classes` | `/workspace/cache/stage4/classes/<source>.npz` | class file for `topk` |
| `--batch-size` | 32 | (image, class) pairs per GPU pass, two sequences each (class and null); may change between resumed runs |
| `--shard-size` | 512 | images per shard file |
| `--limit N` | none | debug: `gen`, the first N images per generating model; `task`, the first N images of the selected splits |
| `--out` | `/workspace/cache/stage4/scores` | output directory |

#### Outputs

```
<out>/<family>/<source>_<tokens>_<class-mode>/<model>/config.json      arguments of the run (except --batch-size, --models)
<out>/<family>/<source>_<tokens>_<class-mode>/<model>/shard_XXXX.npz   one per --shard-size images
<out>/<family>/<source>_<tokens>_<class-mode>/<model>.npz              all shards merged
```

Each archive has one row per image, in the row order of the stage3 token file, with K classes per image (K = 1 for `--class-mode true`):

| keys | shape, dtype | contents |
|---|---|---|
| `row` + `label, index, class` (gen) or `key, split, label, family, name` (task) | `[N]` | identity; `row` is the position in the stage3 token file |
| `classes`, `class_prob` | `[N,K]` int16, float32 | the class scored in each slot and its classifier probability (1 for `true`) |
| `logp_cond`, `logp_uncond`, `logp_guided` | `[N,K,L]` float16 | per-token log-probabilities (`tracer.likelihood.score_tokens`) |
| `rank_guided` | `[N,K,L]` int16 | rank of the token under the guided logits (0 = most likely) |
| `in_support` | `[N,K,L]` int8 | VAR only: 1 if the token survives top-k = 900 / top-p = 0.96 |
| `sum_<q>`, `mean_<q>` | `[N,K]` float32 | per-image sum and mean of each per-token quantity, computed in float32 |
| `min20_logp_guided` | `[N,K]` float32 | mean of the lowest 20% of the image's `logp_guided` (Min-K%) |
| `mean_cond_minus_uncond` | `[N,K]` float32 | mean of `logp_cond - logp_uncond` |
| `in_support_rate` | `[N,K]` float32 | VAR only: fraction of tokens in the top-k / top-p support |

Resuming works as in `extract_features.py`: finished shards are skipped, a model is loaded only if it has shards left, and the script exits if the arguments differ from `config.json`. Size: for `gen` with `--topk 5`, about 0.1 GB per RAR model and 0.3 GB per VAR model. The shards and the merged file each hold a full copy.

#### Speed

Measured on the A40, fp32, `--batch-size 32`: about 100 ms per (image, class) for RAR-XXL and 260 ms for VAR-d30. Time grows roughly with model size, so all 4 models of a family together take about 0.22 s (RAR) and 0.5 s (VAR) per (image, class). Estimates for full runs:

| run | images x classes | RAR | VAR |
|---|---|---|---|
| `gen`, K = 1 (`true` tokens or `recovered`, true class) | 10,240 x 1 | ~40 min | ~1.5 h |
| `gen --class-mode topk` (K = 5) | 10,240 x 5 | ~3 h | ~7 h |
| `task` train + val, K = 5 | 1,250 x 5 | ~25 min | ~55 min |
| `task` test, K = 5 | 9,800 x 5 | ~3 h | ~7 h |

#### Quick check

```bash
/workspace/venv/bin/python scripts/estimate_classes.py --limit 64 --out /tmp/classes_dry
/workspace/venv/bin/python scripts/score_likelihood.py --family rar --models rarxxl --source gen --tokens recovered \
    --class-mode topk --limit 64 --classes /tmp/classes_dry/gen.npz --out /tmp/scores_dry
```
