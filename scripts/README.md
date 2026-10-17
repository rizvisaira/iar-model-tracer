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
| [score_stage5_pilot.py](score_stage5_pilot.py) | 5 | signals, token recovery and $D^{-1}$ inversion error of the VAR generated holdout under several inverse decoders |
| [run_stage5_pilot.sh](run_stage5_pilot.sh) | 5 | the Stage 5 pilot: var24-only $D^{-1}$s (5 and 20 epochs on stage2 data; 5 epochs on 9,840 images with new data), then `score_stage5_pilot.py` |
| [assemble_stage5_gen.py](assemble_stage5_gen.py) | 5 | symlink a model's stage2 shards and its newly generated shards into one generated set (var24 by default; all models of a family with `--labels`), with seed and source checks |
| [run_stage5_full.sh](run_stage5_full.sh) | 5 | Stage 5 full: 7,680 new images, a $D^{-1}$ and task-image features for each of the 8 models (var24's $D^{-1}$ from the pilot) |

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

Output: `<out>/<family>/<label>/shard_XXXX.npz` with `tokens` [n,L] int16 (L = 256 for RAR, 680 for VAR), `images` [n,3,256,256] uint8 and `classes` [n], plus `<out>/<family>/config.json`. Classes cycle through the 1000 ImageNet classes in a fixed shuffled order. With the same seed and batch size, shards are bit-identical on the same GPU and software, so a larger `--per-model` only adds shards. On another GPU the class order is the same but the samples differ (regenerating on the H100 matches only 50% of the A40's 1x1-scale tokens), so extend an existing directory only on the hardware that wrote it.

Speed at batch 64 (ms per image): RAR-B / L / XL / XXL 157 / 198 / 305 / 415; VAR-d16 / 20 / 24 / 30 48 / 58 / 78 / 108. VAR-d30 needs 38 GB of GPU memory.

### finetune_inverse.py

```bash
/workspace/venv/bin/python scripts/finetune_inverse.py --family rar --epochs 10   # paper lr / batch size
```

| argument | default | meaning |
|---|---|---|
| `--family {rar,var}` | required | which $D^{-1}$ to fine-tune |
| `--data` | `/workspace/cache/stage2/gen` | `generate_finetune_data.py` output |
| `--models` | all | train and evaluate only on these model folders of `<data>/<family>`, e.g. `var24`; recorded in `config.json` only when given |
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

## Stage 5 pilot: one inverse decoder per model

### run_stage5_pilot.sh

```bash
SMOKE=1 bash scripts/run_stage5_pilot.sh                    # 1 epoch on 32 images, 1 new shard, --limit 16; about 4 min
tmux new -s stage5 'bash scripts/run_stage5_pilot.sh'
```

Steps, each skipped if its output exists (interrupted steps resume); outputs and logs (`logs/`) go to `/workspace/cache/stage5`:

1. `inv/var24_5ep`, `inv/var24_20ep`: `finetune_inverse.py --models var24` on `stage2/gen` (2,460 training images) for 5 epochs (770 optimizer steps) and 20 epochs (3,080 steps; the shared inverse took 3,075).
2. `gen_extra/`: 7,680 new var24 images (`generate_finetune_data.py --models var24 --seed 10 --per-model 7680`, 15 shards), after `assemble_stage5_gen.py --seed-check-only` has confirmed that none of seed 10's derived RNG seeds was used for `stage2/gen` (seed 0) or `stage3/gen` (seed 1). Skipped if the last shard exists.
3. `gen/`: `assemble_stage5_gen.py` symlinks the 5 var24 shards of `stage2/gen` (`shard_0000-0004`) and the 15 new ones (`shard_0005-0019`), so `load_generated(stage5/gen, "var")` returns 10,240 var24 rows with index 0..10239; `gen/var/config.json` records the composition. Skipped if that file exists.
4. `inv/var24_full`: `finetune_inverse.py --data stage5/gen --models var24 --max-per-model 9940`, 5 epochs: 9,840 training images (index 0..9,839) and 100 held out, 3,075 steps, as for the shared inverse.
5. `scores.csv`: `E`, `shared`, `single5`, `single20` and `singlefull`, with the holdout checked against `stage2/gen` and `stage5/gen` (`--overlap-data`). Inverses already scored are skipped by `score_stage5_pilot.py`.

Every fine-tuning setting other than `--data`, `--models`, `--max-per-model`, `--epochs` and `--out` is as in `stage2/inv/var/config.json`. Nothing is written to `stage2/gen` or `stage3/gen`. `SMOKE=1` writes to `/workspace/cache/stage5_smoke` instead: 40 images per model with the last 8 held out, 1 epoch for every run, 1 new shard (512 images), scoring `--limit 16`.

Images generated on the H100 are decoded slightly differently from the A40 ones in `stage2/gen` and `stage3/gen`: re-decoding stored tokens on the H100 changes 1.3% of uint8 pixel values, almost all by one level, less than changing the batch size on the H100 alone (1.9%).

### assemble_stage5_gen.py

```bash
/workspace/venv/bin/python scripts/assemble_stage5_gen.py --seed-check-only --extra-seed 10 --expect-extra 15
/workspace/venv/bin/python scripts/assemble_stage5_gen.py --expect-extra 15
```

| argument | default | meaning |
|---|---|---|
| `--family` | `var` | family of the models to assemble |
| `--label` | `var24` | model folder to assemble |
| `--labels` | `--label` | several model folders of `--family`, assembled in one call |
| `--base` | `/workspace/cache/stage2/gen` | first source; its shards come first |
| `--extra` | `/workspace/cache/stage5/gen_extra` | second source (`generate_finetune_data.py --out`) |
| `--expect-extra` | required | number of shards the second source must have |
| `--extra-for LABEL DIR N_SHARDS` | none | use `DIR` (with `N_SHARDS` shards) as `LABEL`'s second source instead of `--extra` (repeatable) |
| `--used` | `stage2/gen stage3/gen` | generated sets whose derived seeds (class shuffle `seed + model position`, per-shard `seed * 1000003 + position * 10007 + shard`) must not be reused |
| `--seed-scope` | `all` | `all`: compare with every family of `--used`; `family`: only with `--family` (seeds drive different networks in different families, and `stage2/gen` already uses seed 0 for both) |
| `--seed-check-only`, `--extra-seed` | off | only check a planned seed for every label against `--used`, then exit |
| `--out` | `/workspace/cache/stage5/gen` | output; only symlinks and `<family>/config.json` are written |

Checks before linking: same family, shard size and batch size; shard files without gaps and with `--shard-size` samples each; each source's classes equal the generator's class cycling for its seed; no derived seed collision with `--used` (a `--used` set that is the source itself is skipped, so the pilot's var24 shards can be reused) or between the second sources of one call. `<family>/config.json` holds the sources (directory, seed, batch size, shards, rows, GPU, output shard and index ranges), the seed check, the time, git commit and command; with several labels, these are per model under `models`. It is written after all model folders are in place. Rerunning with the same composition does nothing; a different one exits.

### score_stage5_pilot.py

Scores the VAR holdout of the fresh generated set (holdout rows of `stage3/gen_split.csv`, 512 per depth, images from `stage3/gen`, true tokens from `stage3/tokens/var/gen.npz` matched by label and index) under several inverse decoders: `E` (the original encoder), `shared` (`stage2/inv/var/final.pt`, trained on all four depths) and each `--single NAME=PATH` (for example a $D^{-1}$ trained on var24 only with `finetune_inverse.py --models var24`). Before scoring, it asserts that none of the 512 var24 holdout samples appears, by token sequence or by image bytes, in `<DIR>/var/var24` of any `--overlap-data` directory.

```bash
/workspace/venv/bin/python scripts/finetune_inverse.py --family var --epochs 5 --models var24 \
    --out /workspace/cache/stage5/inv/var24_5ep
/workspace/venv/bin/python scripts/score_stage5_pilot.py --single single5=/workspace/cache/stage5/inv/var24_5ep/final.pt
```

| argument | default | meaning |
|---|---|---|
| `--single NAME=PATH` | none | a fine-tuned $D^{-1}$ checkpoint scored as inverse `NAME` (repeatable; `E` and `shared` are reserved) |
| `--overlap-data DIR` | `/workspace/cache/stage2/gen` | generated set that must not contain any var24 holdout sample (repeatable) |
| `--batch-size` | 64 | images per GPU batch (keep it fixed across runs that will be compared) |
| `--limit N` | none | debug: the first N holdout images of each depth |
| `--out` | `/workspace/cache/stage5/scores.csv` | output CSV |

#### Outputs

`--out` is a long-format CSV with one row per image and inverse: `label, index, inverse`, then

| column | meaning |
|---|---|
| `quant, enc1, enc2, cal, comb, tok_match` | `tracer.signals.var_signals` with `n_iters=0` (greedy quantization) |
| `tok_recovery` | fraction of the 680 greedy tokens $Q(D^{-1}(x))$ equal to the true tokens |
| `inv_mse` | mean $(D^{-1}(x) - f_Z)^2$ with $f_Z$ built from the true tokens: the Eq. 6 training loss per image |

`<out stem>.config.json` records, per inverse, the checkpoint, `--limit`, the row count, the git commit (and whether the tree was dirty) and the full command line. A rerun skips inverses already scored with the same checkpoint and `--limit`, and exits if a name is reused with a different checkpoint or `--limit` differs. Convolutions run in full fp32 (TF32 off) with deterministic cuDNN, so a rerun at the same `--batch-size` gives identical scores, and a different batch size changes about 1 recovered token in $10^5$. With TF32, 2-4% of greedy VAR tokens change with the batch size, so recovered tokens differ from `stage3/tokens` (extracted with TF32) by a few percent, while mean token recovery agrees. About 2.5 minutes per inverse for the full 2,048 images on the H100.

#### Quick check

```bash
/workspace/venv/bin/python scripts/score_stage5_pilot.py --limit 16 --out /tmp/stage5_dry/scores.csv \
    --single dup=/workspace/cache/stage2/inv/var/final.pt   # dup must equal shared
```

## Stage 5 full: one inverse decoder per model, end to end

### run_stage5_full.sh

```bash
SMOKE=1 bash scripts/run_stage5_full.sh                     # 1 new shard per model, 1 epoch on 32 images, --limit 64
tmux new -s stage5full 'bash scripts/run_stage5_full.sh'    # estimated 2.5-3.5 h on the H100 (VAR about 50 min first)
```

Stage 2 with one change: one $D^{-1}$ per model instead of one per family. Outputs and logs (`logs/`) go to `/workspace/cache/stage5full`. Steps, VAR first and then RAR, each skipped if its output exists (interrupted steps resume):

0. Preflight: `stage5/inv/var24_full` (reused for var24) was trained on `stage5/gen` with `--max-per-model 9940 --holdout-per-model 100` and every other setting of `stage2/inv/var/config.json`. Then `assemble_stage5_gen.py --seed-check-only --seed-scope family` for all 7 new models, against `stage2/gen`, `stage3/gen` and `stage5/gen_extra`, before anything is generated.
1. `gen_extra/<family>/<model>`: 7,680 new images for each model except var24 (`generate_finetune_data.py --models ... --seed 10 --per-model 7680`, 15 shards). Skipped if every model's last shard exists.
2. `gen/<family>/<model>`: `assemble_stage5_gen.py --labels <all 4>` symlinks each model's 5 `stage2/gen` shards and 15 new ones (10,240 rows); var24's second source is the pilot's `stage5/gen_extra` (`--extra-for`), and its links are checked to equal `stage5/gen/var/var24`. Skipped if `gen/<family>/config.json` exists.
3. `inv/<model>`: `finetune_inverse.py --models <model> --data stage5full/gen --max-per-model 9940 --holdout-per-model 100` (9,840 training images) with epochs, batch size, learning rate, StepLR and seed read from `stage2/inv/<family>/config.json` (RAR 10 epochs, VAR 5). Not run for var24.
4. `features/<model>/<family>.csv`: `extract_features.py --var-iters 0 --inv-ckpt <the model's own final.pt>` for all 11,050 task images; read with `tracer.stage5.load_model_features`.

Nothing is written to `stage2`, `stage3` or `stage5`. `SMOKE=1` writes to `/workspace/cache/stage5full_smoke` instead: 1 new shard (512 images) per model, 40 images per model with the last 8 held out, 1 epoch, features `--limit 64`; var24 still reads the pilot's (real) data and inverse.
