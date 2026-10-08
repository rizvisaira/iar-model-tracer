# scripts

Long-running jobs. Run them in tmux with the project environment, and they write their outputs to `/workspace/cache` (outside the repo, never committed).

## extract_features.py

Computes the Stage 1 provenance signals (see [tracer/README.md](../tracer/README.md#signals)) for every image, using one family's original tokenizer. All sizes in a family share the tokenizer, so this runs once per family, not once per model. Every image is scored, including the other family's images and outliers.

```bash
cd /workspace/iar-model-tracer
tmux new -s feats
/workspace/venv/bin/python scripts/extract_features.py --family rar
/workspace/venv/bin/python scripts/extract_features.py --family var --var-iters 0
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

Join the two families on `key`, as `notebooks/001_stage_1.ipynb` does.

### Resuming and recomputing

- Shards are written atomically (`.tmp`, then renamed), so the job can be killed at any time. Rerunning the same command skips finished shards and re-merges.
- If the arguments differ from `<out>/<family>/config.json` (apart from `--batch-size`), the script exits instead of mixing configurations. Delete `<out>/<family>/` to recompute.

### Quick check

```bash
/workspace/venv/bin/python scripts/extract_features.py --family rar --splits val --limit 8 --out /tmp/feats_test
```
