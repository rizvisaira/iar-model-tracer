# Final results

## Val metrics per stage (450 images)

| stage | accuracy | family accuracy | outlier recall | outlier precision | size accuracy given family |
|---|---|---|---|---|---|
| Stage 1 | 0.236 | 0.838 | 0.320 | 0.533 | 0.249 |
| Stage 2 | 0.316 | 0.960 | 0.880 | 0.863 | 0.253 |
| Stage 3 | 0.316 | 0.960 | 0.880 | 0.863 | 0.253 |
| Stage 4 | 0.940 | 0.960 | 0.880 | 0.863 | 0.977 |

## Final model (Stage 2 family / outlier + Stage 4 size)

- val accuracy **0.940**, 95% CI [0.918, 0.960]
- size accuracy given family **0.977**, 95% CI [0.961, 0.990]
- family accuracy 0.960, outlier recall / precision 0.880 / 0.863
- bootstrap: 2000 resamples of the val images, seed 0

### Val errors (27 of 450)

| error type | errors | share of errors | decided by |
|---|---|---|---|
| family error (RAR <-> VAR) | 5 | 18.5% | Stage 2 (family / outlier) |
| outlier false positive | 7 | 25.9% | Stage 2 (family / outlier) |
| outlier miss | 6 | 22.2% | Stage 2 (family / outlier) |
| size error (correct family) | 9 | 33.3% | Stage 4 (size) |
| total | 27 | 100.0% |  |

### Predicted shares on test (9,800 images), next to val

| label | test predicted | val predicted | val true |
|---|---|---|---|
| var16 | 0.086 | 0.129 | 0.111 |
| var20 | 0.084 | 0.109 | 0.111 |
| var24 | 0.077 | 0.109 | 0.111 |
| var30 | 0.076 | 0.109 | 0.111 |
| rarb | 0.072 | 0.109 | 0.111 |
| rarl | 0.072 | 0.113 | 0.111 |
| rarxl | 0.067 | 0.100 | 0.111 |
| rarxxl | 0.067 | 0.109 | 0.111 |
| outlier | 0.399 | 0.113 | 0.111 |

| family | test predicted | val predicted | val true |
|---|---|---|---|
| rar | 0.278 | 0.431 | 0.444 |
| var | 0.323 | 0.456 | 0.444 |
| outlier | 0.399 | 0.113 | 0.111 |

### Expected test accuracy, if val's per-group rates carry over to test (Section 4.5)

| scenario | test outlier share | expected accuracy | 95% CI low | 95% CI high | expected predicted-outlier share |
|---|---|---|---|---|---|
| as val (50 / 450) | 0.111 | 0.940 | 0.918 | 0.961 | 0.113 |
| 20% | 0.200 | 0.934 | 0.910 | 0.958 | 0.190 |
| 30% | 0.300 | 0.927 | 0.897 | 0.957 | 0.276 |
| 40% | 0.400 | 0.921 | 0.881 | 0.957 | 0.363 |
| 60% | 0.600 | 0.907 | 0.850 | 0.958 | 0.535 |
| implied by the observed predicted-outlier share | 0.442 | 0.918 | 0.867 | 0.956 | 0.399 |
| implied by the lower-tail mixture check (z0 = 0) | 0.430 | 0.918 | 0.877 | 0.957 | 0.389 |

Observed predicted-outlier share on test: 0.399. Scenarios whose expected predicted-outlier share is far from it are inconsistent with the assumption (e.g. an 11% outlier share would need about a third of the in-family test images to be called outlier; Section 4.5, last column).

## Submission checks (Section 3)

| check | submission.csv (test) | submission_val.csv (val) |
|---|---|---|
| header is exactly image_name,label | PASS | PASS |
| one row per image (test 9,800 / val 450) | PASS | PASS |
| every image appears exactly once, no extras | PASS | PASS |
| every image_name ends with .png | PASS | PASS |
| every label is one of the 9 allowed strings | PASS | PASS |
| no empty fields, exactly 2 columns per row | PASS | PASS |
| round-trips through pandas | PASS | PASS |
| round-trips through csv.reader (stdlib) | PASS | PASS |
| template's loader | N/A (no loader in task_template.py) | N/A (no loader in task_template.py) |
| byte-identical to the template's writer (CRLF, no BOM) | PASS | PASS |
| rows in the template's order (test: TestDataset, val: ImageFolder) | PASS | PASS |
| labels equal this notebook's final predictions | PASS | PASS |
| labels equal 004's saved predictions | PASS | PASS |

## Files

- `submission.csv`: test predictions (the submission)
- `submission_val.csv`: val predictions, same format
- `results/confusion_val.png`
- `results/errors_val.png`
- `results/expected_test_accuracy.csv`
- `results/feature_ablation.csv` (Section 6)
- `results/final_summary.json`
- `results/final_summary.md`
- `results/label_shares.png`
- `results/outliers_test_val.png`
- `results/stages_val.csv`
- `results/stages_val.png`
- `results/zscores_val_test.png`
- `/workspace/cache/final/pred_val.csv` (cache, not in the repo)
- `/workspace/cache/stage1/metrics.json` (cache, not in the repo)
- `/workspace/cache/stage2/metrics.json` (cache, not in the repo)
- `/workspace/cache/stage3/metrics.json` (cache, not in the repo)
- `/workspace/cache/stage4/metrics.json` (cache, not in the repo)
