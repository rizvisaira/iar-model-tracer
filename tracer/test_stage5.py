"""Checks for tracer/stage5.py (CPU, under a minute):

    python -m tracer.test_stage5

1. NearestModelThreshold on a hand-made example: fitted mu_ / sd_ / tau_ match their definitions; the nearest model
   wins; a nearest z above its own tau gives "outlier"; exact ties go to the first model of score_cols; train with
   outliers is rejected.
2. With one pseudo-model per family (rarb, var16), it equals NearestFamilyThreshold with size_cols=None (which
   predicts rarb / var16) on random data.
3. The same equivalence on the real Stage 2 features (stage2/features, scores rar_quant / var_cal), which also
   reproduces 002's val family accuracy (96.0%). Skipped if the features are missing.
4. load_model_features joins per-model tables (here Stage 2's tables copied under 8 model names), renames the
   columns and checks the checkpoints recorded in each config.json; with limit, it loads truncated tables (as
   written by extract_features.py --limit) and without it rejects them.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CACHE, FAMILY_OF
from .metrics import report
from .stage1 import NearestFamilyThreshold
from .stage5 import KNOWN_MODELS, NearestModelThreshold, load_model_features, model_score_cols


def check_hand_made():
    train = pd.DataFrame({"label": ["var16"] * 4 + ["rarb"] * 4,
                          "s_var16": [1.0, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0],
                          "s_rarb": [90.0, 80.0, 70.0, 60.0, 1.0, 2.0, 4.0, 8.0]})
    m = NearestModelThreshold({"var16": "s_var16", "rarb": "s_rarb"}, q=0.75).fit(train)
    for name, col in m.score_cols.items():
        s = np.log(train.loc[train.label == name, col].values)
        assert np.isclose(m.mu_[name], s.mean()) and np.isclose(m.sd_[name], s.std(ddof=0))
        assert np.isclose(m.tau_[name], np.quantile((s - s.mean()) / s.std(), 0.75))
    z = lambda name, v: (np.log(v) - m.mu_[name]) / m.sd_[name]  # noqa: E731
    df = pd.DataFrame({"s_var16": [2.0, 60.0, 1e6], "s_rarb": [80.0, 2.0, 1e6]})
    assert z("var16", 2.0) < m.tau_["var16"] and z("rarb", 2.0) < m.tau_["rarb"]
    assert m.predict(df).tolist() == ["var16", "rarb", "outlier"], m.predict(df).tolist()
    assert m.predict_family(df).tolist() == ["var", "rar", "outlier"]
    # exact tie: both models have the same train scores, and the row has the same score for both
    same = pd.DataFrame({"label": ["var16"] * 4 + ["rarb"] * 4, "s_var16": [1.0, 2.0, 3.0, 4.0] * 2})
    same["s_rarb"] = same.s_var16
    row = pd.DataFrame({"s_var16": [2.5], "s_rarb": [2.5]})
    for first, second in [("var16", "rarb"), ("rarb", "var16")]:
        tied = NearestModelThreshold({first: f"s_{first}", second: f"s_{second}"}, q=0.75).fit(same)
        assert tied.predict(row).tolist() == [first], "ties go to the first model of score_cols"
    try:
        m.fit(pd.concat([train, pd.DataFrame({"label": ["outlier"], "s_var16": [1.0], "s_rarb": [1.0]})]))
    except AssertionError:
        pass
    else:
        raise AssertionError("fit accepted an outlier in train")
    print("1. hand-made example: ok")


def check_random_equivalence(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    fam = rng.choice(["rar", "var"], n)
    shift = np.where(fam == "rar", -1.0, 1.0)
    df = pd.DataFrame({"family": fam, "label": np.where(fam == "rar", "rarb", "var16"),
                       "rar_x": np.exp(rng.normal(shift, 1.0)), "var_x": np.exp(rng.normal(-shift, 1.0))})
    train, test = df.iloc[:200], df.iloc[200:]
    nft = NearestFamilyThreshold({"rar": "rar_x", "var": "var_x"}, q=0.9).fit(train)
    nmt = NearestModelThreshold({"rarb": "rar_x", "var16": "var_x"}, q=0.9).fit(train)
    assert (nft.predict(test) == nmt.predict(test)).all()
    assert (nft.predict(test) == "outlier").any() and (nft.predict(test) == "rarb").any()
    print("2. equals NearestFamilyThreshold with one pseudo-model per family (random data): ok")


def check_stage2_equivalence():
    d = CACHE / "stage2" / "features"
    if not (d / "rar.csv").exists() or not (d / "var.csv").exists():
        print("3. skipped: no Stage 2 features")
        return
    rar, var = pd.read_csv(d / "rar.csv"), pd.read_csv(d / "var.csv")
    df = rar.merge(var.drop(columns=["split", "label", "family", "name"]), on="key", validate="1:1")
    train, val = df[df.split == "train"].copy(), df[df.split == "val"]
    nft = NearestFamilyThreshold({"rar": "rar_quant", "var": "var_cal"}, q=0.95).fit(train)
    train["label"] = train.family.map({"rar": "rarb", "var": "var16"})  # one pseudo-model per family
    nmt = NearestModelThreshold({"rarb": "rar_quant", "var16": "var_cal"}, q=0.95).fit(train)
    assert (nft.predict_family(val) == nmt.predict_family(val)).all()
    fam_acc = report(val.label, nmt.predict(val))["family_accuracy"]
    assert round(fam_acc, 3) == 0.960, fam_acc
    print(f"3. equals NearestFamilyThreshold on Stage 2 features; val family accuracy {fam_acc:.3f}: ok")


def check_loader():
    d = CACHE / "stage2" / "features"
    if not (d / "rar.csv").exists() or not (d / "var.csv").exists():
        print("4. skipped: no Stage 2 features")
        return
    src = {fam: pd.read_csv(d / f"{fam}.csv") for fam in ("rar", "var")}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ckpts = {m: f"/ckpt/{m}/final.pt" for m in KNOWN_MODELS}
        for m in KNOWN_MODELS:
            fam = FAMILY_OF[m]
            (root / m / fam).mkdir(parents=True)
            (root / m / fam / "config.json").write_text(json.dumps({"inv_ckpt": ckpts[m], "var_iters": 0}))
            src[fam].to_csv(root / m / f"{fam}.csv", index=False)
        df = load_model_features(root, ckpts)
        assert len(df) == len(src["rar"]) and df.key.tolist() == src["rar"].key.tolist()
        assert not any(c.startswith(("rar_", "var_")) for c in df.columns)
        for m, col in model_score_cols().items():
            fam = FAMILY_OF[m]
            assert np.array_equal(df[col].values, src[fam][f"{fam}_{col.split('_', 1)[1]}"].values), col
        for m in KNOWN_MODELS:  # as written by extract_features.py --limit 64
            fam = FAMILY_OF[m]
            src[fam].head(64).to_csv(root / m / f"{fam}.csv", index=False)
        assert len(load_model_features(root, ckpts, limit=64)) == 64
        try:
            load_model_features(root, ckpts)
        except AssertionError:
            pass
        else:
            raise AssertionError("loader accepted 64-row tables without limit")
        try:
            load_model_features(root, {**ckpts, "rarxl": "/ckpt/wrong.pt"}, limit=64)
        except AssertionError:
            pass
        else:
            raise AssertionError("loader accepted a table computed with another checkpoint")
    print(f"4. load_model_features: {df.shape[1]} columns, checkpoints checked: ok")


if __name__ == "__main__":
    check_hand_made()
    check_random_equivalence()
    check_stage2_equivalence()
    check_loader()
    print("all stage5 checks passed")
