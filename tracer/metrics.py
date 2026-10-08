"""Metrics shared across stages: overall accuracy plus the family / outlier / size breakdown."""
import numpy as np
import pandas as pd

from .common import FAMILIES, FAMILY_OF, LABELS


def report(y_true, y_pred) -> dict:
    """y_true, y_pred: label strings from LABELS. Family = label prefix (FAMILY_OF).

    Returns:
        n                           number of images
        accuracy                    exact 9-class label match (the task metric)
        family_accuracy             predicted family == true family, over rar / var / outlier
        outlier_recall              share of true outliers predicted "outlier" (NaN if there are none)
        outlier_precision           share of "outlier" predictions that are true outliers (NaN if none predicted)
        size_accuracy_given_family  exact match among rar / var images whose family was predicted correctly
        family_recall_<f>           share of family-f images predicted as family f, for each f present in y_true
    """
    t, p = pd.Series(list(y_true)), pd.Series(list(y_pred))
    assert t.isin(LABELS).all() and p.isin(LABELS).all()
    tf, pf = t.map(FAMILY_OF), p.map(FAMILY_OF)  # true / predicted family
    known_fam_ok = (tf == pf) & (tf != "outlier")  # rows where size accuracy is measured
    out = {
        "n": len(t),
        "accuracy": (t == p).mean(),
        "family_accuracy": (tf == pf).mean(),
        "outlier_recall": (pf[tf == "outlier"] == "outlier").mean() if (tf == "outlier").any() else np.nan,
        "outlier_precision": (tf[pf == "outlier"] == "outlier").mean() if (pf == "outlier").any() else np.nan,
        "size_accuracy_given_family": (t[known_fam_ok] == p[known_fam_ok]).mean() if known_fam_ok.any() else np.nan,
    }
    for fam in FAMILIES:
        if (tf == fam).any():
            out[f"family_recall_{fam}"] = (pf[tf == fam] == fam).mean()
    return out


def confusion(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    """Rows = true, columns = predicted. Rows only for labels present in y_true; all columns, zero-filled."""
    ct = pd.crosstab(pd.Series(list(y_true), name="true"), pd.Series(list(y_pred), name="pred"))
    return ct.reindex(index=[l for l in labels if l in set(y_true)], columns=labels, fill_value=0)


def family_confusion(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix at the family level (rar / var / outlier); inputs are full label strings."""
    return confusion(pd.Series(list(y_true)).map(FAMILY_OF), pd.Series(list(y_pred)).map(FAMILY_OF), FAMILIES)
