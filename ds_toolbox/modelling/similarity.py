"""
similarity.py

> Functions for similarity between variables. Useful for feature selection.

"""
import numpy as np
from scipy.stats import f_oneway, ks_2samp
from collections import Counter


def fscore_similarity(feature, target, error="ignore", return_pval=False):
    """
    Calculate ANOVA f-score between feature and binary target
    @TODO: Support categorical targets
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html

    Parameters
    ----------
    feature, target : array-like
        Must have same shape
    error : str
        How to treat non-binary target variables. Options are ["raise", "zero", "ignore"]
    return_pval :  bool, default False
        Return a tuple of (fscore, pval)

    Returns
    -------
    Float F-score
    """

    if feature.shape[0] != target.shape[0]:
        raise ValueError("Features and target must have same number of rows")

    counts = Counter(target)
    classes = list(counts.keys())
    if len(classes) != 2:
        if error == "raise":
            raise ValueError("Only valid for binary targets")
        elif error == "zero":
            return 0
        elif error == "ignore":
            return np.nan
        else:
            raise ValueError("Invalid option for 'error'")

    x_class0 = feature[target == classes[0]]
    x_class1 = feature[target == classes[1]]
    f_score, pval = f_oneway(x_class0, x_class1)
    if return_pval:
        return f_score, pval
    else:
        return f_score


def ks_similiarity(feature, target, error="ignore", return_pval=False):
    """
    Calculate Kolmogorov-Smirnov statistic between feature and binary target
    @TODO: Support categorical targets
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

    Parameters
    ----------
    feature, target : array-like
        Must have same shape
    error : str
        How to treat non-binary target variables. Options are ["raise", "zero", "ignore"]
    return_pval :  bool, default False
        Return a tuple of (KS stat, pval)

    Returns
    -------
    Float KS statistic
    """
    counts = Counter(target)
    classes = list(counts.keys())
    n_classes = len(classes)
    if n_classes != 2:
        if error == "raise":
            raise ValueError("Only valid for binary targets")
        elif error == "zero":
            return 0
        elif error == "ignore":
            return np.nan

    x_class0 = feature[target == classes[0]]
    x_class1 = feature[target == classes[1]]
    ks, pval = ks_2samp(x_class0, x_class1)
    if return_pval:
        return ks, pval
    else:
        return ks
