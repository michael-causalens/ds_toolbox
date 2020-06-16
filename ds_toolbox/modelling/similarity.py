"""
similarity.py

> Functions for similarity between variables. Useful for feature selection.

"""
import numpy as np
from scipy.stats import f_oneway, ks_2samp
from scipy.spatial.distance import pdist, squareform
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


def distcorr(x, y):
    """
    Compute the distance correlation between two arrays
    @TODO: Simplify this to only be valid for 1D arrays

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    Float

    Example
    -------
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if np.prod(x.shape) == len(x):
        x = x[:, None]
    if np.prod(y.shape) == len(y):
        y = y[:, None]
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n = x.shape[0]
    if y.shape[0] != x.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(x))
    b = squareform(pdist(y))
    a = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    b = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (a * b).sum() / float(n * n)
    dcov2_xx = (a * b).sum() / float(n * n)
    dcov2_yy = (b * b).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
