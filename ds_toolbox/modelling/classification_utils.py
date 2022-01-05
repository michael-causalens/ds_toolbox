"""
classification_utils.py

Helper functions for classification problems, including exploration, model testing and visualisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from typing import Optional, List, Dict, Callable


def get_model_probs(model, features_train, features_test):
    """
    Get class probabilities for train and test samples from pre-trained model.

    Parameters
    ----------
    model : obj
        Pre-trained classification model. Typically a scikit-learn classification
        estimator but can in principle be any object with a `predict_prob()` method.
    features_train, features_test : array-like
        Train and test feature arrays. Can be numpy arrays, dataframes, series or lists.

    Returns
    -------
    Tuple of numpy arrays (train and test) of dimension `n_samples x n_classes`
    containing class probabilities for each sample
    """

    train_pred = model.predict_proba(features_train)
    test_pred = model.predict_proba(features_test)
    return train_pred, test_pred


def get_best_threshold(model, metric_fn, features_train, features_test, y_test):
    """
    Get the value of the model's classification probability threshold that maximises the
    specified classification metric over the test set.

    Parameters
    ----------
    model : obj
        Pre-trained classification model. Typically a scikit-learn classification
        estimator but can in principle be any object with a `predict_prob()` method.
    metric_fn : obj
        classifiction metric from `sklearn.metrics`. Valid values are `accuracy_score`,
        `f1_score`, `precision_score`, `recall_score`   `roc_auc_score` or `average_precision_score`.
    features_train, features_test, y_test : array-like
        Train and test data output from `train_test_split()`

    Returns
    -------
    best_threshold: float
        Threshold that maximises `metric_fn`
    """

    train_pred, test_pred = get_model_probs(model, features_train, features_test)

    metric_max = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 1, 0.05):

        test_preds_at_threshold = (test_pred[:, 1] > threshold)  # .astype(int)
        test_metric_at_threshold = metric_fn(y_test, test_preds_at_threshold)

        if test_metric_at_threshold > metric_max:
            metric_max = test_metric_at_threshold
            best_threshold = threshold

    return best_threshold


def plot_pred_class_distributions(features, y_pred, y_true, feature_names=None, density=False, nbins=10):
    """
    Plot distributions of each feature in X_feat split by predicted classes,
    i.e a row for each feature, a column for each class and on each subplot
    separate distributions for correctly and incorrectly labelled events.

    Parameters
    ----------
    features : array-like
        Numpy array of features data, of shape (n_samples, n_features)
    y_pred : array-like
        Numpy array of predicted labels
    y_true : array-like
        Numpy array of true labels
    feature_names : list (optional)
        List of string feature names. Default is "x1", "x2" etc.
    density : bool (default=False)
        Normalise the distributions to unit area
    nbins : int
        Number of bins in histogram. Default is 10.

    Returns
    -------
    matplotlib.figure.Figure object with (n_features * n_classes) subplots
    """

    n_features = features.shape[1]
    n_classes = len(set(y_pred))

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(n_features)]

    fig, axs = plt.subplots(nrows=n_features, ncols=n_classes, figsize=(10, 10))
    for row_idx in range(len(axs)):

        ax_row = axs[row_idx]
        for col_idx in range(len(ax_row)):

            correct_pred_mask = (y_pred == y_true) & (y_true == col_idx)
            incorrect_pred_mask = (y_pred != y_true) & (y_true == col_idx)

            x_correct = features[correct_pred_mask][:, row_idx]
            x_incorrect = features[incorrect_pred_mask][:, row_idx]

            bins_start = min(min(x_correct), min(x_incorrect))
            bins_end = max(max(x_correct), max(x_incorrect))
            bins = np.linspace(bins_start, bins_end, nbins+1)
            
            ax = ax_row[col_idx]
            ax.hist(x_correct, bins=bins, density=density, label="pred correct", color="forestgreen", alpha=0.6)
            ax.hist(x_incorrect, bins=bins, density=density, label="pred incorrect", color="red", alpha=0.4)
            ax.set_xlabel(f"{feature_names[row_idx]} for class {col_idx}")

            if density:
                ax.set_ylabel("Probability density")
            else:
                ax.set_ylabel("Frequency")

            ax.legend()
    fig.suptitle("Distribution of predicted labels", y=1.02)
    fig.tight_layout()


def plot_features_classes(features, target, ncols=5, nbins=50, density=True, legend=False, xrange=None):
    """
    Plot histograms of numerical features separated by class labels.

    Parameters
    ----------

    features : array-like
        Features array of shape (n_samples, n_features)
    target : array-like
        Binary classes array of shape (n_samples, )
    ncols : int, default 5
        Number of plots per row
    nbins : int, default 50
        Number of equal-sized bins in distributions.
    density : bool, default True
        If False, plot counts instead of probability density
    legend : bool, default False
        Legend on each plot
    xrange : tuple, optional
        Start and end of bin ranges. By default uses the endpoints of the data.

    Returns
    -------
    n_feat subplots each with two histograms, arranged into ncols columns
    """

    if features.shape[0] != target.shape[0]:
        raise ValueError("Features and target must have same number of rows")

    classes = list(Counter(target))
    if len(classes) != 2:
        raise ValueError(f"Only 2 classes supported, {len(classes)} given")

    n_feat = features.shape[1]
    nrows = int(np.ceil(n_feat / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3), )
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i >= features.shape[1]:
            ax.set_visible(False)
            continue

        x = features[:, i]
        notnan_mask = ~np.isnan(x)
        x_notna = x[notnan_mask]
        target_notna = target[notnan_mask]
        x_class0 = x_notna[target_notna == classes[0]]
        x_class1 = x_notna[target_notna == classes[1]]

        if xrange is None:
            bins_start = min(min(x_class0), min(x_class1))
            bins_end = max(max(x_class0), max(x_class1))
        else:
            bins_start, bins_end = xrange
        bins = np.linspace(bins_start, bins_end, nbins + 1)

        ax.hist(x_class0, bins=bins, density=density, label=0, color="forestgreen", alpha=0.6)
        ax.hist(x_class1, bins=bins, density=density, label=1, color="red", alpha=0.4)
        ax.set_xlabel(f"X_{str(i)}")

        ax.set_ylabel("Probability density")
        if legend:
            ax.legend()

    fig.suptitle("Distribution of labels", y=1.01)
    fig.tight_layout()


def _get_roc_curve(estimator, features, y_true):
    scores = estimator.predict_proba(features)[:, 1]
    fpr, tpr, thresh = roc_curve(y_true, scores)
    return fpr, tpr, thresh


def plot_rocs(estimators, features, y_true, model_names=None):
    """
    Plot ROC curves for classification models.

    Parameters
    ----------
    estimators : array-like
        List of fitted models, must have a `predict_proba()` method to get ROC curve
    features : array-like
        Numpy array of features data, of shape (n_samples, n_features)
    y_true : array-like
        Numpy array of true labels
    model_names : array-like (optional)
        List of strings of model names. Defaults to "model 1, 2" etc.

    Returns
    -------
    matplotlib.figure of ROC curves (true positive rate vs. false positive rate)
    """

    plt.figure(figsize=(6, 4))

    if model_names is None:
        model_names = [f"model {i + 1}" for i in range(len(estimators))]

    for i, estimator in enumerate(estimators):
        fpr, tpr, thresh = _get_roc_curve(estimator, features, y_true)
        plt.plot(fpr, tpr, label=model_names[i])

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend()
    plt.show()


def class_metrics_from_model(X, y, model, model_name: Optional[str] = None, as_dataframe=True,
                             extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
    """
    Return a list of classification metrics for a trained model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Features. Must be same shape as the model was trained on.
    y : array-like of shape (n, )
        Target.
    model : obj
        Any model with an sklearn-like predict() method.
    model_name : str, optional
        For column header. Uses a string representation of model if not given.
    as_dataframe : bool, default True
        Return a DataFrame instead of a Series
    extra_metrics : dict, optional
        The default metrics are accuracy, precision, recall, F1, ROCAUC, and average precision.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.logloss}
    precision: int, optional
        Number of signficant figures to display in table

    Returns
    -------
    pd.DataFrame or Series.
    """

    yhat = model.predict(X)
    return class_metrics_from_pred(y, yhat, model_name=model_name, as_dataframe=as_dataframe,
                                   extra_metrics=extra_metrics, precision=precision)


def class_metrics_from_models(X, y, models: List, model_names: Optional[List[str]] = None,
                              extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
    """
    Return a table of classification metrics for a list of trained models.
}
    Parameters
    ----------
    X : array-like of shape (n, p)
        Features. Must be same shape as the model was trained on.
    y : array-like of shape (n, )
        Target.
    models : list of obj
        Elements can be any model with an sklearn-like predict() method.
    model_names : list of str, optional
        For column headers in the table. Uses a string representation of the models if not given.
    extra_metrics : dict, optional
        The default metrics are accuracy, precision, recall, F1, ROCAUC, and average precision.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.logloss}
    precision: int, optional
        Number of signficant figures to display in table

    Returns
    -------
    pd.DataFrame
    """
    df_metrics = pd.DataFrame()
    if model_names is None:
        model_names = [str(m) for m in models]
    assert isinstance(model_names, list), f"Expected a list for model_names, got {type(model_names)}"
    assert len(models) == len(model_names), f"Length mismatch: models and model_names must be same length."
    for model, model_name in zip(models, model_names):
        df_metrics[model_name] = class_metrics_from_model(X, y, model, model_name, as_dataframe=False,
                                                          extra_metrics=extra_metrics, precision=precision)
    return df_metrics


def class_metrics_from_pred(y, yhat, model_name: Optional[str] = None, as_dataframe=True,
                            extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
    """
    Return a list of classification metrics for a truth and predictions array.

    Parameters
    ----------
    y : array-like of shape (n, )
        Target.
    yhat : array-like of shape (n, )
        Predictions
    model_name : str, optional
        For column header. Uses a string representation of model if not given.
    as_dataframe : bool, default True
        Return a DataFrame instead of a Series
    extra_metrics : dict, optional
        The default metrics are accuracy, precision, recall, F1, ROCAUC, and average precision.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.logloss}
    precision: int, optional
        Number of signficant figures to display in table

    Returns
    -------
    pd.DataFrame or Series.
    """
    assert y.shape == yhat.shape, f"Predictions and target must have the same shape. {y.shape} != {yhat.shape}"
    assert yhat.ndim == 1, f"Expected a scalar array."
    if model_name is None:
        if isinstance(yhat, pd.Series):
            if yhat.name is not None:
                model_name = yhat.name
            else:
                model_name = "model"
        else:
            model_name = "model"
    df_metrics = pd.Series(name=model_name, dtype=float)
    df_metrics.index.name = "Metric"
    if precision is None:
        precision = 3

    metrics_dict = {"Accuracy": accuracy_score,
                    "Precision": precision_score,
                    "Recall": recall_score,
                    "F1": f1_score,
                    "ROCAUC": roc_auc_score,
                    "APC": average_precision_score}
    if extra_metrics is not None:
        metrics_dict.update(extra_metrics)

    for metric_string, metric_function in metrics_dict.items():
        df_metrics.loc[metric_string] = round(metric_function(y, yhat), precision)
    if as_dataframe:
        return df_metrics.to_frame(model_name)
    else:
        return df_metrics


def class_metrics_from_preds(y, yhat_lst, model_names: Optional[List[str]] = None,
                             extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
    """
    Return a table of classification metrics for a truth and predictions arrays.

    Parameters
    ----------
    y : array-like of shape (n, )
        Target.
    yhat_lst : list of array-like of shape (n, )
        List of arrays of predictions from each model.
    model_names : str, optional
        For column headers in table. Defaults to "model1", "model2" etc. if not given.
        Or, if yhat is a pd.Series, tries to use the name of the Series.
    extra_metrics : dict, optional
        The default metrics are accuracy, precision, recall, F1, ROCAUC, and average precision.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.logloss}
    precision: int, optional
        Number of signficant figures to display in table

    Returns
    -------
    pd.DataFrame.
    """
    df_metrics = pd.DataFrame()
    if model_names is None:
        model_names = ["model_" + str(i + 1) for i in range(len(yhat_lst))]
    assert isinstance(model_names, list), f"Expected a list for model_names, got {type(model_names)}"
    assert len(model_names) == len(yhat_lst), f"Length mismatch: yhat_lst and model_names must be same length."
    for yhat, model_name in zip(yhat_lst, model_names):
        df_metrics[model_name] = class_metrics_from_pred(y, yhat, model_name, as_dataframe=False,
                                                         extra_metrics=extra_metrics, precision=precision)
    return df_metrics
