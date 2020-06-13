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


def get_classification_metrics(model, metric_fn, features_train, features_test, y_train, y_test):
    """
    Print various classification metrics for a model on training and test data
    at the value of the threshold that optimizes `metric_fn`.

    Parameters
    ----------
    model : list
        List of pre-trained classification models. These must all have a `predict_prob()` method.
    metric_fn : obj
        classifiction metric from `sklearn.metrics`. Valid values are `accuracy_score`,
        `f1_score`, `precision_score`, `recall_score`   `roc_auc_score` or `average_precision_score`.
    features_train, features_test, y_train, y_test : array-like
        Train and test data.

    Returns
    -------
    A pandas DataFrame with classification metrics as index and a column each for train and test.
    """
    # @TODO: add balanced accuracy, logloss, Brier score
    train_pred, test_pred = get_model_probs(model, features_train, features_test)
    best_threshold = get_best_threshold(model, metric_fn, features_train, features_test, y_test)

    train_pred_best = (train_pred[:, 1] > best_threshold)
    test_pred_best = (test_pred[:, 1] > best_threshold)

    df_metrics = pd.DataFrame(columns=['Train', 'Test'])
    df_metrics.index.name = "Metric"

    metrics_dict = {"Accuracy": accuracy_score,
                    "Precision": precision_score,
                    "Recall": recall_score,
                    "F1": f1_score,
                    "ROCAUC": roc_auc_score,
                    "APC": average_precision_score}

    for metric_string, metric_function in metrics_dict.items():
        train_score = metric_function(y_train, train_pred_best)
        test_score = metric_function(y_test, test_pred_best)
        df_metrics.loc[metric_string] = [train_score, test_score]

    return df_metrics


def get_confusion_matrix():
    pass


def compare_model_metrics(models, metric_fn, features_train, features_test, y_train, y_test, model_names: list):
    """
    Compare classification performance metrics across several models.
    Note that because a classifier performance depends on the chosen threshold,
    first this finds the threshold for each model that optimizes `metric_fn`, then compares
    the models' performance in several metrics, but with each at its "best threshold"
    for a given user-provided metric (e.g. f1_score).


    Parameters
    ----------
    models : list
        List of pre-trained classification models. These must all have a `predict_prob()` method.
    metric_fn : obj
        classifiction metric from `sklearn.metrics`. Valid values are `accuracy_score`,
        `f1_score`, `precision_score`, `recall_score`   `roc_auc_score` or `average_precision_score`.
    features_train, features_test, y_train, y_test : array-like
        Train and test data.
    model_names : list of strs
        Names of models.

    Returns
    -------
    A pandas DataFrame with classification metrics as index and a column for each model.
    """

    if not isinstance(models, list):
        raise TypeError("First argument (models) should be a list of pre-trained models")

    if not len(models) == len(model_names):
        raise ValueError("models and model_names should be the same length")

    df_metrics = pd.DataFrame()
    for i, model in enumerate(models):
        model_metrics = get_classification_metrics(model, metric_fn, features_train, features_test, y_train, y_test)
        model_metrics = model_metrics["Test"]  # compare test set only

        model_metrics.name = model_names[i]
        df_metrics = pd.concat([df_metrics, model_metrics], axis=1, sort=False)

    return df_metrics


def compare_model_predictions(y_preds: list, y_true, model_names: list):
    """
    Compare classification metrics across several models to a single test set
    in case where one or more model does not have a `predict_prob()` method and
    so `compare_model_metrics()` cannot be used.
    This compares the output predictions from each model directly.
    Useful for comparing ML predictions to simple baselines (constant, random etc.)

    Parameters
    ----------
    y_preds : list
        List (of length n_models) of numpy arrays of predicted labels
    y_true : array-like
        Numpy array of actual labels.
    model_names : list
        List of strings of model names.

    Returns
    -------
    A pandas DataFrame with classification metrics as index and a column for each model.
    """
    metrics_dict = {"Accuracy": accuracy_score,
                    "Precision": precision_score,
                    "Recall": recall_score,
                    "F1": f1_score,
                    "ROCAUC": roc_auc_score,
                    "APC": average_precision_score}

    n_models = len(y_preds)
    assert len(model_names) == n_models, "y_preds and model_names must be same length"

    df_metrics = pd.DataFrame()

    for i in range(n_models):
        y_pred = y_preds[i]
        model_name = model_names[i]

        for metric_string, metric_function in metrics_dict.items():
            metric_result = metric_function(y_true, y_pred)
            df_metrics.loc[metric_string, model_name] = metric_result

    return df_metrics


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


def plot_features_classes(features, target, ncols=5, nbins=50, density=True, legend=False):
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
        x_class0 = features[target == classes[0]][:, i]
        x_class1 = features[target == classes[1]][:, i]

        bins_start = min(min(x_class0), min(x_class1))
        bins_end = max(max(x_class0), max(x_class1))
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
