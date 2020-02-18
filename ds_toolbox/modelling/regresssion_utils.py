"""
regression_utils.py

Helper functions for regression modelling
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, max_error, mean_absolute_error
from sklearn.utils import check_array


def mean_abs_perc_error(y_true, y_pred):
    check_array([y_true])
    return np.mean(np.abs((y_true - y_pred) / y_true))


def median_abs_perc_error(y_true, y_pred):
    check_array([y_true])
    return np.median(np.abs((y_true - y_pred) / y_true))


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def symm_mean_abs_perc_error(y_true, y_pred, regulate=True):
    """
    Symmetric mean abolsute percentage error.

    Parameters
    ----------
    y_true : float or array-like
        Actuals
    y_pred : float or array-like
        Predicted values
    regulate : bool (default=True)
        Add a small epsilon to avoid division by zero

    Returns
    -------
    Float
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    if regulate:
        denom += np.finfo(float).eps
    return np.mean(2 * np.abs(y_pred - y_true) / denom)


def get_regression_metrics(model, X_train, X_test, y_train, y_test):
    """
    Print various regression metrics for a model on training and test data

    Parameters
    ----------
    model : obj
        Any object with a .predict() method
    X_train, X_test, y_train, y_test : array-like
        Train and test data.

    Returns
    -------
    A pandas DataFrame with regression metrics as index and a column each for its values for train and test.
    """

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    df_metrics = pd.DataFrame(columns=['Train', 'Test'])
    df_metrics.index.name = "Metric"

    metrics_dict = {"Mean abs error": mean_absolute_error,
                    "Mean abs % error": mean_abs_perc_error,
                    "Median abs % error": median_absolute_error,
                    "RMSE": root_mean_square_error,
                    "Max error": max_error,
                    f"R\N{superscript TWO}": r2_score}

    for metric_string, metric_function in metrics_dict.items():
        train_score = metric_function(y_train, train_pred)
        test_score = metric_function(y_test, test_pred)
        df_metrics.loc[metric_string] = [train_score, test_score]

    return df_metrics


def compare_regression_model_predictions(y_preds, y_true, model_names=None):
    """
    Compare regression metrics across several models to a single test set.

    Parameters
    ----------
    y_preds : list
        List (of length n_models) of numpy arrays of predictions.
    y_true : array-like
        Numpy array of actual.
    model_names : list
        List of strings of model names.

    Returns
    -------
    A pandas DataFrame with regression metrics as index and a column for each model.
    """
    metrics_dict = {"Mean abs error": mean_absolute_error,
                    "Mean abs % error": mean_abs_perc_error,
                    "Median abs % error": median_absolute_error,
                    "RMSE": root_mean_square_error,
                    "Max error": max_error,
                    f"R\N{superscript TWO}": r2_score}

    n_models = len(y_preds)
    if n_models != len(model_names):
        raise ValueError("y_preds and model_names should be the same length")

    df_metrics = pd.DataFrame()

    for i in range(n_models):
        y_pred = y_preds[i]
        model_name = model_names[i]

        for metric_string, metric_function in metrics_dict.items():
            metric_result = metric_function(y_true, y_pred)
            df_metrics.loc[metric_string, model_name] = metric_result

    return df_metrics
