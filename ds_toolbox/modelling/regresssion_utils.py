"""
regression_utils.py

Helper functions for regression modelling
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, max_error, mean_absolute_error
from sklearn.utils import check_array
from typing import Optional, List
from warnings import warn


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
    warn("This function is deprecated and will be removed. Please use metrics_from_model() instead.")

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


def compare_regression_model_predictions(y_preds, y_true, model_names):
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
    warn("This function is deprecated and will be removed. Please use metrics_from_model() instead.")
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


def reg_metrics_from_model(X, y, model, model_name: Optional[str] = None, as_dataframe=True):
    """
    Return a list of regression metrics for a trained model.

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


    Returns
    -------
    pd.DataFrame or Series.
    """

    if model_name is None:
        model_name = str(model)
    df_metrics = pd.Series(name=model_name, dtype=float)
    df_metrics.index.name = "Metric"

    metrics_dict = {"MAE": mean_absolute_error,
                    "MSE": mean_squared_error,
                    "pearson": np.corrcoef,
                    f"R\N{superscript TWO}": r2_score}

    yhat = model.predict(X)
    for metric_string, metric_function in metrics_dict.items():
        if metric_string == "pearson":
            df_metrics.loc[metric_string] = round(metric_function(y, yhat)[0, 1], 3)
        else:
            df_metrics.loc[metric_string] = round(metric_function(y, yhat), 3)
    if as_dataframe:
        return df_metrics.to_frame(model_name)
    else:
        return df_metrics


def reg_metrics_from_models(X, y, models, model_names: Optional[List[str]] = None):
    """
    Return a table of regression metrics for a list of trained models.

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

    Returns
    -------
    pd.DataFrame
    """
    # @TODO: add custom metrics
    df_metrics = pd.DataFrame()
    if model_names is None:
        model_names = [str(m) for m in models]
    assert isinstance(model_names, list), f"Expected a list for model_names, got {type(model_names)}"
    assert len(models) == len(model_names), f"Length mismatch: models and model_names must be same length."
    for model, model_name in zip(models, model_names):
        df_metrics[model_name] = reg_metrics_from_model(X, y, model, model_name, as_dataframe=False)
    return df_metrics


def reg_metrics_from_pred(y, yhat, model_name: Optional[str] = None, as_dataframe=True):
    """
    Return a list of regression metrics for a truth and predictions array.

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

    metrics_dict = {"MAE": mean_absolute_error,
                    "MSE": mean_squared_error,
                    "pearson": np.corrcoef,
                    f"R\N{superscript TWO}": r2_score}
    for metric_string, metric_function in metrics_dict.items():
        if metric_string == "pearson":
            df_metrics.loc[metric_string] = round(metric_function(y, yhat)[0, 1], 3)
        else:
            df_metrics.loc[metric_string] = round(metric_function(y, yhat), 3)
    if as_dataframe:
        return df_metrics.to_frame(model_name)
    else:
        return df_metrics


def reg_metrics_from_preds(y, yhat_lst, model_names: Optional[List[str]] = None):
    """
    Return a table of regression metrics for a truth and predictions arrays.

    Parameters
    ----------
    y : array-like of shape (n, )
        Target.
    yhat_lst : list of array-like of shape (n, )
        List of arrays of predictions from each model.
    model_names : str, optional
        For column headers in table. Defaults to "model1", "model2" etc. if not given.
        Or, if yhat is a pd.Series, tries to use the name of the Series.

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
        df_metrics[model_name] = reg_metrics_from_pred(y, yhat, model_name, as_dataframe=False)
    return df_metrics
