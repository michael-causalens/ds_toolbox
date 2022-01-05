"""
regression_utils.py

Helper functions for regression modelling
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import check_array
from typing import Optional, List, Dict, Callable


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


def reg_metrics_from_model(X, y, model, model_name: Optional[str] = None, as_dataframe=True,
                           extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
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
    extra_metrics : dict, optional
        The default metrics are MAE, MSE, pearson r and R^2.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.explained_variance_score}
    precision: int, optional
        Number of signficant figures to display in table

    Returns
    -------
    pd.DataFrame or Series.
    """

    yhat = model.predict(X)
    return reg_metrics_from_pred(y, yhat, model_name=model_name, as_dataframe=as_dataframe,
                                 extra_metrics=extra_metrics, precision=precision)


def reg_metrics_from_models(X, y, models: List, model_names: Optional[List[str]] = None,
                            extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
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
    extra_metrics : dict, optional
        The default metrics are MAE, MSE, pearson r and R^2.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.explained_variance_score}
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
        df_metrics[model_name] = reg_metrics_from_model(X, y, model, model_name, as_dataframe=False,
                                                        extra_metrics=extra_metrics, precision=precision)
    return df_metrics


def reg_metrics_from_pred(y, yhat, model_name: Optional[str] = None, as_dataframe=True,
                          extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
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
    extra_metrics : dict, optional
        The default metrics are MAE, MSE, pearson r and R^2.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.explained_variance_score}
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

    metrics_dict = {"MAE": mean_absolute_error,
                    "MSE": mean_squared_error,
                    "pearson": np.corrcoef,
                    f"R\N{superscript TWO}": r2_score}
    if extra_metrics is not None:
        metrics_dict.update(extra_metrics)
    for metric_string, metric_function in metrics_dict.items():
        if metric_string == "pearson":
            df_metrics.loc[metric_string] = round(metric_function(y, yhat)[0, 1], precision)
        else:
            df_metrics.loc[metric_string] = round(metric_function(y, yhat), precision)
    if as_dataframe:
        return df_metrics.to_frame(model_name)
    else:
        return df_metrics


def reg_metrics_from_preds(y, yhat_lst, model_names: Optional[List[str]] = None,
                           extra_metrics: Optional[Dict[str, Callable]] = None, precision: Optional[int] = None):
    """
    Return a table of regression metrics for a truth and list of predictions arrays.

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
        The default metrics are MAE, MSE, pearson r and R^2.
        Extra metrics can be passed in the form {"name" : func}
        where func is a callable of two arrays (y and yhat) that returns a single float.
        e.g. {"explained_var": sklearn.metrics.explained_variance_score}
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
        df_metrics[model_name] = reg_metrics_from_pred(y, yhat, model_name, as_dataframe=False,
                                                       extra_metrics=extra_metrics, precision=precision)
    return df_metrics
