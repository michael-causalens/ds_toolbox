"""
feature_engineering.py

> Time-series features like rolling averages, standard-deviations, z-scores
@TODO: Add Kalman filter features
"""
import numpy as np
import pandas as pd


def construct_rolling_features(data, windows, features=None, **kwargs):
    """
    Get rolling mean, std-dev, max over various windows
    Output columns have names like "input_col_5_step_mean" etc.

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.
    features : list of strs, optional
        Rolling features to calculate e.g. mean, std, max
    **kwargs
        Extra args for pandas.Series.rolling()

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Series object must have a 'name' attribute")

    if features is None:
        features = ["mean", "std", "max"]

    for window in windows:
        rolling = data.rolling(window=window, **kwargs)
        for feature in features:
            result = getattr(rolling, feature)()
            result.name = data.name + "_" + str(window) + "_step_" + feature
            lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def construct_ewm_features(data, windows, features=None, **kwargs):
    """
    Get exponential weighted moving mean and std-dev over various windows
    Output columns have names like "input_col_5_step_ewm_mean" etc.

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.
    features : list of strs, optional
        Which EWM features to calculate
    **kwargs
        Extra args for pandas.Series.ewm()

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Series object must have a 'name' attribute")

    if features is None:
        features = ["mean", "std"]

    for window in windows:
        ewm = data.ewm(span=window, **kwargs)
        for feature in features:
            result = getattr(ewm, feature)()
            result.name = data.name + "_" + str(window) + "_step_ewm_" + feature
            lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def ewmzscore(x, window, **kwargs):
    """
    Compute exponentially weighted moving z-score

    Parameters
    ----------
    x : pandas.Series
        Input data
    window : int
        Span parameter for EWM.
    **kwargs
        Extra args for pandas.Series.ewm()

    Returns
    -------
    pandas.Series
    """
    r = x.ewm(span=window, **kwargs)
    mu = r.mean()
    sigma = r.std()
    return (x - mu) / sigma


def construct_ewmzscore_features(data, windows, **kwargs):
    """
    Get exponential weighted moving z-score over various windows
    Output columns have names like "input_col_5_step_zscore" etc.

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.
    **kwargs
        Extra args for pandas.Series.ewm()

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Input Series must have a 'name' attribute")

    for window in windows:
        result = ewmzscore(data, window, **kwargs)
        result.name = data.name + "_" + str(window) + "_step_ewmzscore"
        lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def construct_time_features(data_df, method="sine"):
    """
    Calculate sine or cosine of hour of day and day of week as features. Useful if periodicity is suspected.

    Parameters
    ----------
    data_df : pandas.Dataframe
        Must be hourly frequency or higher.
    method: str, default "sine"
        "sine" or "cosine"

    Returns
    -------
    pandas.Dataframe with two columns: sine_hour and sine_day
    """
    sine_hour = pd.Series(np.sin(data_df.index.hour * 2 * np.pi / 24), index=data_df.index, name="sine_hour")
    sine_day = pd.Series(np.sin(data_df.index.dayofweek * 2 * np.pi / 7), index=data_df.index, name="sine_day")
    cos_hour = pd.Series(np.cos(data_df.index.hour * 2 * np.pi / 24), index=data_df.index, name="cosine_hour")
    cos_day = pd.Series(np.cos(data_df.index.dayofweek * 2 * np.pi / 7), index=data_df.index, name="cosine_day")

    if method == "sine":
        all_time_features = pd.concat([sine_hour, sine_day], axis=1)
    elif method == "cosine":
        all_time_features = pd.concat([cos_hour, cos_day], axis=1)
    else:
        raise ValueError(f"Invalid method {method}. Use either 'sine' or 'cosine'. ")

    return all_time_features


def construct_lagged_features(data: pd.Series, lags: list):
    """
    Lag input feature. Output columns have names like "input_col_5_step_lag" etc.

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    lags : list of ints

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Series object must have a 'name' attribute")

    for lag in lags:
        if not isinstance(lag, int) or lag == 0:
            raise ValueError(f"Invalid lag {lag}. Must be nonzero integer")

        shifted = data.shift(lag)
        if lag > 0:
            shifted.name = data.name + "_" + str(lag) + "_step_lag"
        elif lag < 0:
            shifted.name = data.name + "_" + str(lag) + "_step_fwd_lag"
        lst_series.append(shifted)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def construct_panel_features():
    """
    """
    # @TODO implement this
    raise NotImplementedError
