"""
feature_engineering.py

> Time-series features like rolling averages, standard-deviations, z-scores
"""
import pandas as pd


def construct_rolling_features(data, windows):
    """
    Get rolling mean, std-dev, max over various windows
    Output columns have names like "input_col_5_day_mean" etc.
    @TODO: "day" is hard-coded into column names, make resolution a parameter

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Series object must have a 'name' attribute")

    features = ["mean", "std", "max"]

    for window in windows:
        rolling = data.rolling(window=window)
        for feature in features:
            result = getattr(rolling, feature)()
            result.name = data.name + "_" + str(window) + "_day_" + feature
            lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def construct_ewm_features(data, windows):
    """
    Get exponential weighted moving mean and std-dev over various windows
    Output columns have names like "input_col_5_day_ewm_mean" etc.
    @TODO: "day" is hard-coded into column names, make resolution a parameter

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Series object must have a 'name' attribute")

    features = ["mean", "std"]

    for window in windows:
        ewm = data.ewm(span=window)
        for feature in features:
            result = getattr(ewm, feature)()
            result.name = data.name + "_" + str(window) + "_day_ewm" + feature
            lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df


def ewmzscore(x, window):
    """
    Compute exponentially weighted moving z-score

    Parameters
    ----------
    x : pandas.Series
        Input data
    window : int
        Span parameter for EWM.

    Returns
    -------
    pandas.Series
    """
    r = x.ewm(span=window)
    mu = r.mean()
    sigma = r.std()
    return (x - mu) / sigma


def construct_ewmzscore_features(data, windows):
    """
    Get exponential weighted moving z-score over various windows
    Output columns have names like "input_col_5_day_zscore" etc.
    @TODO: "day" is hard-coded into column names, make resolution a parameter

    Parameters
    ----------
    data : pandas.Series
        Input data. Must be a time-series with a "name" attribute
    windows : list of ints
        Which windows to calculate over.

    Returns
    -------
    pandas.Dataframe

    """
    lst_series = []

    if data.name is None:
        raise ValueError("Input Series must have a 'name' attribute")

    for window in windows:
        result = ewmzscore(data, window)
        result.name = data.name + "_" + str(window) + "_day_ewmzscore"
        lst_series.append(result)

    results_df = pd.concat(lst_series, axis=1)
    return results_df
