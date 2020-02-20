"""
time_series_utils.py

Helper functions for time-series data
 @todo: add line to check input checking all rows are numeric (no sum at end)
 @todo: fix tick_freq binding issue in candlesticks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.frequencies import to_offset

from statsmodels.tsa.stattools import adfuller
from ..visualisation import plot_colors


def _check_input(df):
    """
    Check time-series input DataFrame is correct format
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except ValueError:
            print(f"Failed to convert Dataframe index of type {type(df.index)} to a pandas DatetimeIndex")


def _plot_check_input(df):
    """
    Check time-series input DataFrame is correct format for plotting
    """
    _check_input(df)

    max_plots = len(plot_colors)
    if len(df.columns) > max_plots:
        raise ValueError(f"requested number of plots {len(df.columns)} exceeds maximum {max_plots}")
        

def standardize(df):
    """
    Scale original time-series to zero-mean and unit variance.
    """
    _check_input(df)
    return (df - df.mean()) / df.std()


def normalize(df):
    """
    Scale original time-series to range [0,1]
    """
    _check_input(df)
    return (df - df.min()) / (df.max() - df.min())


def plot(df, normalized=False, standardized=False, start_date=None, end_date=None, **kwargs):
    """
    Plot one or more time-series organized as columns in a pandas.DataFrame with a datetime index.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Can also plot a subset of columns with df[column_names_list]
    normalized : bool
        min-max scale the time-series to between [0, 1]
    standardized : bool
        standard scale the time series to zero mean, unit variance
    start_date, end_date : str
        Optional. Format "YYYY-MM-DD"
    **kwargs :
        Valid arguments are: title (str), tick_freq (int)
        
    Returns
    -------
    a matplotlib.figure.Figure object
    """
    _plot_check_input(df)

    if normalized:
        df = normalize(df)
    if standardized:
        df = standardize(df)

    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    # only plot ticks if fewer than 100 points shown, otherwise looks cluttered
    if "style" not in kwargs:
        if len(df) > 100:
            linestyle = "-"
        else:
            linestyle = "-o"
    else:
        linestyle = kwargs["style"]

    if kwargs.get("color"):
        color = kwargs["color"]
    else:
        color = plot_colors[: len(df.columns)]

    fig, ax = plt.subplots()
    df.plot(style=linestyle, ax=ax, color=color, figsize=(15, 6), x_compat=True)

    if kwargs.get("title"):
        ax.title(kwargs["title"])

    if kwargs.get("tick_freq"):
        ticks = _get_plot_ticks(df, kwargs["tick_freq"])
        ax.xaxis.set_major_locator(ticks)

    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')


def _get_plot_ticks(data_df, tick_freq):
    """
    Get appropriate ticks for plot based on inferred frequency.
    Matplotlib/pandas does not do this as well as it should.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Input data
    tick_freq : int
        Frequency of tick marks in units of input data resolution

    Returns
    -------
    A matplotlib.dates Locator object, e.g HourLocator
    For use with an axes plot object with ax.xaxis.set_major_locator(ticks)
    """

    tick_dict = {"T": mdates.MinuteLocator(interval=tick_freq),
                 "H": mdates.HourLocator(interval=tick_freq),
                 "D": mdates.DayLocator(interval=tick_freq),
                 "M": mdates.MonthLocator(interval=tick_freq),
                 "Y": mdates.YearLocator(tick_freq)}

    inferred_freq_str = infer_freq(data_df).resolution_string

    try:
        inferred_ticks = tick_dict[inferred_freq_str]
    except KeyError:
        print(f"No supported tick frequency for this data. Obtained {inferred_freq_str}.")
        raise

    return inferred_ticks


def plot_candlesticks(data_in, start_date=None, end_date=None, **kwargs):
    """
    Candlestick finance plot. Only a single time-series can be provided at a time.

    Parameters
    ----------
    data_in : pandas.Dataframe
        Single time-series converted to OHLC format (see DataFrame.resample().ohlc())
        For a list see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    start_date, end_date : str
        Optional. Format "YYYY-MM-DD"
    **kwargs
        Options for matplotlib. Implemented 'title', 'xlabel', 'ylabel', 'tick_freq'

    Returns
    -------
    a matplotlib.figure.Figure object

    Raises
    ------
    ValueError
        If DataFrame does not have exactly 4 columns labelled "open", "high", "low", "close".
        Case-insensitive.
    """

    candle_data = data_in.copy()

    if candle_data.shape[1] != 4:
        raise ValueError(f"OHLC plot requires exactly 4 columns, {candle_data.shape[1]} provided")

    if start_date is not None:
        candle_data = candle_data[candle_data.index >= start_date]
    if end_date is not None:
        candle_data = candle_data[candle_data.index <= end_date]

    candle_data.columns = [x.lower() for x in candle_data.columns]

    up = candle_data[candle_data.close > candle_data.open]
    down = candle_data[candle_data.close < candle_data.open]

    plt.figure(figsize=(15, 6))

    # pyplot bar width units are always in days, need to convert width to frequency of data
    freq_str = candle_data.index.freqstr

    # business days have the same width as calendar days
    if freq_str == "B":
        freq_str = "D"
    divisor = pd.Timedelta(1, "D") / pd.to_timedelta(to_offset(freq_str))

    # positive returns in green
    plt.bar(up.index, up.close - up.open, 1 / divisor, bottom=up.open, color='g')
    plt.bar(up.index, up.high - up.close, 0.2 / divisor, bottom=up.close, color='g')
    plt.bar(up.index, up.low - up.open, 0.2 / divisor, bottom=up.open, color='g')

    # negative returns in red
    plt.bar(down.index, down.close - down.open, 1 / divisor, bottom=down.open, color='r')
    plt.bar(down.index, down.high - down.open, 0.2 / divisor, bottom=down.open, color='r')
    plt.bar(down.index, down.low - down.close, 0.2 / divisor, bottom=down.close, color='r')

    if kwargs.get("tick_freq"):
        plt.xticks(candle_data.index[::kwargs.get("tick_freq")], rotation=45)
    else:
        plt.xticks(rotation=45)
    plt.title(kwargs.get("title"), fontsize=kwargs.get("fontsize"))

    plt.xlabel(kwargs.get("xlabel"), fontsize=kwargs.get("fontsize"))
    plt.ylabel(kwargs.get("ylabel"), fontsize=kwargs.get("fontsize"))
    plt.grid(linestyle=":")


def get_crosscorr(datax, datay, start_date=None, end_date=None, lag=0):
    """ 
    Lag-N cross correlation. Based on pandas.Series.autocorr().
    Simple measure of causal effect of datay on datax,
    but remember https://en.wikipedia.org/wiki/Post_hoc_ergo_propter_hoc
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    start_date, end_date : str or pd.DatetimeIndex, optional
        Define period over which to calculate the correlation

    Returns
    -------
    crosscorr : float

    Example
    -------
    >>> datay = pd.Series(np.random.randn(10))
    >>> datax = datay.shift(1)
    >>> get_crosscorr(datax, datay, lag=1)
    1.0
    """
    if not (datax.index == datay.index).all():
        raise ValueError("datax and datay must have the same index")

    if start_date is not None:
        datax = datax[datax.index >= start_date]
        datay = datay[datay.index >= start_date]
    if end_date is not None:
        datax = datax[datax.index <= end_date]
        datay = datay[datay.index <= end_date]
        
    return datax.corr(datay.shift(lag))


def infer_freq(df_in):
    """
    Infer the frequency of a time-series from the first two timestamps.
    Dataframe must have a pandas.DatetimeIndex index or a string index that can be converted to one.

    Parameters
    ----------
    df_in : pandas.Dataframe

    Returns
    -------
    pandas.Timedelta object, e.g. Timedelta('0 days 00:05:00')
    Can be converted to a string like "5T" with infer_freq(...).resolution_string
    """
    _check_input(df_in)

    inferred_freq = df_in.index.to_series().diff().iloc[1]
    return inferred_freq


def convert_to_wide_fmt():
    pass


def sliding_windows(x, window, stride=1):
    """
    Generates sliding windows in the rows dimension, for each column of the matrix.
    Windows are overlapping in the rows dimension.

    Parameters
    ----------
    x : numpy.ndarray
        data of dimension (n,m) i.e. (num observations, num features)
    window : int
        length of sequences to be generated
    stride : int
        Gap between consecutive windows (default 1)

    Returns
    -------
    numpy.ndarray
        data of dimension (n-window, window, m)

    """
    sequence_lst = []
    # create all possible sequences of length seq_len
    for i in range(0, len(x) + 1 - window, stride):
        sequence_lst.append(x[i: i + window])
    return np.array(sequence_lst)


def get_volatility(df, start_date=None, end_date=None):
    """
    Get historical volatility i.e. standard-deviation of log-returns
    https://www.macroption.com/historical-volatility-calculation/

    Parameters
    ----------
    df : pandas.DataFrame
        Individual time-series as columns and pandas.DatetimeIndex as index.
    start_date, end_date: str or pandas.DatetimeIndex (optional)

    Returns
    -------
    pandas.Series of volatility for each column
    """
    _check_input(df)
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    return np.std(np.log(df / df.shift(1)))


def volatile_periods(data, period=1, threshold=0.1, how=None):
    """
    Get indices of time-series where fractional change in value over period was larger than threshold
    i.e. values of idx where abs(data.loc[idx] - data.loc[idx-period]) >= threshold * data.loc[idx-period]

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Single time-series with pandas.DatetimeIndex as index.
    period : int
        Number of ticks defining period of change
    threshold : float
        Minimum fractional change
    how : str
        If data is a Dataframe, set to "any" or "all" to filter rows if any or all columns have a large change.

    Returns
    -------
    pandas.index of same dtype as original series index
    """
    mask = data.pct_change(periods=period).abs() > threshold

    if isinstance(data, pd.DataFrame):
        if how is None:
            raise ValueError("If data is a DataFrame, need to specify how=\"any\" or \"all\". ")
        elif how == "any":
            mask = mask.any(axis=1)
        elif how == "all":
            mask = mask.all(axis=1)

    idx = data[mask].index
    return idx


def generate_random_walk(start_datetime, start_y, n_obs, freq="D", step="gaussian", random_state=None):
    """
    Generate a random walk time-series with the specified settings

    Parameters
    ----------
    start_datetime : str or pandas.Datetime object
        First timestamp
    start_y : float
        Initial value
    n_obs : int
       Number of timesteps
    freq : str (default "D")
        Frequency of time-series
    step : int, float, list of ints or floats, or "gaussian"
        if int or float, walk is [step, -step] with equal probs, can also be specified with a list with equal probs.
        If "gaussian", steps are drawn from a unit normal distribution.
    random_state : int (optional)
        Numpy random seed

    Returns
    -------
    pandas.DataFrame with one column and n_obs rows
    """
    if random_state is not None:
        np.random.seed(random_state)

    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(step, int) or isinstance(step, float):
        step_list = np.random.choice([step, -step], size=n_obs)
    elif isinstance(step, list):
        step_list = np.random.choice(step, size=n_obs)
    elif step == "gaussian":
        step_list = np.random.randn(n_obs)
    else:
        raise TypeError(f"step must be an int, float, list of ints/floats, or \"gaussian\", not a {type(step)} ")

    x_range = pd.date_range(start_datetime, periods=n_obs, freq=freq)

    y = start_y
    y_range = []

    for current_step in step_list:
        y_range.append(y)
        y += current_step
    return pd.DataFrame(y_range, index=x_range, columns=["random_walk"])


def drop_weekends(df_in):
    """
    Hopefully self-explanatory
    """
    df = df_in.copy()
    _check_input(df)
    df = df.loc[df.index.dayofweek < 5]
    return df


def is_stationary(data, **kwargs):
    """
    Simple test of stationarity of a time-series using augmented Dicky-Fuller test implemented in statsmodels

    Parameters
    ----------
    data : A pandas.Series object
    **kwargs
        Extra options. See https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    Returns
    -------
    Tuple of bool of time-series stationarity and corresponding p-value

    """
    adf, pval, used_lag, n_obs, dict_crit, icbest = adfuller(data, **kwargs)
    return pval < 0.05, pval
