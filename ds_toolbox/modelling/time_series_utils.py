"""
time_series_utils.py

> helper functions for time-series data
 @todo: add line to check input checking all rows are numeric (no sum at end)
 @todo: fix tick_freq binding issue in candlesticks, 5 minute candlestick widths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import cm
from pandas.tseries.frequencies import to_offset
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller
from ds_toolbox.visualisation import PLOT_COLORS
from pandas.tseries.offsets import BDay
from typing import Union, Optional


register_matplotlib_converters()


def _check_input(df: pd.DataFrame):
    """
    Check time-series input DataFrame is correct format
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input data must be a pandas Dataframe not a {type(df)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except ValueError:
            print(f"Failed to convert Dataframe index of type {type(df.index)} to a pandas DatetimeIndex")


def _plot_check_input(df, limit_plots: Optional[bool] = True):
    """
    Check time-series input DataFrame is correct format for plotting
    """
    _check_input(df)

    max_plots = len(PLOT_COLORS)
    if len(df.columns) > max_plots and limit_plots:
        raise ValueError(f"requested number of plots {len(df.columns)} exceeds maximum {max_plots}")


def standardize(df):
    """
    Scale original time-series to zero-mean and unit variance.
    """
    return (df - df.mean()) / df.std()


def normalize(df):
    """
    Scale original time-series to range [0,1]
    """
    return (df - df.min()) / (df.max() - df.min())


def plot(df_in, normalized=False, standardized=False, start_date=None, end_date=None, tick_freq=None, tick_fmt=None,
         ax=None, labels: list = None, filled=False, **kwargs):
    """
    Plot one or more time-series organized as columns in a pandas.DataFrame with a datetime index.

    Parameters
    ----------
    df_in : pandas.DataFrame
        Can also plot a subset of columns with df[column_names_list]
    normalized : bool
        min-max scale the time-series to between [0, 1]
    standardized : bool
        standard scale the time series to zero mean, unit variance
    start_date, end_date : str
        Optional. Format "%Y-%m-%d"
    tick_freq : int or str, optional
        Datetime x-axis tick interval.
        Either a string such as "2Y" for every 2 years, or an int "n" for every n points in the data resolution.
        See _interpret_tick_freq() for full documentation.
    tick_fmt : str, optional
        String format of x-axis ticks. Defaults to %Y-%m-%d but can be any valid strftime specifier.
        See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    ax: matplotlib AxesSubplot, optional
        Plot the figure on an existing Axes instead of creating a new one.
    labels : list of strs, optional
        Use these labels instead of column names
    filled : bool, default False
        Stacked area chart instead of lines
    **kwargs
        Other plotting options for matplotlib
        Valid arguments are: cmap - named color palette e.g. viridis (see matplotlib for list),
                             style - e.g. "-o" for line and dot
                             color - any valid color or list of colors (must be same length as number of columns)
                             figsize - tuple, defaults to (15, 6)
                             dpi - int, figure resolution, defaults to 72
                             linewidth - int, defaults to 2
                             alpha - default 1
                             title - str, default is None

    Returns
    -------
    a matplotlib.figure.Figure object if retplot is True
    """
    df = df_in.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # do not limit number of plots if colours are specified, will check if the numbers match up later
    if kwargs.get("cmap") or kwargs.get("color"):
        limit_plots = False
    else:
        limit_plots = True
    _plot_check_input(df, limit_plots)

    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        assert len(labels) == len(df.columns), f"Length mismatch: {len(labels)} labels for {len(df.columns)} columns."
        df.columns = labels

    if normalized:
        df = normalize(df)
    if standardized:
        df = standardize(df)

    # specify plot colors
    if kwargs.get("cmap"):
        if kwargs.get("color"):
            raise ValueError("Cannot specify both cmap and color")
        cmap = cm.get_cmap(kwargs["cmap"], len(df.columns))
        color = cmap(range(len(df.columns)))
    elif kwargs.get("color"):
        color = kwargs["color"]
        if isinstance(color, list) and len(color) != len(df.columns):
            raise ValueError(f"Number of colours requested does not match number of columns to plot.")
    else:
        color = PLOT_COLORS[: len(df.columns)]

    if ax is None:
        fig, ax = plt.subplots(dpi=kwargs.get("dpi"))
        figsize = kwargs.get("figsize", (15, 6))
    else:
        figsize = (ax.figure.get_figwidth(), ax.figure.get_figheight())
    linewidth = kwargs.get("linewidth", 2)
    alpha = kwargs.get("alpha")
    if filled:
        df.plot.area(stacked=True, ax=ax, color=color, figsize=figsize, lw=linewidth, x_compat=True, alpha=alpha)
    else:
        # for a line plot, only plot individual data points if fewer than 100, otherwise looks cluttered
        if "style" not in kwargs:
            if len(df) > 100:
                linestyle = "-"
            else:
                linestyle = "-o"
        else:
            linestyle = kwargs["style"]
        df.plot(style=linestyle, ax=ax, color=color, figsize=figsize, lw=linewidth, x_compat=True, alpha=alpha)

    if tick_fmt is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(tick_fmt))

    if tick_freq is not None:
        ticks = _interpret_tick_freq(df, tick_freq)
        ax.xaxis.set_major_locator(ticks)

    ax.set_title(kwargs.get("title"), None)

    ax.xaxis.grid(True, which='major', linestyle=':')
    ax.yaxis.grid(True, which='major', linestyle=':')


def double_yaxis_plot(ts1, ts2, start_date=None, end_date=None, **kwargs):
    """
    Plot two time-series with different scales using two y-axes.
    Alternative to ts.plot(normalize=True) for special case of two time-series.

    @TODO: add labels, colors, style, legend_loc parameters, DataFrame options

    Parameters
    ----------
    ts1, ts2 : pandas.Series
        Input time-series. Requires a pandas.DatetimeIndex index.
    start_date, end_date : str, optional
        Format "YYYY-MM-DD"
    **kwargs

    """

    if start_date is not None:
        ts1 = ts1[ts1.index >= start_date]
        ts2 = ts2[ts2.index >= start_date]
    if end_date is not None:
        ts1 = ts1[ts1.index <= end_date]
        ts2 = ts2[ts2.index <= end_date]

    if "style" not in kwargs:
        linestyle = "-"  # "-o"
    else:
        linestyle = kwargs["style"]

    fig, ax1 = plt.subplots()
    ts1.plot(style=linestyle, ax=ax1, color='red', figsize=(15, 6), x_compat=True)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ts2.plot(style=linestyle, ax=ax2, color='dodgerblue', figsize=(15, 6), x_compat=True, label=ts2.name)
    ax2.tick_params(axis='y', labelcolor='dodgerblue')
    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

    ax1.xaxis.grid(True, which='major', linestyle=':')
    ax1.yaxis.grid(True, which='major', linestyle=':')


def plot_candlesticks(data_in, start_date=None, end_date=None, tick_freq=None, tick_fmt=None, retplot=False, **kwargs):
    """
    Candlestick finance plot. Only a single time-series can be provided at a time.

    Parameters
    ----------
    data_in : pandas.Dataframe
        Single time-series converted to OHLC format (see DataFrame.resample().ohlc())
    start_date, end_date : str
        Optional. Format "YYYY-MM-DD"
    tick_freq : int or str, optional
        Datetime x-axis tick interval.
        Either a string such as "2Y" for every 2 years, or an int "n" for every n points in the data resolution.
        See _interpret_tick_freq() for full documentation.
    tick_fmt : str, optional
        String format of x-axis ticks. Defaults to %Y-%m-%d but can be any valid strftime specifier.
        See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    retplot : bool
        return an axis object from the function call as well as plot it
    **kwargs
        Options for matplotlib. Implemented 'title', 'xlabel', 'ylabel', 'figsize', 'dpi'.
        See time_series_utils.plot() documentation.

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

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (15, 6)), dpi=kwargs.get("dpi"))

    # pyplot bar width units are always in days, need to convert width to frequency of data
    freq_str = candle_data.index.freqstr
    if freq_str is None:
        freq_str = infer_freq(candle_data)

    # business days have the same width as calendar days
    if freq_str == "B":
        freq_str = "D"
    divisor = pd.Timedelta(1, "D") / pd.to_timedelta(to_offset(freq_str))

    # positive returns in green
    plt.bar(up.index, up.close - up.open, width=1 / divisor, bottom=up.open, color='g')
    plt.bar(up.index, up.high - up.close, width=0.2 / divisor, bottom=up.close, color='g')
    plt.bar(up.index, up.low - up.open, width=0.2 / divisor, bottom=up.open, color='g')

    # negative returns in red
    plt.bar(down.index, down.close - down.open, width=1 / divisor, bottom=down.open, color='r')
    plt.bar(down.index, down.high - down.open, width=0.2 / divisor, bottom=down.open, color='r')
    plt.bar(down.index, down.low - down.close, width=0.2 / divisor, bottom=down.close, color='r')

    if tick_freq is not None:
        ticks = _interpret_tick_freq(candle_data, tick_freq)
        ax.xaxis.set_major_locator(ticks)
    plt.xticks(rotation=45)

    if tick_fmt is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(tick_fmt))

    ax.set_title(kwargs.get("title"), fontsize=kwargs.get("fontsize"))
    ax.set_xlabel(kwargs.get("xlabel"), fontsize=kwargs.get("fontsize"))
    ax.set_ylabel(kwargs.get("ylabel"), fontsize=kwargs.get("fontsize"))

    plt.grid(linestyle=":")

    if retplot:
        return fig


def get_crosscorr(datax: pd.Series, datay: pd.Series, start_date=None, end_date=None, lag=0):
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

    mask: Union[pd.Series, pd.DataFrame] = data.pct_change(periods=period).abs() > threshold

    if isinstance(data, pd.DataFrame):
        if how is None:
            raise ValueError("If data is a DataFrame, need to specify how=\"any\" or \"all\". ")
        elif how == "any":
            mask = mask.any(axis=1)
        elif how == "all":
            mask = mask.all(axis=1)

    idx = data[mask].index
    return idx


def generate_random_walk(start_datetime="2001-01-01", start_y=0,
                         n_obs=1000, freq="D", step="gaussian", random_state=None):
    """
    Generate a random walk time-series with the specified settings
    @TODO: Add drift term, drift_up=in range [0,1], how to implement for Gaussian: shift mean up by drift_up - 0.5

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
        step_list = np.random.choice([step, -step], size=n_obs-1)
    elif isinstance(step, list):
        step_list = np.random.choice(step, size=n_obs-1)
    elif step == "gaussian":
        step_list = np.random.randn(n_obs-1)
    else:
        raise TypeError(f"step must be an int, float, list of ints/floats, or \"gaussian\", not a {type(step)} ")

    x_range = pd.date_range(start_datetime, periods=n_obs, freq=freq)
    y_range = np.concatenate(([start_y], step_list), axis=0).cumsum()

    return pd.DataFrame(y_range, index=x_range, columns=["random_walk"])


def yesterday(output_fmt="str"):
    """
    Last business day before today.

    Parameters
    ----------
    output_fmt : str
        Either "str" or "datetime" for desired output format

    Returns
    -------
    str in format %Y-%m-%d or pd.datetime object
    """
    yday = pd.to_datetime('today') - BDay(1)

    if output_fmt == "str":
        return yday.strftime("%Y-%m-%d")
    elif output_fmt == "datetime":
        return yday
    else:
        raise ValueError("output_fmt must be either \"str\" or \"datetime\"")


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
    @TODO: Add output="short/long" option displaying dict for latter.

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


def _interpret_tick_freq(df_in, tick_freq):
    """
    Handle tick_freq input to plot().

    If an integer is provided, give ticks in equal-spaced intervals
    of size tick_freq in the same frequency as the input data.
    If a string is provided, such as "2M", use ticks in that frequency directly.

    If data has weekly frequency and tick_freq = 4, plot every 4 weeks.
    If tick_freq = "2W", plot ticks every 2 weeks, irrespective of frequency of data.

    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Parameters
    ----------
    df_in : pandas.Dataframe
        Input data
    tick_freq: int or str
        Frequency of ticks.
    """

    if isinstance(tick_freq, int):
        inferred_freq = infer_freq(df_in)
        ticks = _get_ticks_from_str(freq_str=inferred_freq, interval=tick_freq)

    elif isinstance(tick_freq, str):
        ticks = _get_ticks_from_str(freq_str=tick_freq)
    else:
        raise TypeError(f"tick_freq must be a string or int not a {type(tick_freq)}")
    return ticks


def _get_ticks_from_str(freq_str=None, interval=None):
    """
    Infer valid datetime axis ticks for plotting using the provided arguments.
    There are multiple ways of expressing the same ticks, e.g.

    _get_ticks("W-SUN", interval=2)
    _get_ticks("2W-SUN")

    """

    # "W-SUN", "Q-FEB" etc.
    anchor = None
    if "W-" in freq_str:
        inferred_freq_str, anchor = freq_str.split("-")
        print(inferred_freq_str)

    # "6M" etc.
    elif any(char.isdigit() for char in freq_str):
        inferred_freq_str = freq_str[-1]
        interval = int(freq_str[:-1])
    else:
        inferred_freq_str = freq_str

    # if interval not provided, assume 1
    if interval is None:
        interval = 1

    # use Sunday as start of week if not specified
    if anchor is None:
        anchor = "SUN"

    dict_weekdays = {"SUN": mdates.SU,
                     "MON": mdates.MO,
                     "TUE": mdates.TU,
                     "WED": mdates.WE,
                     "THU": mdates.TH,
                     "FRI": mdates.FR,
                     "SAT": mdates.SA}

    tick_dict = {"s": mdates.SecondLocator(interval=interval),
                 "T": mdates.MinuteLocator(interval=interval),
                 "H": mdates.HourLocator(interval=interval),
                 "D": mdates.DayLocator(interval=interval),
                 "W": mdates.WeekdayLocator(byweekday=dict_weekdays[anchor], interval=interval),
                 "M": mdates.MonthLocator(interval=interval),
                 "Y": mdates.YearLocator(interval)}

    # put business days on weekday ticks
    if inferred_freq_str == "B":
        inferred_freq_str = "D"
    try:
        inferred_ticks = tick_dict[inferred_freq_str]
    except KeyError:
        print(f"No supported tick frequency for this data. Obtained {inferred_freq_str}.")
        raise
    return inferred_ticks


def infer_freq(df_in):
    """
    Infer the frequency of a time-series, either using pandas directly or from the first two timestamps.
    Dataframe must have a pandas.DatetimeIndex index or a string index that can be converted to one.

    Parameters
    ----------
    df_in : pandas.Dataframe

    Returns
    -------
    A string, e.g. "6M", "5T", "W-SUN"
    """
    # first try to get frequency of time-index directly
    inferred_freq_str = df_in.index.freqstr

    # failing that, infer it manually @todo: this is still buggy
    if inferred_freq_str is None:
        inferred_freq_str = df_in.index.to_series().diff().iloc[1].resolution_string

    return inferred_freq_str
