"""
visualisation.py

> Plotting functions with matplotlib and bokeh. The aim is to make this more interactive.
@todo: fix bug where hover tooltip shows multiple date index values at same x
@todo: fix tick fmt in ts.plot()

"""
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text

from sklearn.metrics import r2_score
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import all_palettes

plot_colors = ["red", "dodgerblue", "forestgreen", "gold", "magenta", "turquoise", "darkorange", "darkviolet",
               "firebrick", "navy", "lime", "goldenrod", "mediumpurple", "royalblue", "orange", "violet",
               "springgreen", "sandybrown", "aquamarine", "skyblue", "salmon", "chartreuse", "indigo"]


def rgb2hex(*args):
    """
    Get a hexadecimal color code from its RGB values

    Parameters
    ----------
    args: 3 integers corresponding to RGB values

    Returns
    -------
    str
    """
    assert len(args) == 3, f"Expected exactly 3 args, got {len(args)}"
    for arg in args:
        assert arg in range(256), f"{arg} is invalid. Expected integer in range 0-255"
    return '#%02x%02x%02x' % args


def barplot(df_in, normed=False, retplot=False, **kwargs):
    """
    Bar chart of a pandas.Series with extra options

    Parameters
    ----------
    df_in : pandas.Series
        data
    normed : bool, default False
        Sum heights of bars to 1 rather than total counts
    retplot: bool, default False
        Return the matplotlib.Figure instance
    **kwargs
        Options for matplotlib, such as "xlabel", "ylabel", "title", "figsize", "palette"
    """
    data = df_in.copy()
    if normed:
        data = data / data.sum()
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (15, 6)))

    sns.barplot(x=data.index, y=data, palette=kwargs.get("palette", "viridis"), ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel(kwargs.get("xlabel"))
    plt.ylabel(kwargs.get("ylabel"))
    plt.title(kwargs.get("title"))
    if retplot:
        return fig


def countplot_sns(df_in, col_name, normed=True, **kwargs):
    """
    Seaborn countplot of value counts for a categorical column

    Parameters
    ----------
    df_in : pandas.Dataframe
        data
    col_name : str or list of strs
        Which column(s) to plot
    normed : bool
        Sum heights of bars to 1 rather than total counts
    **kwargs
        Options for seaborn.countplot()

    """
    vc = df_in[col_name].value_counts(normalize=normed).sort_index()
    plt.figure(figsize=(15, 6))
    sns.countplot(x=col_name, data=df_in, order=vc.index, **kwargs)
    plt.xticks(rotation=45)
    plt.xlabel(kwargs.get("xlabel"))
    plt.ylabel(kwargs.get("ylabel"))
    plt.title(kwargs.get("title"))
    plt.show()


def corrplot(data, method="pearson", retplot=False, **kwargs):
    """
    Plot correlations between Dataframe columns in a heatmap.

    Parameters
    ----------
    data : pandas.Dataframe
        Input data. Remember time-series should be stationary
    method : str or callable
        See Dataframe.corr() for options
    retplot: bool, default False
        Return the figure object as well as plot it.
    **kwargs
        Plotting options: figsize (tuple of ints), cmap (str)

    """
    figsize = kwargs.get("figsize", (10, 8))
    cmap = kwargs.get("cmap", "viridis")

    corrs = data.corr(method=method)

    fig = plt.figure(figsize=figsize)
    sns.heatmap(corrs, cmap=cmap)
    plt.xticks(rotation=45, ha="right")
    if retplot:
        return fig


def plot_r2(y_pred, y_true, **kwargs):
    """
    Scatter plot of y_predicted vs. actuals and associated R^2 score.

    Parameters
    ----------
    y_pred, y_true : numpy.array
        predicted and actual values of target
    **kwargs
        Options for pyplot.scatter(), such as title
    """

    assert len(y_pred) == len(y_true), "Input arrays must be same shape"

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(kwargs["title"])
    ax.scatter(x=y_pred, y=y_true, marker='o', color='springgreen', label='')

    # plot y = x to show deviations
    ax.plot(range(np.ceil(max(y_true)) + 1), color='firebrick')

    limits = min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true))
    plt.xlim((limits[0], limits[1]))
    plt.ylim((limits[0], limits[1]))
    plt.xlabel('Predicted value')
    plt.ylabel('True value')

    r2 = r2_score(y_pred, y_true)
    # r2 text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, 'R$^2$ = {:.3f}'.format(r2), fontsize=14, transform=ax.transAxes, verticalalignment='top',
            bbox=props)

    plt.grid(b=True, color='grey', linestyle=':', linewidth=0.5)


def multiscatter(X, labels, max_plots=None, keep_labels=None, **kwargs):
    """
    Plot 2D scatter data with label information.
    @TODO: Add color, cbar kwarg

    Parameters
    ----------
    X : array-like
        Input data of shape (n_samples, 2)
    labels : array-like
        List of labels for data points of shape (n_samples, )
    keep_labels : array-like, optional
        Only plot this subset of labels.
    max_plots : int, optional
        Only plot n most frequent labels
    **kwargs
        Matplotlib options, e.g. figsize, xlim, ylim, title, xlabel, ylabel
    """

    if X.shape[1] != 2:
        raise ValueError("Scatter data must have two dimensions")

    assert X.shape[0] == len(labels), f"Mismatch in length of input data {X.shape[0]} and labels {len(labels)}"

    # sort labels from most frequent to least
    labels_uniq, counts = np.unique(labels, return_counts=True)
    idx_sorted_by_counts = np.argsort(-counts)
    labels_uniq = labels_uniq[idx_sorted_by_counts]

    plt.figure(figsize=kwargs.get("figsize"))
    for i, label in enumerate(labels_uniq[:max_plots]):
        if keep_labels is not None and label not in keep_labels:
            continue
        plot_data = X[(labels == label)]
        plt.scatter(plot_data[:, 0], plot_data[:, 1], color=plot_colors[i], label=label)

    plt.xlim(kwargs.get("xlim"))
    plt.ylim(kwargs.get("ylim"))
    plt.xlabel(kwargs.get("xlabel"))
    plt.ylabel(kwargs.get("ylabel"))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(kwargs.get("title"))
    plt.show()


def plot_missingness(data_in, start_date=None, end_date=None, tick_freq=None, tick_fmt=None, **kwargs):
    """
    Plot missing periods in black, filled periods in white for time-series Dataframe.

    @TODO seaborn heatmap is still a bit buggy. Do this with matplotlib barcode instead.

    Parameters
    ----------
    data_in : pandas.Dataframe
        Input time-series data with a pandas.DatetimeIndex index.
    start_date, end_date : str, optional
        Set plotting range.
    tick_freq : int, optional
        Frequency of x-axis ticks in units of data index.
    tick_fmt: str, optional
        Date format of x-axis ticks
    **kwargs
        Options for matplotlib
    """
    plt.figure(figsize=(15, 6))

    data = data_in.copy()
    if start_date is not None:
        data = data[data.index >= start_date]
    if end_date is not None:
        data = data[data.index <= end_date]
    if tick_fmt is not None:
        data.index = data.index.strftime(tick_fmt)

    if tick_freq is not None:
        xticks = np.arange(0, len(data) - 1, tick_freq, dtype=int)
        xticklabels = data.index[xticks]
        sns.heatmap(data.T.isnull(), cmap=sns.cm.rocket_r, cbar=False)
        plt.xticks(xticks, labels=xticklabels, rotation=45, ha="right")
    else:
        sns.heatmap(data.T.isnull(), cmap=sns.cm.rocket_r, cbar=False)
        plt.xticks(rotation=45, ha="right")

    plt.xlabel(kwargs.get("xlabel"))
    plt.title(kwargs.get("title"))


def bokeh_chart(timeseries_df, normalized=False, legend_labels: list = None,
                linestyle="line", palette=None, colors=None, title=None):
    """
    Plot time-series in an interactive Bokeh chart
    @TODO: Add various plotting options, legend location, tick_freq
    @TODO: Move all this into visualisation.bokeh?

    Parameters
    ----------
    timeseries_df : pandas.DataFrame
        Must have a pandas.DatetimeIndex index otherwise it will not plot
    normalized : bool, default False
        Scale all time-series between 0 and 1.
    legend_labels: list of strs, optional
        Custom plot legend entries, default is column headers
    linestyle : str, default "line"
        Either "line", "linescatter" or "scatter"
    palette : str, optional
        Use a custom colour palette. See https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    colors : list of strs, optional
        Explicitly pass a list of colors. Otherwise random colors or a named palette are used
    title : str, optional
        Plot title

    Returns
    -------
    A bokeh Figure object, use show(fig) to display it. Remember to run output_notebook() to display in jupyter.
    """
    valid_linestyles = ["line", "linescatter", "scatter"]
    if linestyle not in valid_linestyles:
        raise ValueError(f"linestyle {linestyle} not valid. Must be one of {valid_linestyles}")

    if timeseries_df.index.name is None:
        raise ValueError("timeseries index must have a name")
    xlabel = timeseries_df.index.name

    if normalized:
        timeseries_df = (timeseries_df - timeseries_df.min()) / (timeseries_df.max() - timeseries_df.min())
    source = ColumnDataSource(timeseries_df)

    num_series = len(timeseries_df.columns)

    # set plot color options: either named palette, user list of colors, or randomly chosen
    if palette is not None:
        if colors is not None:
            raise ValueError("Using both 'palette' and 'colors' is incompatible. Choose one or the other.")
        if palette not in all_palettes:
            raise ValueError(f"Invalid palette {palette}. See bokeh docs for a list")
        if num_series not in range(3, 12):
            raise ValueError(f"Can only use named palettes if number of plots between 3 and 11.")  # @TODO: improve this
        used_colors = all_palettes[palette][num_series]
    elif colors is not None:
        used_colors = colors
    else:
        shuffled_colors = plot_colors.copy()
        np.random.shuffle(shuffled_colors)
        used_colors = shuffled_colors

    p = figure(x_axis_type="datetime", plot_width=950, plot_height=400, outline_line_color='black', title=title)

    if legend_labels is not None and len(legend_labels) != num_series:
        raise ValueError(f"Length mismatch: {len(legend_labels)} labels for {num_series} columns")

    for i, col in enumerate(timeseries_df.columns):

        if legend_labels is None:
            label = col
        else:
            label = legend_labels[i]

        color = used_colors[i]
        if "line" in linestyle:
            p.line(x=xlabel, y=col, source=source, line_width=2, color=color,
                   legend_label=label, muted_color=color, muted_alpha=0.2)
        if "scatter" in linestyle:
            p.circle(x=xlabel, y=col, legend_label=label, source=source,
                     color=color, muted_color=color, muted_alpha=0)

    p.legend.click_policy = "mute"
    p.xaxis.axis_label = 'Datetime'

    hover = HoverTool(tooltips=[("Date", "$x{%F}"), ("Value", "$y")], formatters={"$x": "datetime"})
    p.add_tools(hover)

    return p


def bokeh_candlestick(timeseries_df, **kwargs):
    """
    Interactive candlestick chart with Bokeh.
    Expects a dataframe with exactly 4 columns: ["open", "high", "low", "close"] (case-insensitive)

    Parameters
    ----------
    timeseries_df : pandas.DataFrame
        Must have a pandas.DatetimeIndex index otherwise it will not plot
    **kwargs
        Only 'title' so far.

    Returns
    -------
    A bokeh Figure object, use show(fig) to display it. Remember to run output_notebook() to display in jupyter.
    """

    df = timeseries_df.copy()
    df.columns = df.columns.str.lower()
    expected_columns = ["close", "high", "low", "open"]

    if sorted(df.columns.tolist()) != expected_columns:
        raise ValueError(f"Expected exactly 4 columns: {expected_columns}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    inc = df.close > df.open
    dec = df.open > df.close

    # bokeh datetime axes are in ms units so bar widths need to be in ms
    resolution_seconds = abs((df.index[1] - df.index[0]).total_seconds())
    w = 0.75 * resolution_seconds * 1000  # x0.75 to keep a small gap between bars

    p = figure(x_axis_type="datetime", plot_width=950, plot_height=400, title=kwargs.get("title"),
               outline_line_color="black")
    #     p.xaxis.major_label_orientation = pi/4

    p.segment(df.index, df.high, df.index, df.low, color="black")
    p.vbar(df.index[inc], w, df.open[inc], df.close[inc], fill_color="forestgreen", line_color="black")
    p.vbar(df.index[dec], w, df.open[dec], df.close[dec], fill_color="red", line_color="black")

    hover = HoverTool(tooltips=[("Date", "$x{%F}"), ("Value", "$y")], formatters={"$x": "datetime"})
    p.add_tools(hover)

    return p


def customise_fonts(mpl_plot, font_name, font_size, font_folder=None):
    """
    Modify the fonts on a pre-drawn matplotlib Figure to a custom font stored in the ~/Library/Fonts folder.
    Note, matplotlib figures cannot be copied, so pickle the original if you want to save it before updating.

    Parameters
    ----------
    mpl_plot : Figure, AxesSubplot or list of AxesSubplots
        The figure to customise.
    font_name : str
        The name of the ttf file in the ~/Library/Fonts, i.e. <font_name>.ttf
    font_size: int
        Making this a required argument as different fonts scale in size differently.
    font_folder: str, optional
        Folder containing the font ttf file. Defaults to ~/Library/Fonts as on a Mac.

    Returns
    -------
    matplotlib.figure.Figure object
    """
    if isinstance(mpl_plot, Figure):
        axes = mpl_plot.axes
    elif isinstance(mpl_plot, plt.Axes):
        axes = [mpl_plot]
    elif isinstance(mpl_plot, list) and all(isinstance(ax, plt.Axes) for ax in mpl_plot):
        axes = mpl_plot
    else:
        raise TypeError(f"Expected a Figure, pyplot.Axes or a list of pyplot.Axes, not a {type(mpl_plot)}.")

    if font_folder is None:
        font_file_path = f"{os.environ['HOME']}/Library/Fonts/{font_name}.ttf"  # default Mac locations
    else:
        font_file_path = os.path.join(font_folder, f"{font_name}.ttf")
    prop = fm.FontProperties(fname=font_file_path, size=font_size)

    # unicode minus signs are not rendered in certain fonts, use hyphen instead
    # https://stackoverflow.com/questions/58361594/
    plt.rcParams['axes.unicode_minus'] = False

    # modify all the subplot texts @TODO: Could this be done in one loop over ax.get_children()?
    for ax in axes:

        ax.set_title(ax.get_title(), fontproperties=prop)
        ax.set_xlabel(ax.get_xlabel(), fontproperties=prop)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=prop)

        # in case of scientific notation '1e6' etc.
        ax.yaxis.get_offset_text().set_font_properties(prop)

        for label in ax.get_xticklabels():
            label.set_fontproperties(prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(prop)

        if ax.get_legend() is not None:

            # changing the legend font resets its position so store the old position and update it later
            loc = ax.legend_._loc
            loc = [k for k, v in ax.legend_.codes.items() if v == loc][0]  # convert from '0' to 'upper left' etc.
            try:
                bbox = ax.legend_._bbox_to_anchor._bbox
            except AttributeError:  # no bbox defined
                bbox = None
            ax.legend(prop=prop, loc=loc, bbox_to_anchor=bbox)

        # if there are text annotations on the plot
        for child in ax.get_children():
            if isinstance(child, Annotation) or isinstance(child, Text):
                child.set_font_properties(prop)

    # finally, if fig.suptitle() was used, update it. Annoyingly it is a protected attribute so triggers a warning.
    if hasattr(mpl_plot, "_suptitle"):
        if mpl_plot._suptitle is not None:
            st = mpl_plot._suptitle
            x, y = st.get_position()
            text = st.get_text()
            mpl_plot.suptitle(x=x, y=y, t=text, fontproperties=prop)

    return mpl_plot
