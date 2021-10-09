"""
pandas_utils.py

> useful helper functions for pandas Dataframes
@todo: clean up repeated renamed_values code in map_to_signs()
@todo: read_and_stack_csvs should accept relative paths as well
@todo: typing hints for all functions
"""
import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import t

from typing import List, Optional, Union


def count_nans(df_in, header=None, sort=False):
    """
    Get number of nans in each column

    Parameters
    ----------
    df_in : pandas DataFrame or Series
    header : str (optional)
        Name of the output DataFrame. Default is "nan_counts"
    sort : bool, default False
        Sort columns from fewest nans to most nans.

    Returns
    -------
    pandas.Dataframe with original columns as index and number of nans in each column as value
    """
    if header is None:
        header = "nan_counts"
    nan_counts = df_in.isnull().sum().to_frame(header)
    if sort:
        nan_counts = nan_counts.sort_values(by=header)
    return nan_counts


def count_nan_fracs(df_in, header=None, sort=False, percent=False):
    """
    Get fraction of nans in each column

    Parameters
    ----------
    df_in : pandas DataFrame or Series
    header : str, optional
        Name of the output DataFrame. Default is "nan_fracs"
    sort : bool, default False
        Sort columns from fewest nans to most nans.
    percent : bool, default False
        Display as percentage instead of fractions

    Returns
    -------
    pandas.Dataframe with original columns as index and fraction of nans in each column as value
    """
    if header is None:
        header = "nan_fracs"
    nan_fracs = (df_in.isnull().sum()/len(df_in)).to_frame(header)
    if sort:
        nan_fracs = nan_fracs.sort_values(by=header)
    if percent:
        nan_fracs = nan_fracs.applymap("{:.2%}".format)
    return nan_fracs


def _get_series_signs(data_in):
    """
    Map numeric values to [-1, 1] based on sign. Zeros and nans stay the same.
    See map_to_signs() for full documentation.
    """
    if data_in.dtype.kind not in 'biufc':
        raise TypeError(f"Expected numeric values but {data_in.name} Series has type {data_in.dtype.name} ")
    data = data_in.copy()
    data.loc[data > 0] = 1
    data.loc[data < 0] = -1
    return data


def map_to_signs(df_in, renamed_values: list = None):
    """
    Map numeric values to [-1, 1] based on sign. Zeros and nans stay the same.

    Parameters
    ----------
    df_in : pandas.Series or pandas.Dataframe
        Input data.
    renamed_values : list, optional
        Map -1.0 and 1.0 to something more descriptive, e.g. ["down", "up"]

    Returns
    -------
    Original dataframe with numeric columns mapped to [1, -1]

    Raises
    ------
    TypeError if any of Dataframe columns are non-numeric.
    """

    if renamed_values is not None:
        assert len(renamed_values) == 2, f"renamed_values has {len(renamed_values)} elements, expected 2"

    if isinstance(df_in, pd.Series):
        df_out = _get_series_signs(df_in)
        if renamed_values is not None:
            mapping = {-1.0: renamed_values[0], 1.0: renamed_values[1]}
            df_out = df_out.map(mapping)
    
    elif isinstance(df_in, pd.DataFrame):
        df_out = df_in.copy()
    
        for column in df_in.columns:
            df_out[column] = _get_series_signs(df_in[column])
            if renamed_values is not None:
                mapping = {-1.0: renamed_values[0], 1.0: renamed_values[1]}
                df_out[column] = df_out[column].map(mapping)
    else:
        raise TypeError(f"Input must be a pandas Series or DataFrame not a {type(df_in)}")
        
    return df_out


def sequential_merge(lst_dfs, **kwargs):
    """
    Sequentially join together a list of DataFrames into a single one using pandas.merge

    Parameters
    ----------
    lst_dfs : list
        DataFrames to merge together
    **kwargs
        Options for pandas.merge()

    Returns
    -------
    Combined Dataframe
    """
    df_combined = reduce(lambda x, y: pd.merge(x, y, **kwargs), lst_dfs)
    return df_combined


def inverse_diff(y0, diffs):
    """
    Convert differences to levels, i.e inverse of Series.diff()

    Parameters
    ----------
    y0 : float
        Initial value
    diffs : pandas.Series
        Values of differences in units of levels

    Returns
    -------
    pandas.Series with levels instead of returns

    """
    levels = np.concatenate(([y0], diffs), axis=0).cumsum()
    return pd.Series(levels)


def get_mem_usage(df):
    """
    Get total memory usage of a DataFrame in GB.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    Float
    """
    return df.memory_usage().sum() / 1e9


def unpack_series(data):
    """
    Unpack a pandas Series into index and values

    Parameters
    ----------
    data : pandas.Series

    Returns
    -------
    List of indices and list of values
    """
    return data.index, data.values


def explode_dict_column(df_in, column, column_suffix=None):
    """
    Unpack a Dataframe column of dicts into a new column for each key and drop the original column.
    Like a dict equivalent of pandas.Series.explode()

    Parameters
    ----------
    df_in : pandas DataFrame
        Input data. Must contain a column with dict values
    column : str
        The DataFrame column with dicts to expand
    column_suffix : str, optional
        Append "_<column_suffix>" to the end of output columns. Useful if applying this function on multiple columns.

    Returns
    -------
    Original DataFrame with dict column expanded

    Example
    -------
    >>> d = pd.DataFrame({'a':[1,2,3], 'b':[{'c':1}, {'d':3}, {'c':5, 'd':6}]})
    >>> explode_dict_column(d, "b")
       a    c    d
    0  1  1.0  NaN
    1  2  NaN  3.0
    2  3  5.0  6.0
    """
    df = df_in.copy()
    data = df[column]
    if not isinstance(data.iloc[0], dict):
        raise TypeError(f"Column must have type dict, not {type(data.iloc[0])}")

    data = data.apply(pd.Series)
    if column_suffix is not None:
        data.columns = [c + "_" + column_suffix for c in data.columns]
    df = pd.concat([df.drop(columns=column), data], axis=1)
    return df


def smart_log(data_in, base=None):
    """
    Extension of numpy.log handling invalid values according to
    log(x) -> -log(-x) if x < 0
    log(x) -> 0 if x == 0
    log(x) -> log(x) otherwise

    Note that this may not be a valid way of dealing with those cases.
    All this guarantees is the invalid value warnings from numpy are suppressed.

    Parameters
    ----------
    data_in : pandas Series
        Input data.
    base : int, optional
        Base of logarithm, default is natural.

    Returns
    -------
    pandas.Series
    """
    neg = data_in < 0
    zero = data_in == 0
    pos = data_in > 0

    data_out = data_in.copy()
    data_out.loc[neg] = - np.log(-data_in[neg])
    data_out.loc[pos] = np.log(data_in[pos])
    data_out.loc[zero] = 0

    if base is not None:
        data_out = data_out / np.log(base)
    return data_out


def smart_pivot(lst_dfs_in: List[pd.DataFrame], df_names: List[str],
                row_index_name: Optional[str] = "index", col_index_name: Optional[str] = "experiment"):
    """
    Join together several DataFrames along the column axis into a MultiIndex.
    Designed for when each DataFrame is a results table and you want multiple results tables together.
    Basically DataFrame.pivot_table() except it actually does what I want.


    Parameters
    ----------
    lst_dfs_in : list of DataFrames
       All the input DataFrames must have exactly the same index and columns.
    df_names: list of str
        Names for each sub-table that will be the column headers in the output DataFrame
    row_index_name: str, optional
        Default is "index"
    col_index_name: str, optional
        Default is "experiment". @TODO: Better default name?

    Returns
    -------
    pandas.DataFrame with a MultiIndex column axis.

    """

    lst_dfs = []
    for df in lst_dfs_in:
        lst_dfs.append(df.copy())

    assert len(lst_dfs) > 1, "Need more than one DataFrame to join"
    assert len(lst_dfs) == len(df_names), f"Must have same number of DataFrames as DataFrame names"
    assert all([(x.index == lst_dfs[0].index).all() for x in lst_dfs[1:]]), \
        f"All DataFrames must have identical indices"
    assert all([(x.columns == lst_dfs[0].columns).all() for x in lst_dfs[1:]]), \
        f"All DataFrames must have identical columns"

    for i in range(len(lst_dfs)):
        lst_dfs[i][col_index_name] = df_names[i]

    long_df = pd.concat(lst_dfs)
    long_df.index.name = row_index_name

    orig_row_ordering = long_df.index.unique()
    orig_col_ordering = long_df[col_index_name].unique()

    wide_df = long_df.pivot_table(index=row_index_name, columns=col_index_name)
    wide_df = wide_df.swaplevel(axis=1).sort_index(1)
    wide_df = wide_df[orig_col_ordering]
    wide_df = wide_df.loc[orig_row_ordering]
    return wide_df


def summarize(x: Union[np.ndarray, pd.Series, pd.DataFrame], name: Optional[str] = None, **kwargs):
    """
    Series.describe() function in pandas with nicer formatting and extra statistics. Also accepts numpy ndarray.

    Parameters
    ----------
    x : pd.Series or 1d DataFrame or 1d np.array
        Data
    name: str, optional
        Header of output table will be "<name> summary stats".
    **kwargs
        Extra options for pandas.DataFrame.describe()

    Returns
    -------
    DataFrame
    """
    # @todo: could tidy this up a bit
    if isinstance(x, pd.Series):
        if name is None:
            if x.name is None:
                name = ""
            else:
                name = x.name
    elif isinstance(x, np.ndarray):
        if name is None:
            name = ""
        assert x.ndim == 1, f"array must be one-dimensional but this has ndim = {x.ndim}"
        x = pd.Series(x)
    elif isinstance(x, pd.DataFrame):
        assert len(x.columns) == 1, "Only 1 column DataFrames allowed"
        if name is None:
            name = x.columns[0]
        x = pd.Series(x)
    else:
        raise TypeError(f"Invalid input type {type(x)}")

    summary = x.describe(**kwargs).apply('{:.3f}'.format)
    summary["skew"] = x.skew()
    summary["kurtosis"] = x.kurtosis()

    tstat = x.mean() / (x.std() / len(x))
    summary["tstat"] = tstat
    summary["tstat_pval"] = t(df=len(x)-1).sf(abs(tstat))
    summary = summary.to_frame(f"{name} summary stats")
    return summary


def read_and_stack_csvs(file_list: list, rename_columns: list = None, concat_axis=None, verbose=False, **kwargs):
    """
    Load a list of csv files and concatenate into a pandas Datafame.

    Parameters
    ----------
    file_list : list of strings
        Paths to csv files to load
    rename_columns : list of strings
        If concatenating along columns, rename the columns if they have the same name in all input files.
    concat_axis : int (optional)
        Concatenate the final list of Dataframes: 0 for along the row axis, 1 for along the column axis.
    verbose : bool (default=False)
        Display iteration step of file loop.
    **kwargs
        Options for pandas.read_csv()

    Returns
    -------
    Either a list of dataframes if concat_axis not specified or a single dataframe if concat_axis specified.
    """
    lst_dataframes = []

    if concat_axis not in [0, 1, None]:
        raise ValueError(f"Invalid value {concat_axis} for concat_axis.")

    for i, filename in enumerate(file_list):
        if verbose:
            print(f"At file {i} of {len(file_list)}")
        this_df = pd.read_csv(filename, **kwargs)
        if this_df.shape[1] > 1 and concat_axis == 1:
            # @TODO can we remove this error?
            raise ValueError("Currently can only stack a single column from each file. Pass usecols argument.")
        lst_dataframes.append(this_df)

    if concat_axis == 0:
        return pd.concat(lst_dataframes, axis=0, ignore_index=True)
    elif concat_axis == 1:
        joined_dataframe = pd.concat(lst_dataframes, axis=1, sort=True)
        if rename_columns is not None:
            joined_dataframe.columns = rename_columns
        return joined_dataframe
    else:
        return lst_dataframes
