"""
misc_utils.py

Assorted useful helper functions for pandas Dataframes
"""
import numpy as np
import pandas as pd
from functools import reduce


def count_nans(df_in, header=None):
    """
    Get number of nans in each column

    Parameters
    ----------
    df_in : pandas DataFrame or Series
    header : str (optional)
        Name of the output DataFrame. Default is "nan_counts"

    Returns
    -------
    pandas.Dataframe with original columns as index and number of nans in each column as value
    """
    if header is None:
        header = "nan_counts"
    nan_counts = df_in.isnull().sum().to_frame(header)
    return nan_counts


def count_nan_fracs(df_in, header=None, percent=False):
    """
    Get fraction of nans in each column

    Parameters
    ----------
    df_in : pandas DataFrame or Series
    header : str (optional)
        Name of the output DataFrame. Default is "nan_fracs"
    percent : bool (default False)
        Display as percentage instead of fractions

    Returns
    -------
    pandas.Dataframe with original columns as index and fraction of nans in each column as value
    """
    if header is None:
        header = "nan_fracs"
    nan_fracs = (df_in.isnull().sum()/len(df_in)).to_frame(header)
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


def map_to_signs(df_in):
    """
    Map numeric values to [-1, 1] based on sign. Zeros and nans stay the same.

    Parameters
    ----------
    df_in : pandas.Series or pandas.Dataframe
        Input data.

    Returns
    -------
    Original dataframe with numeric columns mapped to [1, -1]

    Raises
    ------
    TypeError if any of Dataframe columns are non-numeric.
    """
    
    if isinstance(df_in, pd.Series):
        df_out = _get_series_signs(df_in)
    
    elif isinstance(df_in, pd.DataFrame):
        df_out = df_in.copy()
    
        for column in df_in.columns:
            df_out[column] = _get_series_signs(df_in[column])
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


def inverse_diff(y0, returns):
    """
    Convert returns to levels, i.e inverse of Series.diff()

    Parameters
    ----------
    y0 : float
        Initial value
    returns : pandas.Series
        Values of returns

    Returns
    -------
    pandas.Series with levels instead of returns

    """
    levels = np.concatenate(([y0], returns), axis=0).cumsum()
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


def explode_dict_column(df_in, column):
    """
    Unpack a Dataframe column of dicts into a new column for each key and drop the original column.
    Like a dict equivalent of pandas.Series.explode()

    Parameters
    ----------
    df_in : pandas DataFrame
        Input data. Must contain a column with dict values
    column : str
        The DataFrame column with dicts to expand

    Returns
    -------
    Original DataFrame with dict column expanded

    Example
    -------
    >>> df = pd.DataFrame({'a':[1,2,3], 'b':[{'c':1}, {'d':3}, {'c':5, 'd':6}]})
    >>> explode_dict_column(df, "b")
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
    df = pd.concat([df.drop(columns=column), data], axis=1)
    return df


# def read_csvs(file_list: list, concat_axis=None, verbose=False, **kwargs):
#     """
#     Load a list of csv files into a pandas Datafame
#     @todo: this is still not general enough
#
#     Parameters
#     ----------
#     file_list : list of strings
#         Paths to csv files to load
#     concat_axis : int (optional)
#         Concatenate the final list of Dataframes: 0 for along the row axis, 1 for along the column axis.
#     verbose : bool (default=False)
#         Display iteration step of file loop.
#     **kwargs
#         Options for pandas.read_csv()
#
#     Returns
#     -------
#     Either a list of dataframes if concat_axis not specified or a single dataframe if concat_axis specified.
#     """
#     lst_dataframes = []
#
#     for i, filename in enumerate(file_list):
#         if verbose:
#             print(f"At file {i} of {len(file_list)}")
#         this_df = pd.read_csv(filename, **kwargs)
#         lst_dataframes.append(this_df)
#
#     if concat_axis is not None:
#         return pd.concat(lst_dataframes, axis=concat_axis, ignore_index=True)
#     else:
#         return lst_dataframes
