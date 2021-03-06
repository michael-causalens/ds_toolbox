"""
json_helpers.py

Helper functions for reading and writing to json and accessing data in nested python dictionaries.
"""

import json
from collections import OrderedDict


def write_json(out_dict, out_path, sort_keys=True):
    """
    Output dictionary/JSON tree to file
    """
    if sort_keys:
        d = OrderedDict()
        for k in sorted(out_dict.keys()):
            d[k] = out_dict[k]
    else:
        d = out_dict
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)


def read_json(in_path):
    """
    Load JSON from file
    """
    with open(in_path, "r") as f_in:
        return json.load(f_in)


def _extract(input_dict, arr, key):
    """
    Recursively search for values of key in JSON tree.
    """
    if isinstance(input_dict, dict):

        for k, v in input_dict.items():
            if k == key:
                arr.append(v)
            elif k != key and isinstance(v, (dict, list)):
                _extract(v, arr, key)
    elif isinstance(input_dict, list):
        for item in input_dict:
            _extract(item, arr, key)
    return arr


def get_dict_depth(input_dict):
    """
    Get the depth (number of nestings) of a dictionary

    Parameters
    ----------
    input_dict : dict
        If input_dict is not a dictionary, returns depth 0

    Returns
    -------
    Integer depth of dictionary.

    Examples
    --------
    >>> get_dict_depth( {'A' : { 'a' : 1 } } )
    2
    """
    if isinstance(input_dict, dict):
        return 1 + (max(map(get_dict_depth, input_dict.values())) if input_dict else 0)
    return 0


def extract_values(input_dict, key):
    """
    Get a list of all values corresponding to a specified key in a dictionary.
    Note that the same key can appear at multiple nesting levels in dictionary.

    Parameters
    ----------
    input_dict : dict
    key : str
        inner-most key to get corresponding value from

    Returns
    -------
    List of all values for that key

    Example
    -------
    >>> test_dict = \
        { "Key_Uno" : { "Ein" : {"data" : [1, 2, 3],
                                 "labels" : ["one", "two", "three"], },
                        "Zwei" : {"data" : [8.4, 2.2, np.nan],
                                  "labels" : ["l_1", "l_2", "x"],
                                  "tags" : "has_nans" },
                        "labels" :  ["label_Ein", "label_Zwei"], },
        "Key_Dos" : "nada"}
    >>> extract_values(input_dict, "labels")
    [['one', 'two', 'three'], ['l_1', 'l_2', 'x'], ['label_Ein', 'label_Zwei']]
    """
    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dict")

    arr = []
    result = _extract(input_dict, arr, key)

    return result


def _get_keys_at_depth(input_dict, depth):
    """
    Recurse through a nested dictionary
    Get all keys at a certain depth, including duplicates

    Parameters
    ----------
    input_dict : dict
    depth : int
        The maximum tree depth to recurse

    Returns
    -------
    Lazy iterator over all keys at a certain level
     - call list(iterator), or set(iterator) to drop duplicates

    Raises
    ------
    AttributeError:
        If input_dict is not a dict
    """

    dict_depth = get_dict_depth(input_dict)
    if depth > dict_depth:
        raise ValueError(f"requested depth {depth} exceeds actual depth {dict_depth}")

    if depth == 1:
        yield from input_dict.keys()
    else:
        for v in input_dict.values():
            if isinstance(v, dict):
                yield from _get_keys_at_depth(v, depth - 1)
            else:
                # v is not a dictionary key
                continue


def get_unique_keys_at_depth(input_dict, depth):
    """
    Get a list of all unique keys at a certain depth of a dictionary.
    Note that a dictionary might have mixed levels of nesting, i.e. both keys and values at a certain depth.
    This only extracts the keys.

    Parameters
    ----------
    input_dict : dict
    depth : int
        Maximum depth

    Returns
    -------
    A list of all unique keys at that depth.

    Example
    -------
    >>> test_dict = \
        { "Key_Uno" : { "Ein" : {"data" : [1, 2, 3],
                                 "labels" : ["one", "two", "three"], },
                        "Zwei" : {"data" : [8.4, 2.2, np.nan],
                                  "labels" : ["l_1", "l_2", "x"],
                                  "tags" : "has_nans" },
                        "labels" :  ["label_Ein", "label_Zwei"], },
        "Key_Dos" : "nada"}
    >>> get_unique_keys_at_depth(test_dict, 3)
    ['data', 'labels', 'tags']
    """
    if depth == 0:
        return []

    unique_keys = []
    set_unique_keys = set()

    keys_iterator = _get_keys_at_depth(input_dict, depth)
    for key in keys_iterator:
        if isinstance(key, list):
            # set elements can't be lists
            key_unmutable = tuple(key)
        else:
            key_unmutable = key
        if key_unmutable not in set_unique_keys:
            set_unique_keys.add(key_unmutable)
            unique_keys.append(key)

    return unique_keys
