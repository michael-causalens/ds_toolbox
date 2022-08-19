"""
json_helpers.py

Helper functions for reading and writing to json and accessing data in nested python dictionaries.
"""

import json
import numpy as np
from warnings import warn
from collections import OrderedDict


def nan_to_none(obj):
    # todo: move this to low-level
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


class NanConverter(json.JSONEncoder):
    def default(self, obj):
        # possible other customizations here
        pass

    def encode(self, obj, *args, **kwargs):
        obj = nan_to_none(obj)
        return super().encode(obj, *args, **kwargs)

    def iterencode(self, obj, *args, **kwargs):
        obj = nan_to_none(obj)
        return super().iterencode(obj, *args, **kwargs)


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)


def write_json(out_dict, out_path, sort_keys=True, cls=None):
    """
    Output dictionary/JSON tree to file

    Parameters
    ----------
    out_dict : dict
        data to save
    out_path : str
        Name of output file
    sort_keys : bool, default True
        Alphanumerically sort keys before writing
    cls: json.JSONEncoder
        custom decoder for json serialising

    """
    if sort_keys:
        d = OrderedDict()
        for k in sorted(out_dict.keys()):
            d[k] = out_dict[k]
    else:
        d = out_dict
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2, cls=cls)


def read_json(in_path, cls=None):
    """
    Load JSON from file

    Parameters
    ----------
    in_path : str
        Full or relative path to json file.
    cls: json.JSONEncoder
        custom decoder for json serialising

    Returns
    -------
    dict
    """
    with open(in_path, "r") as f_in:
        return json.load(f_in, cls=cls)


def read_jsonlines(in_path, output="list"):
    """
    Load a jsonl (jsonlines format) file into a list of dicts or a plain dict

    Parameters
    ----------
    in_path : str
        Full or relative path to jsonL file.
    output : str, optional
        Either 'list' for a list of dicts or 'dict' for a single dict.
        The latter requires each line to have a single outermost key.

    Returns
    -------
    list of dicts
    """
    if not in_path.endswith("jsonl"):
        warn("This may not be a jsonl file, may get parser errors.")

    if output == "list":
        lines = []
    elif output == "dict":
        lines = {}
    else:
        raise ValueError(f"output must be either 'list' or 'dict', not '{output}")

    for i, line in enumerate(open(in_path, "r")):
        parsed_line = json.loads(line)
        if output == "list":
            lines.append(parsed_line)
        else:
            keys = parsed_line.keys()
            assert len(keys) == 1, f"Each line must have a single key, but line {i} has {len(keys)}"
            k = list(parsed_line.keys())[0]
            lines[k] = parsed_line[k]

    return lines


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
