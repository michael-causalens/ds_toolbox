"""
s3_helpers.py

Helper functions for reading and writing to s3

@todo: better error catching for invalid paths, s3 access permissions etc.

"""

import os
import io
import json
import gzip
import boto3
import pickle
import warnings
import pandas as pd


def create_aws_client(profile=None):
    """
    Create a client to read/write to s3.

    Parameters
    ----------
    profile : str (optional)
        Name of aws profile to read credentials from, if needed.

    Returns
    -------
    A boto3 S3.Client object
    """
    if profile is not None:
        session = boto3.Session(profile_name=profile)
        client = session.client('s3')
    else:
        client = boto3.client('s3')

    return client


def _yield_all_s3_objects(client, **kwargs):
    """
    Get a lazy iterator over all objects in an s3 folder.

    It has the same basic functionality as boto3.client.list_objects()
    Annoyingly, however, boto3.client.list_objects returns a maximum of 1000 keys.
    This is hardcoded, so we need a workaround.

    This function repeatedly calls the list_objects_v2 function  
    until the response from that call no longer returns True for the "IsTruncated" key
    i.e. the end of the list of Contents has been reached.

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    **kwargs
        Options for client.list_objects_v2 call, such as Bucket, Prefix etc.

    Returns
    -------
    A generator expression iterating over all objects under 'Prefix' in 'Bucket'
    """

    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = client.list_objects_v2(**list_kwargs)

        yield from response.get('Contents')
        # stop when the end of the list is reached
        if not response.get('IsTruncated'):
            break
        continuation_token = response.get('NextContinuationToken')


def _get_streaming_body(client, bucket_name, s3_filename):
    """
    Get a boto3 streaming body object from the file given.
    Essentially this is a bytestream with a "read" method,
    which can be passed as an argument to open(), pandas.read_csv() etc.
    
    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the file on the s3 bucket

    Returns
    -------
    A boto3 StreamingBody object
    """

    obj = client.get_object(Bucket=bucket_name, Key=s3_filename)
    body = obj["Body"]
    return body


def list_all_s3_contents(client, bucket_name, prefix, matching=None, matching_rule=None):
    """
    List of all keys in a folder of an s3 bucket
    Calls a generator expression from yield_all_s3_contents
    and builds a list of all values of 'Key' from that expression

    Note: this will store ALL key names in memory, so be wary of size...

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    prefix : str
        The subfolder of the bucket to look for the files
    matching : str or list of strs
        Only keep keys containing string provided or keys containing any/all strings in list provided
    matching_rule : str
        If matching a list of strings, succeed if matching any in list or only if all in list. Default is 'any'

    Returns
    -------
    List of all keys in that prefix - i.e list of all files in that folder
    """

    objs_iterator = _yield_all_s3_objects(client, Bucket=bucket_name, Prefix=prefix)
    list_of_keys = [x['Key'] for x in objs_iterator]

    if matching is not None:
        if isinstance(matching, str):
            if matching_rule is not None:
                raise TypeError("matching_rule only valid when matching is a list")
            list_of_keys = [x for x in list_of_keys if matching in x]
        elif isinstance(matching, list):
            if matching_rule == "any" or matching_rule is None:
                list_of_keys = [x for x in list_of_keys if any(i in x for i in matching)]
            elif matching_rule == "all":
                list_of_keys = [x for x in list_of_keys if all(i in x for i in matching)]
            else:
                raise ValueError("matching_rule must be 'any' or 'all'")
        else:
            raise TypeError("matching must be a string or a list of strings")
    return list_of_keys


def load_pickle_from_s3(client, bucket_name, s3_filename, compression=None):
    """
    Load a pickle file from an s3 bucket.  
    
    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the pickle file on the s3 bucket
    compression: str
        Either "gzip" or None
        
    Returns
    -------
    The unpickled human-readable object.
    
    Raises
    ------
    ValueError
        If any compression other than "gzip" selected.
    
    Note
    ----
    Pickled objects contain zero protection against malicious code.
    NEVER load pickled objects from untrusted sources. 
    """
    
    body = _get_streaming_body(client, bucket_name, s3_filename)

    if compression is not None:
        if compression != "gzip":
            raise ValueError('Only "gzip" is a valid compression')
        with gzip.open(body) as f:
            unpickled_data = pickle.load(f)
    else:
        if s3_filename.endswith("z"):
            warnings.warn('This file may be compressed, consider setting compression="gzip".')
        try:
            with open(body, 'rb') as f:
                unpickled_data = pickle.load(f)
        except TypeError:  # "expected not StreamingBody object"
            unpickled_data = pickle.loads(body.read())
    return unpickled_data
    

def download_from_s3(client, bucket_name, s3_filename, local_path=None, overwrite=False):
    """
    Download a file from an s3 bucket to a local folder.
    
    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The path to the file on the s3 bucket
    local_path : str (optional)
        Local output path of file. If not specified, uses the current folder.
    overwrite : bool
        Whether to overwrite if local file already exists.
        
    Raises
    ------
    OSError 
        If download path already exists and overwrite not True.
    """ 
    
    if local_path is None:
        local_path = os.path.basename(s3_filename)
    if os.path.exists(local_path) and not overwrite:
        raise OSError("This file already exists. Set overwrite=True or use a different name.")
    
    client.download_file(bucket_name, s3_filename, local_path)


def read_json_from_s3(client, bucket_name, s3_filename, **kwargs):
    """
    Load a json file into a python dictionary from an s3 bucket

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the json file on the s3 bucket
    **kwargs
        Currently only "encoding" parameter for boto3.StreamingBody.read().decode() implemented

    Returns
    -------
    A python dictionary
    """
    body = _get_streaming_body(client, bucket_name, s3_filename)

    if kwargs.get("encoding"):
        file_content = body.read().decode(kwargs["encoding"])
    else:
        file_content = body.read().decode()
    json_content = json.loads(file_content)
    return json_content


def pd_read_csv_from_s3(client, bucket_name, s3_filename, **kwargs):
    """
    Load a csv file into a pandas DataFrame from an s3 bucket

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the csv file on the s3 bucket
    **kwargs
        Options for pandas.read_csv method

    Returns
    -------
    A pandas DataFrame

    Note
    ----
    If file is 'zip' compressed rather than 'gzip', pd.read_csv cannot read the file-handle directly from s3
    It must be converted to a bit stream. We use a BytesIO object from the `io` library.
    """

    body = _get_streaming_body(client, bucket_name, s3_filename)

    if kwargs.get("compression") == "zip":
        df = pd.read_csv(io.BytesIO(body.read()), **kwargs)
    else:
        df = pd.read_csv(body, **kwargs)

    return df


def pd_read_excel_from_s3(client, bucket_name, s3_filename, **kwargs):
    """
    Load an Excel spreadsheet into a pandas DataFrame from an s3 bucket

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the .xlsx file on the s3 bucket
    **kwargs
        Options for pandas.read_excel method, e.g. sheet_name

    Returns
    -------
    A pandas DataFrame

    Note
    ----
    If file is 'zip' compressed rather than 'gzip', pd.read_excel cannot read the file-handle directly from s3
    It must be converted to a bit stream. We use a BytesIO object from the `io` library.
    """

    body = _get_streaming_body(client, bucket_name, s3_filename)
    if kwargs.get("compression") == "zip":
        df = pd.read_excel(io.BytesIO(body.read()), compression='zip')
    else:
        df = pd.read_excel(body, **kwargs)

    return df


def pd_read_parquet_from_s3(client, bucket_name, s3_filename, **kwargs):
    """
    Load a parquet file into a pandas DataFrame from an s3 bucket

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    bucket_name : str
        The name of the bucket
    s3_filename : str
        The full path to the parquet file on the s3 bucket
    **kwargs
        Options for pandas.read_parquet method

    Returns
    -------
    A pandas DataFrame

    Note
    ----
    If file is 'zip' compressed rather than 'gzip', pd.read_parquet cannot read the file-handle directly from s3
    It must be converted to a bit stream. We use a BytesIO object from the `io` library.
    """

    body = _get_streaming_body(client, bucket_name, s3_filename)
    df = pd.read_parquet(io.BytesIO(body.read()), **kwargs)

    return df


def write_file_to_s3(client, full_path, bucket_name, prefix, verbose=False):
    """
    Upload a file of any type to s3

    Parameters
    ----------
    client :
        A boto3 s3 Client object, assumed to be the output from create_aws_client()
    full_path : str
        The full local path name of the file to upload.
    bucket_name : str
        The name of the bucket
    prefix : str
        The subfolder of the bucket to upload the files
    verbose : bool
        Print upload information.
    """

    # in case full path to source file is given, we just want the filename
    filename = os.path.basename(full_path)

    # make sure supplied path has trailing slash
    if not prefix.endswith("/"):
        prefix += "/"

    key_name = prefix + filename

    if verbose:
        print(f"Uploading {full_path} to {key_name} in {bucket_name}")
    client.upload_file(Filename=filename, Bucket=bucket_name, Key=key_name)
    if verbose:
        print("Done")
