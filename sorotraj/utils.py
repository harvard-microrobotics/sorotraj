import yaml
import os

from typing import Any

def save_yaml(data: Any, filename: str):
    """
    Save data to a yaml file.

    Parameters
    ----------
    data : Any
        pythonic data object to save
    filename : str
        The filename to save

    Raises
    ------
    ValueError
        If the filename is not of type 'str'
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
        return

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=None)


def load_yaml(filename: str):
    """
    Load data from a yaml file.

    Parameters
    ----------
    filename : str
        The filename to save

    Returns
    -------
    data : Any
        The data read from the file

    Raises
    ------
    ValueError
        If the filename is not of type 'str'
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
        return

    with open(filename) as f:
        # use safe_load instead of load
        data = yaml.safe_load(f)

    return data