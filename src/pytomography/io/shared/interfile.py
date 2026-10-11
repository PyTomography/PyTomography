from __future__ import annotations
from pathlib import Path
import numpy as np
import os
import re
import torch
import pytomography


def get_header_value(
    list_of_attributes: list[str],
    header: str,
    dtype: type = np.float32,
    split_substr = ':=',
    split_idx = -1,
    return_all = False
    ) -> float|str|int:
    """Finds the first entry in an Interfile with the string ``header``

    Args:
        list_of_attributes (list[str]): Simind data file, as a list of lines.
        header (str): The header looked for
        dtype (type, optional): The data type to be returned corresponding to the value of the header. Defaults to np.float32.

    Returns:
        float|str|int: The value corresponding to the header (header).
    """
    header = header.replace('[', r'\[').replace(']',r'\]').replace('(', r'\(').replace(')', r'\)')
    y = np.vectorize(lambda y, x: bool(re.compile(x).search(y)))
    selection = y(list_of_attributes, header).astype(bool)
    lines = list_of_attributes[selection]
    if len(lines)==0:
        return False
    values = []
    for i, line in enumerate(lines):
        if dtype == np.float32:
            values.append(np.float32(line.replace('\n', '').split(split_substr)[split_idx]))
        elif dtype == str:
            values.append(line.replace('\n', '').split(split_substr)[split_idx].replace(' ', ''))
        elif dtype == int:
            values.append(int(line.replace('\n', '').split(split_substr)[split_idx].replace(' ', '')))
        if not(return_all):
            return values[0]
    return values
    
