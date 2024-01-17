# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
from types import GenericAlias
import sys

if sys.version_info >= (3, 10):
    from types import UnionType

from typing import Any, Callable, Union, cast, get_args, get_origin
import numpy as np
from . import yardl_types as yardl


def make_get_dtype_func(
    dtype_map: dict[
        Union[type, GenericAlias],
        Union[np.dtype[Any], Callable[[tuple[type, ...]], np.dtype[Any]]],
    ]
) -> Callable[[Union[type, GenericAlias]], np.dtype[Any]]:
    dtype_map[bool] = np.dtype(np.bool_)
    dtype_map[yardl.Int8] = np.dtype(np.int8)
    dtype_map[yardl.UInt8] = np.dtype(np.uint8)
    dtype_map[yardl.Int16] = np.dtype(np.int16)
    dtype_map[yardl.UInt16] = np.dtype(np.uint16)
    dtype_map[yardl.Int32] = np.dtype(np.int32)
    dtype_map[yardl.UInt32] = np.dtype(np.uint32)
    dtype_map[yardl.Int64] = np.dtype(np.int64)
    dtype_map[yardl.UInt64] = np.dtype(np.uint64)
    dtype_map[yardl.Size] = np.dtype(np.uint64)
    dtype_map[yardl.Float32] = np.dtype(np.float32)
    dtype_map[yardl.Float64] = np.dtype(np.float64)
    dtype_map[yardl.ComplexFloat] = np.dtype(np.complex64)
    dtype_map[yardl.ComplexDouble] = np.dtype(np.complex128)
    dtype_map[datetime.date] = np.dtype("datetime64[D]")
    dtype_map[yardl.Time] = np.dtype("timedelta64[ns]")
    dtype_map[yardl.DateTime] = np.dtype("datetime64[ns]")
    dtype_map[str] = np.dtype(np.object_)

    # Add the Python types to the dictionary too, but these may not be
    # correct since they map to several dtypes
    dtype_map[int] = np.dtype(np.int64)
    dtype_map[float] = np.dtype(np.float64)
    dtype_map[complex] = np.dtype(np.complex128)

    def get_dtype_impl(
        dtype_map: dict[
            Union[type, GenericAlias],
            Union[np.dtype[Any], Callable[[tuple[type, ...]], np.dtype[Any]]],
        ],
        t: Union[type, GenericAlias],
    ) -> np.dtype[Any]:
        # type_args = list(filter(lambda t: type(t) != TypeVar, get_args(t)))
        origin = get_origin(t)

        if origin == Union or (
            sys.version_info >= (3, 10) and isinstance(t, UnionType)
        ):
            return _get_union_dtype(get_args(t))

        # If t is found in dtype_map here, t is either a Python type
        # or t is a types.GenericAlias with missing type arguments
        if (res := dtype_map.get(t, None)) is not None:
            if callable(res):
                raise RuntimeError(f"Generic type arguments not provided for {t}")
            else:
                return res

        # Here, t is either invalid (no dtype registered)
        # or t is a types.GenericAlias with type arguments specified
        if origin is not None and (res := dtype_map.get(origin, None)) is not None:
            if callable(res):
                return res(get_args(t))

        raise RuntimeError(f"Cannot find dtype for {t}")

    def _get_union_dtype(args: tuple[type, ...]) -> np.dtype[Any]:
        if len(args) == 2 and args[1] == cast(type, type(None)):
            # This is an optional type
            inner_type = get_dtype_impl(dtype_map, args[0])
            return np.dtype(
                [("has_value", np.bool_), ("value", inner_type)], align=True
            )
        return np.dtype(np.object_)

    return lambda t: get_dtype_impl(dtype_map, t)
