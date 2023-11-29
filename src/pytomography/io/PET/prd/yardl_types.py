# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from abc import ABC
from enum import Enum
from typing import Annotated, Generic, TypeVar, Union
import numpy as np
import datetime
import time


class ProtocolError(Exception):
    """Raised when the contract of a protocol is not respected."""

    pass


class OutOfRangeEnum(Enum):
    """Enum that allows values outside of the its defined values."""

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, int):
            return None

        obj = object.__new__(cls)
        obj._value_ = value
        obj._name_ = ""
        return obj

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        if self._name_ != "":
            return super().__str__()

        return f"{self.__class__.__name__}({self.value})"

    def __repr__(self) -> str:
        if self._name_ != "":
            return super().__repr__()

        return f"<{self.__class__.__name__}: {self.value}>"


class DateTime:
    """A basic datetime with nanosecond precision, always in UTC."""

    def __init__(self, nanoseconds_from_epoch: Union[int, np.datetime64] = 0):
        if isinstance(nanoseconds_from_epoch, np.datetime64):
            if nanoseconds_from_epoch.dtype != "datetime64[ns]":
                self._value = np.datetime64(nanoseconds_from_epoch, "ns")
            else:
                self._value = nanoseconds_from_epoch
        else:
            self._value = np.datetime64(nanoseconds_from_epoch, "ns")

    @property
    def numpy_value(self) -> np.datetime64:
        return self._value

    def to_datetime(self) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(self._value.astype(int) / 1e9)

    @staticmethod
    def from_components(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        nanosecond: int = 0,
    ) -> "DateTime":
        if not 0 <= nanosecond <= 999_999_999:
            raise ValueError("nanosecond must be in 0..999_999_999", nanosecond)

        return DateTime(
            int(datetime.datetime(year, month, day, hour, minute, second).timestamp())
            * 1_000_000_000
            + nanosecond
        )

    @staticmethod
    def from_datetime(dt: datetime.datetime) -> "DateTime":
        return DateTime(round(dt.timestamp() * 1e6) * 1000)

    @staticmethod
    def parse(s: str) -> "DateTime":
        return DateTime(np.datetime64(s, "ns"))

    @staticmethod
    def now() -> "DateTime":
        return DateTime(time.time_ns())

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"DateTime({self})"

    def __eq__(self, other: object) -> bool:
        return (
            self._value == other._value
            if isinstance(other, DateTime)
            else (isinstance(other, np.datetime64) and self._value == other)
        )

    def __hash__(self) -> int:
        return hash(self._value)


class Time:
    """A basic time of day with nanosecond precision. It is not timezone-aware and is meant
    to represent a wall clock time.
    """

    _NANOSECONDS_PER_DAY = 24 * 60 * 60 * 1_000_000_000

    def __init__(self, nanoseconds_since_midnight: Union[int, np.timedelta64] = 0):
        if isinstance(nanoseconds_since_midnight, np.timedelta64):
            if nanoseconds_since_midnight.dtype != "timedelta64[ns]":
                self._value = np.timedelta64(nanoseconds_since_midnight, "ns")
                nanoseconds_since_midnight = nanoseconds_since_midnight.astype(int)
            else:
                self._value = nanoseconds_since_midnight
        else:
            self._value = np.timedelta64(nanoseconds_since_midnight, "ns")

        if (
            nanoseconds_since_midnight < 0
            or nanoseconds_since_midnight >= Time._NANOSECONDS_PER_DAY
        ):
            raise ValueError(
                "TimeOfDay must be between 00:00:00 and 23:59:59.999999999"
            )

    @property
    def numpy_value(self) -> np.timedelta64:
        return self._value

    @staticmethod
    def from_components(
        hour: int, minute: int, second: int = 0, nanosecond: int = 0
    ) -> "Time":
        if not 0 <= hour <= 23:
            raise ValueError("hour must be in 0..23", hour)
        if not 0 <= minute <= 59:
            raise ValueError("minute must be in 0..59", minute)
        if not 0 <= second <= 59:
            raise ValueError("second must be in 0..59", second)
        if not 0 <= nanosecond <= 999_999_999:
            raise ValueError("nanosecond must be in 0..999_999_999", nanosecond)

        return Time(
            hour * 3_600_000_000_000
            + minute * 60_000_000_000
            + second * 1_000_000_000
            + nanosecond
        )

    @staticmethod
    def from_time(t: datetime.time) -> "Time":
        return Time(
            t.hour * 3_600_000_000_000
            + t.minute * 60_000_000_000
            + t.second * 1_000_000_000
            + t.microsecond * 1_000
        )

    @staticmethod
    def parse(s: str) -> "Time":
        components = s.split(":")
        if len(components) == 2:
            hour = int(components[0])
            minute = int(components[1])
            return Time(hour * 3_600_000_000_000 + minute * 60_000_000_000)
        if len(components) == 3:
            hour = int(components[0])
            minute = int(components[1])
            second_components = components[2].split(".")
            if len(second_components) <= 2:
                second = int(second_components[0])
                if len(second_components) == 2:
                    fraction = int(second_components[1].ljust(9, "0")[:9])
                else:
                    fraction = 0
                return Time(
                    hour * 3_600_000_000_000
                    + minute * 60_000_000_000
                    + second * 1_000_000_000
                    + fraction
                )

        raise ValueError("TimeOfDay must be in the format HH:MM:SS[.fffffffff]")

    def __str__(self) -> str:
        nanoseconds_since_midnight = self._value.astype(int)
        hours, r = divmod(nanoseconds_since_midnight, 3_600_000_000_000)
        minutes, r = divmod(r, 60_000_000_000)
        seconds, nanoseconds = divmod(r, 1_000_000_000)
        if nanoseconds == 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        return f"{hours:02}:{minutes:02}:{seconds:02}.{str(nanoseconds).rjust(9, '0').rstrip('0')}"

    def __repr__(self) -> str:
        return f"Time({self})"

    def __eq__(self, other: object) -> bool:
        return (
            self._value == other._value
            if isinstance(other, Time)
            else (isinstance(other, np.timedelta64) and self._value == other)
        )


Int8 = Annotated[int, "Int8"]
UInt8 = Annotated[int, "UInt8"]
Int16 = Annotated[int, "Int16"]
UInt16 = Annotated[int, "UInt16"]
Int32 = Annotated[int, "Int32"]
UInt32 = Annotated[int, "UInt32"]
Int64 = Annotated[int, "Int64"]
UInt64 = Annotated[int, "UInt64"]
Size = Annotated[int, "Size"]
Float32 = Annotated[float, "Float32"]
Float64 = Annotated[float, "Float64"]
ComplexFloat = Annotated[complex, "ComplexFloat"]
ComplexDouble = Annotated[complex, "ComplexDouble"]


def structural_equal(a: object, b: object) -> bool:
    if a is None:
        return b is None

    if isinstance(a, list):
        if not isinstance(b, list):
            if isinstance(b, np.ndarray):
                return b.shape == (len(a),) and all(
                    structural_equal(x, y) for x, y in zip(a, b)
                )
            return False
        return len(a) == len(b) and all(structural_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            if isinstance(b, list):
                return a.shape == (len(b),) and all(
                    structural_equal(x, y) for x, y in zip(a, b)
                )
            return False
        if a.dtype.hasobject:  # pyright: ignore [reportUnknownMemberType]
            return (
                a.dtype == b.dtype  # pyright: ignore [reportUnknownMemberType]
                and a.shape == b.shape
                and all(structural_equal(x, y) for x, y in zip(a, b))
            )
        return np.array_equal(a, b)

    if isinstance(a, np.void):
        if not isinstance(b, np.void):
            return b == a
        return a.dtype == b.dtype and all(
            structural_equal(x, y)
            for x, y in zip(a, b)  # pyright: ignore [reportGeneralTypeIssues]
        )

    if isinstance(b, np.void):
        return a == b

    return a == b


_T = TypeVar("_T")


class UnionCase(ABC, Generic[_T]):
    index: int
    tag: str

    def __init__(self, value: _T) -> None:
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, other: object) -> bool:
        # Note we could codegen a more efficient version of this that does not
        # (always) call structural_equal
        return type(self) == type(other) and structural_equal(
            self.value, other.value  # pyright: ignore
        )
