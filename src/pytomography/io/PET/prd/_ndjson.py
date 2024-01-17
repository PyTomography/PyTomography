# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from abc import ABC, abstractmethod
import datetime
from enum import IntFlag
import io
import json
from typing import Any, Generic, Optional, TextIO, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions

from .yardl_types import *

CURRENT_NDJSON_FORMAT_VERSION: int = 1

INT8_MIN: int = np.iinfo(np.int8).min
INT8_MAX: int = np.iinfo(np.int8).max

UINT8_MAX: int = np.iinfo(np.uint8).max

INT16_MIN: int = np.iinfo(np.int16).min
INT16_MAX: int = np.iinfo(np.int16).max

UINT16_MAX: int = np.iinfo(np.uint16).max

INT32_MIN: int = np.iinfo(np.int32).min
INT32_MAX: int = np.iinfo(np.int32).max

UINT32_MAX: int = np.iinfo(np.uint32).max

INT64_MIN: int = np.iinfo(np.int64).min
INT64_MAX: int = np.iinfo(np.int64).max

UINT64_MAX: int = np.iinfo(np.uint64).max

MISSING_SENTINEL = object()


class NDJsonProtocolWriter(ABC):
    def __init__(self, stream: Union[TextIO, str], schema: str) -> None:
        if isinstance(stream, str):
            self._stream = open(stream, "w", encoding="utf-8")
            self._owns_stream = True
        else:
            self._stream = stream
            self._owns_stream = False

        self._write_json_line(
            {
                "yardl": {
                    "version": CURRENT_NDJSON_FORMAT_VERSION,
                    "schema": json.loads(schema),
                },
            },
        )

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def _end_stream(self) -> None:
        pass

    def _write_json_line(self, value: object) -> None:
        json.dump(
            value,
            self._stream,
            ensure_ascii=False,
            separators=(",", ":"),
            check_circular=False,
        )
        self._stream.write("\n")


class NDJsonProtocolReader:
    def __init__(
        self, stream: Union[io.BufferedReader, TextIO, str], schema: str
    ) -> None:
        if isinstance(stream, str):
            self._stream = open(stream, "r", encoding="utf-8")
            self._owns_stream = True
        else:
            self._stream = stream
            self._owns_stream = False

        self._unused_value: Optional[dict[str, object]] = None

        line = self._stream.readline()
        try:
            header_json = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(
                "Data in the stream is not in the expected Yardl NDJSON format."
            )

        if not isinstance(header_json, dict) or not "yardl" in header_json:
            raise ValueError(
                "Data in the stream is not in the expected Yardl NDJSON format."
            )

        header_json = header_json["yardl"]
        if not isinstance(header_json, dict):
            raise ValueError(
                "Data in the stream is not in the expected Yardl NDJSON format."
            )

        if (
            header_json.get("version")  # pyright: ignore [reportUnknownMemberType]
            != CURRENT_NDJSON_FORMAT_VERSION
        ):
            raise ValueError("Unsupported yardl version.")

        if header_json.get(  # pyright: ignore [reportUnknownMemberType]
            "schema"
        ) != json.loads(schema):
            raise ValueError(
                "The schema of the data to be read is not compatible with the current protocol."
            )

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def _read_json_line(self, stepName: str, required: bool) -> object:
        missing = MISSING_SENTINEL
        if self._unused_value is not None:
            if (value := self._unused_value.get(stepName, missing)) is not missing:
                self._unused_value = None
                return value
            if required:
                raise ValueError(f"Expected protocol step '{stepName}' not found.")

        line = self._stream.readline()
        if line == "":
            if not required:
                return MISSING_SENTINEL
            raise ValueError(
                f"Encountered EOF but expected to find protocol step '{stepName}'."
            )

        json_object = json.loads(line)
        if (value := json_object.get(stepName, missing)) is not MISSING_SENTINEL:
            return value

        if not required:
            self._unused_value = json_object
            return MISSING_SENTINEL

        raise ValueError(f"Expected protocol step '{stepName}' not found.")


T = TypeVar("T")
T_NP = TypeVar("T_NP", bound=np.generic)


class JsonConverter(Generic[T, T_NP], ABC):
    def __init__(self, dtype: npt.DTypeLike) -> None:
        self._dtype: np.dtype[Any] = np.dtype(dtype)

    def overall_dtype(self) -> np.dtype[Any]:
        return self._dtype

    @abstractmethod
    def to_json(self, value: T) -> object:
        raise NotImplementedError

    @abstractmethod
    def numpy_to_json(self, value: T_NP) -> object:
        raise NotImplementedError

    @abstractmethod
    def from_json(self, json_object: object) -> T:
        raise NotImplementedError

    @abstractmethod
    def from_json_to_numpy(self, json_object: object) -> T_NP:
        raise NotImplementedError

    def supports_none(self) -> bool:
        return False


class BoolConverter(JsonConverter[bool, np.bool_]):
    def __init__(self) -> None:
        super().__init__(np.bool_)

    def to_json(self, value: bool) -> object:
        if not isinstance(value, bool):
            raise TypeError(f"Expected a bool but got {type(value)}")

        return value

    def numpy_to_json(self, value: np.bool_) -> object:
        return bool(value)

    def from_json(self, json_object: object) -> bool:
        return bool(json_object)

    def from_json_to_numpy(self, json_object: object) -> np.bool_:
        return np.bool_(json_object)


bool_converter = BoolConverter()


class Int8Converter(JsonConverter[int, np.int8]):
    def __init__(self) -> None:
        super().__init__(np.int8)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not a signed 8-bit integer: {value}")
        if value < INT8_MIN or value > INT8_MAX:
            raise ValueError(
                f"Value {value} is outside the range of a signed 8-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.int8) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.int8:
        return np.int8(cast(int, json_object))


int8_converter = Int8Converter()


class UInt8Converter(JsonConverter[int, np.uint8]):
    def __init__(self) -> None:
        super().__init__(np.uint8)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not an unsigned 8-bit integer: {value}")
        if value < 0 or value > UINT8_MAX:
            raise ValueError(
                f"Value {value} is outside the range of an unsigned 8-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.uint8) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.uint8:
        return np.uint8(cast(int, json_object))


uint8_converter = UInt8Converter()


class Int16Converter(JsonConverter[int, np.int16]):
    def __init__(self) -> None:
        super().__init__(np.int16)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not a signed 16-bit integer: {value}")
        if value < INT16_MIN or value > INT16_MAX:
            raise ValueError(
                f"Value {value} is outside the range of a signed 16-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.int16) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.int16:
        return np.int16(cast(int, json_object))


int16_converter = Int16Converter()


class UInt16Converter(JsonConverter[int, np.uint16]):
    def __init__(self) -> None:
        super().__init__(np.uint16)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not an unsigned 16-bit integer: {value}")
        if value < 0 or value > UINT16_MAX:
            raise ValueError(
                f"Value {value} is outside the range of an unsigned 16-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.uint16) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.uint16:
        return np.uint16(cast(int, json_object))


uint16_converter = UInt16Converter()


class Int32Converter(JsonConverter[int, np.int32]):
    def __init__(self) -> None:
        super().__init__(np.int32)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not a signed 32-bit integer: {value}")
        if value < INT32_MIN or value > INT32_MAX:
            raise ValueError(
                f"Value {value} is outside the range of a signed 32-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.int32) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.int32:
        return np.int32(cast(int, json_object))


int32_converter = Int32Converter()


class UInt32Converter(JsonConverter[int, np.uint32]):
    def __init__(self) -> None:
        super().__init__(np.uint32)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not an unsigned 32-bit integer: {value}")
        if value < 0 or value > UINT32_MAX:
            raise ValueError(
                f"Value {value} is outside the range of an unsigned 32-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.uint32) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.uint32:
        return np.uint32(cast(int, json_object))


uint32_converter = UInt32Converter()


class Int64Converter(JsonConverter[int, np.int64]):
    def __init__(self) -> None:
        super().__init__(np.int64)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not a signed 64-bit integer: {value}")
        if value < INT64_MIN or value > INT64_MAX:
            raise ValueError(
                f"Value {value} is outside the range of a signed 64-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.int64) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.int64:
        return np.int64(cast(int, json_object))


int64_converter = Int64Converter()


class UInt64Converter(JsonConverter[int, np.uint64]):
    def __init__(self) -> None:
        super().__init__(np.uint64)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not an unsigned 64-bit integer: {value}")
        if value < 0 or value > UINT64_MAX:
            raise ValueError(
                f"Value {value} is outside the range of an unsigned 64-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.uint64) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.uint64:
        return np.uint64(cast(int, json_object))


uint64_converter = UInt64Converter()


class SizeConverter(JsonConverter[int, np.uint64]):
    def __init__(self) -> None:
        super().__init__(np.uint64)

    def to_json(self, value: int) -> object:
        if not isinstance(value, int):
            raise ValueError(f"Value in not an unsigned 64-bit integer: {value}")
        if value < 0 or value > UINT64_MAX:
            raise ValueError(
                f"Value {value} is outside the range of an unsigned 64-bit integer"
            )

        return value

    def numpy_to_json(self, value: np.uint64) -> object:
        return int(value)

    def from_json(self, json_object: object) -> int:
        return cast(int, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.uint64:
        return np.uint64(cast(int, json_object))


size_converter = SizeConverter()


class Float32Converter(JsonConverter[float, np.float32]):
    def __init__(self) -> None:
        super().__init__(np.float32)

    def to_json(self, value: float) -> object:
        if not isinstance(value, float):
            raise ValueError(f"Value in not a 32-bit float: {value}")

        return value

    def numpy_to_json(self, value: np.float32) -> object:
        return float(value)

    def from_json(self, json_object: object) -> float:
        return cast(float, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.float32:
        return np.float32(cast(float, json_object))


float32_converter = Float32Converter()


class Float64Converter(JsonConverter[float, np.float64]):
    def __init__(self) -> None:
        super().__init__(np.float64)

    def to_json(self, value: float) -> object:
        if not isinstance(value, float):
            raise ValueError(f"Value in not a 64-bit float: {value}")

        return value

    def numpy_to_json(self, value: np.float64) -> object:
        return float(value)

    def from_json(self, json_object: object) -> float:
        return cast(float, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.float64:
        return np.float64(cast(float, json_object))


float64_converter = Float64Converter()


class Complex32Converter(JsonConverter[complex, np.complex64]):
    def __init__(self) -> None:
        super().__init__(np.complex64)

    def to_json(self, value: complex) -> object:
        if not isinstance(value, complex):
            raise ValueError(f"Value in not a 32-bit complex value: {value}")

        return [value.real, value.imag]

    def numpy_to_json(self, value: np.complex64) -> object:
        return [float(value.real), float(value.imag)]

    def from_json(self, json_object: object) -> complex:
        if not isinstance(json_object, list) or len(json_object) != 2:
            raise ValueError(f"Expected a list of two floating-point numbers.")

        return complex(json_object[0], json_object[1])

    def from_json_to_numpy(self, json_object: object) -> np.complex64:
        return np.complex64(self.from_json(json_object))


complexfloat32_converter = Complex32Converter()


class Complex64Converter(JsonConverter[complex, np.complex128]):
    def __init__(self) -> None:
        super().__init__(np.complex128)

    def to_json(self, value: complex) -> object:
        if not isinstance(value, complex):
            raise ValueError(f"Value in not a 64-bit complex value: {value}")

        return [value.real, value.imag]

    def numpy_to_json(self, value: np.complex128) -> object:
        return [float(value.real), float(value.imag)]

    def from_json(self, json_object: object) -> complex:
        if not isinstance(json_object, list) or len(json_object) != 2:
            raise ValueError(f"Expected a list of two floating-point numbers.")

        return complex(json_object[0], json_object[1])

    def from_json_to_numpy(self, json_object: object) -> np.complex128:
        return np.complex128(self.from_json(json_object))


complexfloat64_converter = Complex64Converter()


class StringConverter(JsonConverter[str, np.object_]):
    def __init__(self) -> None:
        super().__init__(np.object_)

    def to_json(self, value: str) -> object:
        if not isinstance(value, str):
            raise ValueError(f"Value in not a string: {value}")
        return value

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(str, value))

    def from_json(self, json_object: object) -> str:
        return cast(str, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return np.object_(json_object)


string_converter = StringConverter()


class DateConverter(JsonConverter[datetime.date, np.datetime64]):
    def __init__(self) -> None:
        super().__init__(np.datetime64)

    def to_json(self, value: datetime.date) -> object:
        if not isinstance(value, datetime.date):
            raise ValueError(f"Value in not a date: {value}")
        return value.isoformat()

    def numpy_to_json(self, value: np.datetime64) -> object:
        return str(value.astype("datetime64[D]"))

    def from_json(self, json_object: object) -> datetime.date:
        return datetime.date.fromisoformat(cast(str, json_object))

    def from_json_to_numpy(self, json_object: object) -> np.datetime64:
        return np.datetime64(cast(str, json_object), "D")


date_converter = DateConverter()


class TimeConverter(JsonConverter[Time, np.timedelta64]):
    def __init__(self) -> None:
        super().__init__(np.timedelta64)

    def to_json(self, value: Time) -> object:
        if isinstance(value, Time):
            return str(value)
        elif isinstance(value, datetime.time):
            return value.isoformat()

        raise ValueError(f"Value in not a time: {value}")

    def numpy_to_json(self, value: np.timedelta64) -> object:
        return str(Time(value))

    def from_json(self, json_object: object) -> Time:
        return Time.parse(cast(str, json_object))

    def from_json_to_numpy(self, json_object: object) -> np.timedelta64:
        return self.from_json(json_object).numpy_value


time_converter = TimeConverter()


class DateTimeConverter(JsonConverter[DateTime, np.datetime64]):
    def __init__(self) -> None:
        super().__init__(np.datetime64)

    def to_json(self, value: DateTime) -> object:
        if isinstance(value, DateTime):
            return str(value)
        elif isinstance(value, datetime.datetime):
            return value.isoformat()

        raise ValueError(f"Value in not a datetime: {value}")

    def numpy_to_json(self, value: np.datetime64) -> object:
        return str(value)

    def from_json(self, json_object: object) -> DateTime:
        return DateTime.parse(cast(str, json_object))

    def from_json_to_numpy(self, json_object: object) -> np.datetime64:
        return self.from_json(json_object).numpy_value


datetime_converter = DateTimeConverter()

TEnum = TypeVar("TEnum", bound=OutOfRangeEnum)


class EnumConverter(Generic[TEnum, T_NP], JsonConverter[TEnum, T_NP]):
    def __init__(
        self,
        enum_type: type[TEnum],
        numpy_type: type,
        name_to_value: dict[str, TEnum],
        value_to_name: dict[TEnum, str],
    ) -> None:
        super().__init__(numpy_type)
        self._enum_type = enum_type
        self._name_to_value = name_to_value
        self._value_to_name = value_to_name

    def to_json(self, value: TEnum) -> object:
        if not isinstance(value, self._enum_type):
            raise ValueError(f"Value in not an enum or not the right type: {value}")
        if value.name == "":
            return value.value

        return self._value_to_name[value]

    def numpy_to_json(self, value: T_NP) -> object:
        return self.to_json(self._enum_type(value))

    def from_json(self, json_object: object) -> TEnum:
        if isinstance(json_object, int):
            return self._enum_type(json_object)

        return self._name_to_value[cast(str, json_object)]

    def from_json_to_numpy(self, json_object: object) -> T_NP:
        return self.from_json(json_object).value


TFlag = TypeVar("TFlag", bound=IntFlag)


class FlagsConverter(Generic[TFlag, T_NP], JsonConverter[TFlag, T_NP]):
    def __init__(
        self,
        enum_type: type[TFlag],
        numpy_type: type,
        name_to_value: dict[str, TFlag],
        value_to_name: dict[TFlag, str],
    ) -> None:
        super().__init__(numpy_type)
        self._enum_type = enum_type
        self._name_to_value = name_to_value
        self._value_to_name = value_to_name
        self._zero_enum = enum_type(0)
        self._zero_json = (
            [value_to_name[self._zero_enum]] if self._zero_enum in value_to_name else []
        )

    def to_json(self, value: TFlag) -> object:
        if not isinstance(value, self._enum_type):
            raise ValueError(f"Value in not an enum or not the right type: {value}")
        if value.value == 0:
            return self._zero_json

        remaining_int_value = value.value
        result: list[str] = []
        for enum_value in self._value_to_name:
            if enum_value.value == 0:
                continue
            if enum_value.value & remaining_int_value == enum_value.value:
                result.append(self._value_to_name[enum_value])
                remaining_int_value &= ~enum_value.value
                if remaining_int_value == 0:
                    break

        if remaining_int_value == 0:
            return result

        return value.value

    def numpy_to_json(self, value: T_NP) -> object:
        return self.to_json(self._enum_type(int(value)))  # type: ignore

    def from_json(self, json_object: object) -> TFlag:
        if isinstance(json_object, int):
            return self._enum_type(json_object)

        assert isinstance(json_object, list)
        res = self._zero_enum

        for name in json_object:
            res |= self._name_to_value[name]

        return res

    def from_json_to_numpy(self, json_object: object) -> T_NP:
        return self.from_json(json_object).value  # type: ignore


class OptionalConverter(Generic[T, T_NP], JsonConverter[Optional[T], np.void]):
    def __init__(self, element_converter: JsonConverter[T, T_NP]) -> None:
        super().__init__(
            np.dtype(
                [("has_value", np.bool_), ("value", element_converter.overall_dtype())]
            )
        )
        self._element_converter = element_converter
        self._none = cast(np.void, np.zeros((), dtype=self.overall_dtype())[()])

    def to_json(self, value: Optional[T]) -> object:
        if value is None:
            return None
        return self._element_converter.to_json(value)

    def numpy_to_json(self, value: np.void) -> object:
        if value["has_value"]:
            return self._element_converter.numpy_to_json(value["value"])
        return None

    def from_json(self, json_object: object) -> Optional[T]:
        if json_object is None:
            return None
        return self._element_converter.from_json(json_object)

    def from_json_to_numpy(self, json_object: object) -> np.void:
        if json_object is None:
            return self._none
        return (True, self._element_converter.from_json_to_numpy(json_object))  # type: ignore

    def supports_none(self) -> bool:
        return True


class UnionConverter(JsonConverter[T, np.object_]):
    def __init__(
        self,
        union_type: type,
        cases: list[Optional[tuple[type, JsonConverter[Any, Any], list[type]]]],
        simple: bool,
    ) -> None:
        super().__init__(np.object_)
        self._union_type = union_type
        self._cases = cases
        self._simple = simple
        self._offset = 1 if cases[0] is None else 0
        if self._simple:
            self._json_type_to_case_index = {
                json_type: case_index
                for (case_index, case) in enumerate(cases)
                if case is not None
                for json_type in case[2]
            }
        else:
            self.tag_to_case_index: dict[str, int] = {
                case[0].tag: case_index  # type: ignore
                for (case_index, case) in enumerate(cases)
                if case is not None
            }

    def to_json(self, value: T) -> object:
        if value is None:
            if self._cases[0] is None:
                return None
            else:
                raise ValueError("None is not a valid for this union type")

        if not isinstance(value, self._union_type):
            raise ValueError(f"Value in not a union or not the right type: {value}")

        tag_index = value.index + self._offset  # type: ignore
        inner_json_value = self._cases[tag_index][1].to_json(value.value)  # type: ignore

        if self._simple:
            return inner_json_value
        else:
            return {value.tag: inner_json_value}  # type: ignore

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(T, value))

    def from_json(self, json_object: object) -> T:
        if json_object is None:
            if self._cases[0] is None:
                return None  # type: ignore
            else:
                raise ValueError("None is not a valid for this union type")

        if self._simple:
            idx = self._json_type_to_case_index[type(json_object)]
            case = self._cases[idx]
            return case[0](case[1].from_json(json_object))  # type: ignore
        else:
            assert isinstance(json_object, dict)
            tag, inner_json_object = next(iter(json_object.items()))
            case = self._cases[self.tag_to_case_index[tag]]
            return case[0](case[1].from_json(inner_json_object))  # type: ignore

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return self.from_json(json_object)  # type: ignore

    def supports_none(self) -> bool:
        return self._cases[0] is None


class VectorConverter(Generic[T, T_NP], JsonConverter[list[T], np.object_]):
    def __init__(self, element_converter: JsonConverter[T, T_NP]) -> None:
        super().__init__(np.object_)
        self._element_converter = element_converter

    def to_json(self, value: list[T]) -> object:
        if not isinstance(value, list):
            raise ValueError(f"Value in not a list: {value}")
        return [self._element_converter.to_json(v) for v in value]

    def numpy_to_json(self, value: object) -> object:
        if isinstance(value, list):
            return [self._element_converter.to_json(v) for v in value]

        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value in not a list or ndarray: {value}")

        if value.ndim != 1:
            raise ValueError(f"Value in not a 1-dimensional ndarray: {value}")

        return [self._element_converter.numpy_to_json(v) for v in value]

    def from_json(self, json_object: object) -> list[T]:
        if not isinstance(json_object, list):
            raise ValueError(f"Value in not a list: {json_object}")
        return [self._element_converter.from_json(v) for v in json_object]

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return cast(np.object_, self.from_json(json_object))


class FixedVectorConverter(Generic[T, T_NP], JsonConverter[list[T], np.object_]):
    def __init__(self, element_converter: JsonConverter[T, T_NP], length: int) -> None:
        super().__init__(np.dtype((element_converter.overall_dtype(), length)))
        self._element_converter = element_converter
        self._length = length

    def to_json(self, value: list[T]) -> object:
        if not isinstance(value, list):
            raise ValueError(f"Value in not a list: {value}")
        if len(value) != self._length:
            raise ValueError(f"Value in not a list of length {self._length}: {value}")
        return [self._element_converter.to_json(v) for v in value]

    def numpy_to_json(self, value: np.object_) -> object:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value in not an ndarray: {value}")
        if value.shape != (self._length,):
            raise ValueError(f"Value does not have expected shape of {self._length}")

        return [self._element_converter.numpy_to_json(v) for v in value]

    def from_json(self, json_object: object) -> list[T]:
        if not isinstance(json_object, list):
            raise ValueError(f"Value in not a list: {json_object}")
        if len(json_object) != self._length:
            raise ValueError(
                f"Value in not a list of length {self._length}: {json_object}"
            )
        return [self._element_converter.from_json(v) for v in json_object]

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        if not isinstance(json_object, list):
            raise ValueError(f"Value in not a list: {json_object}")
        if len(json_object) != self._length:
            raise ValueError(
                f"Value in not a list of length {self._length}: {json_object}"
            )
        return cast(
            np.object_,
            [self._element_converter.from_json_to_numpy(v) for v in json_object],
        )


TKey = TypeVar("TKey")
TKey_NP = TypeVar("TKey_NP", bound=np.generic)
TValue = TypeVar("TValue")
TValue_NP = TypeVar("TValue_NP", bound=np.generic)


class MapConverter(
    Generic[TKey, TKey_NP, TValue, TValue_NP],
    JsonConverter[dict[TKey, TValue], np.object_],
):
    def __init__(
        self,
        key_converter: JsonConverter[TKey, TKey_NP],
        value_converter: JsonConverter[TValue, TValue_NP],
    ) -> None:
        super().__init__(np.object_)
        self._key_converter = key_converter
        self._value_converter = value_converter

    def to_json(self, value: dict[TKey, TValue]) -> object:
        if not isinstance(value, dict):
            raise ValueError(f"Value in not a dict: {value}")

        if isinstance(self._key_converter, StringConverter):
            return {
                cast(str, k): self._value_converter.to_json(v) for k, v in value.items()
            }

        return [
            [self._key_converter.to_json(k), self._value_converter.to_json(v)]
            for k, v in value.items()
        ]

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(dict[TKey, TValue], value))

    def from_json(self, json_object: object) -> dict[TKey, TValue]:
        if isinstance(self._key_converter, StringConverter):
            if not isinstance(json_object, dict):
                raise ValueError(f"Value in not a dict: {json_object}")

            return {
                cast(TKey, k): self._value_converter.from_json(v)
                for k, v in json_object.items()
            }

        if not isinstance(json_object, list):
            raise ValueError(f"Value in not a list: {json_object}")

        return {
            self._key_converter.from_json(k): self._value_converter.from_json(v)
            for [k, v] in json_object
        }

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return cast(np.object_, self.from_json(json_object))


class NDArrayConverterBase(
    Generic[T, T_NP], JsonConverter[npt.NDArray[Any], np.object_]
):
    def __init__(
        self,
        overall_dtype: npt.DTypeLike,
        element_converter: JsonConverter[T, T_NP],
        dtype: npt.DTypeLike,
    ) -> None:
        super().__init__(overall_dtype)
        self._element_converter = element_converter

        (
            self._array_dtype,
            self._subarray_shape,
        ) = NDArrayConverterBase._get_dtype_and_subarray_shape(
            dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        )
        if self._subarray_shape == ():
            self._subarray_shape = None

    @staticmethod
    def _get_dtype_and_subarray_shape(
        dtype: np.dtype[Any],
    ) -> tuple[np.dtype[Any], tuple[int, ...]]:
        if dtype.subdtype is None:
            return dtype, ()
        subres = NDArrayConverterBase._get_dtype_and_subarray_shape(dtype.subdtype[0])
        return (subres[0], dtype.subdtype[1] + subres[1])

    def check_dtype(self, input_dtype: npt.DTypeLike):
        if input_dtype != self._array_dtype:
            # see if it's the same dtype but packed, not aligned
            packed_dtype = recfunctions.repack_fields(self._array_dtype, align=False, recurse=True)  # type: ignore
            if packed_dtype != input_dtype:
                if packed_dtype == self._array_dtype:
                    message = f"Expected dtype {self._array_dtype}, got {input_dtype}"
                else:
                    message = f"Expected dtype {self._array_dtype} or {packed_dtype}, got {input_dtype}"

                raise ValueError(message)

    def _read(
        self, shape: tuple[int, ...], json_object: list[object]
    ) -> npt.NDArray[Any]:
        subarray_shape_not_none = (
            () if self._subarray_shape is None else self._subarray_shape
        )

        partially_flattened_shape = (np.prod(shape),) + subarray_shape_not_none  # type: ignore
        result = np.ndarray(partially_flattened_shape, dtype=self._array_dtype)
        for i in range(partially_flattened_shape[0]):
            result[i] = self._element_converter.from_json_to_numpy(json_object[i])

        return result.reshape(shape + subarray_shape_not_none)


class FixedNDArrayConverter(Generic[T, T_NP], NDArrayConverterBase[T, T_NP]):
    def __init__(
        self,
        element_converter: JsonConverter[T, T_NP],
        shape: tuple[int, ...],
    ) -> None:
        dtype = element_converter.overall_dtype()
        super().__init__(np.dtype((dtype, shape)), element_converter, dtype)
        self._shape = shape

    def to_json(self, value: npt.NDArray[Any]) -> object:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value in not an ndarray: {value}")

        self.check_dtype(value.dtype)

        required_shape = (
            self._shape
            if self._subarray_shape is None
            else self._shape + self._subarray_shape
        )

        if value.shape != required_shape:
            raise ValueError(f"Expected shape {required_shape}, got {value.shape}")

        if self._subarray_shape is None:
            return [self._element_converter.numpy_to_json(v) for v in value.flat]

        reshaped = value.reshape((-1,) + self._subarray_shape)
        return [self._element_converter.numpy_to_json(v) for v in reshaped]

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(npt.NDArray[Any], value))

    def from_json(self, json_object: object) -> npt.NDArray[Any]:
        if not isinstance(json_object, list):
            raise ValueError(f"Value in not a list: {json_object}")

        return self._read(self._shape, json_object)

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return cast(np.object_, self.from_json(json_object))


class DynamicNDArrayConverter(NDArrayConverterBase[T, T_NP]):
    def __init__(
        self,
        element_serializer: JsonConverter[T, T_NP],
    ) -> None:
        super().__init__(
            np.object_, element_serializer, element_serializer.overall_dtype()
        )

    def to_json(self, value: npt.NDArray[Any]) -> object:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value in not an ndarray: {value}")

        self.check_dtype(value.dtype)

        if self._subarray_shape is None:
            return {
                "shape": value.shape,
                "data": [self._element_converter.numpy_to_json(v) for v in value.flat],
            }

        if len(value.shape) < len(self._subarray_shape) or (
            value.shape[-len(self._subarray_shape) :] != self._subarray_shape
        ):
            raise ValueError(
                f"The array is required to have shape (..., {(', '.join((str(i) for i in self._subarray_shape)))})"
            )

        reshaped = value.reshape((-1,) + self._subarray_shape)
        return {
            "shape": value.shape[: -len(self._subarray_shape)],
            "data": [self._element_converter.numpy_to_json(v) for v in reshaped],
        }

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(npt.NDArray[Any], value))

    def from_json(self, json_object: object) -> npt.NDArray[Any]:
        if not isinstance(json_object, dict):
            raise ValueError(f"Value in not a dict: {json_object}")

        if "shape" not in json_object or "data" not in json_object:
            raise ValueError(f"Value in not a dict with shape and data: {json_object}")

        shape = tuple(json_object["shape"])
        data = json_object["data"]

        return self._read(shape, data)

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return cast(np.object_, self.from_json(json_object))


class NDArrayConverter(Generic[T, T_NP], NDArrayConverterBase[T, T_NP]):
    def __init__(
        self,
        element_converter: JsonConverter[T, T_NP],
        ndims: int,
    ) -> None:
        super().__init__(
            np.object_, element_converter, element_converter.overall_dtype()
        )
        self._ndims = ndims

    def to_json(self, value: npt.NDArray[Any]) -> object:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value in not an ndarray: {value}")

        self.check_dtype(value.dtype)

        if self._subarray_shape is None:
            if value.ndim != self._ndims:
                raise ValueError(f"Expected {self._ndims} dimensions, got {value.ndim}")

            return {
                "shape": value.shape,
                "data": [self._element_converter.numpy_to_json(v) for v in value.flat],
            }

        total_dims = len(self._subarray_shape) + self._ndims
        if value.ndim != total_dims:
            raise ValueError(f"Expected {total_dims} dimensions, got {value.ndim}")

        if value.shape[-len(self._subarray_shape) :] != self._subarray_shape:
            raise ValueError(
                f"The array is required to have shape (..., {(', '.join((str(i) for i in self._subarray_shape)))})"
            )

        reshaped = value.reshape((-1,) + self._subarray_shape)
        return {
            "shape": value.shape[: -len(self._subarray_shape)],
            "data": [self._element_converter.numpy_to_json(v) for v in reshaped],
        }

    def numpy_to_json(self, value: np.object_) -> object:
        return self.to_json(cast(npt.NDArray[Any], value))

    def from_json(self, json_object: object) -> npt.NDArray[Any]:
        if not isinstance(json_object, dict):
            raise ValueError(f"Value in not a dict: {json_object}")

        if "shape" not in json_object or "data" not in json_object:
            raise ValueError(f"Value in not a dict with shape and data: {json_object}")

        shape = tuple(json_object["shape"])
        data = json_object["data"]

        return self._read(shape, data)

    def from_json_to_numpy(self, json_object: object) -> np.object_:
        return cast(np.object_, self.from_json(json_object))
