:py:mod:`pytomography.io.PET.prd._ndjson`
=========================================

.. py:module:: pytomography.io.PET.prd._ndjson


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd._ndjson.NDJsonProtocolWriter
   pytomography.io.PET.prd._ndjson.NDJsonProtocolReader
   pytomography.io.PET.prd._ndjson.JsonConverter
   pytomography.io.PET.prd._ndjson.BoolConverter
   pytomography.io.PET.prd._ndjson.Int8Converter
   pytomography.io.PET.prd._ndjson.UInt8Converter
   pytomography.io.PET.prd._ndjson.Int16Converter
   pytomography.io.PET.prd._ndjson.UInt16Converter
   pytomography.io.PET.prd._ndjson.Int32Converter
   pytomography.io.PET.prd._ndjson.UInt32Converter
   pytomography.io.PET.prd._ndjson.Int64Converter
   pytomography.io.PET.prd._ndjson.UInt64Converter
   pytomography.io.PET.prd._ndjson.SizeConverter
   pytomography.io.PET.prd._ndjson.Float32Converter
   pytomography.io.PET.prd._ndjson.Float64Converter
   pytomography.io.PET.prd._ndjson.Complex32Converter
   pytomography.io.PET.prd._ndjson.Complex64Converter
   pytomography.io.PET.prd._ndjson.StringConverter
   pytomography.io.PET.prd._ndjson.DateConverter
   pytomography.io.PET.prd._ndjson.TimeConverter
   pytomography.io.PET.prd._ndjson.DateTimeConverter
   pytomography.io.PET.prd._ndjson.EnumConverter
   pytomography.io.PET.prd._ndjson.FlagsConverter
   pytomography.io.PET.prd._ndjson.OptionalConverter
   pytomography.io.PET.prd._ndjson.UnionConverter
   pytomography.io.PET.prd._ndjson.VectorConverter
   pytomography.io.PET.prd._ndjson.FixedVectorConverter
   pytomography.io.PET.prd._ndjson.MapConverter
   pytomography.io.PET.prd._ndjson.NDArrayConverterBase
   pytomography.io.PET.prd._ndjson.FixedNDArrayConverter
   pytomography.io.PET.prd._ndjson.DynamicNDArrayConverter
   pytomography.io.PET.prd._ndjson.NDArrayConverter




Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd._ndjson.CURRENT_NDJSON_FORMAT_VERSION
   pytomography.io.PET.prd._ndjson.INT8_MIN
   pytomography.io.PET.prd._ndjson.INT8_MAX
   pytomography.io.PET.prd._ndjson.UINT8_MAX
   pytomography.io.PET.prd._ndjson.INT16_MIN
   pytomography.io.PET.prd._ndjson.INT16_MAX
   pytomography.io.PET.prd._ndjson.UINT16_MAX
   pytomography.io.PET.prd._ndjson.INT32_MIN
   pytomography.io.PET.prd._ndjson.INT32_MAX
   pytomography.io.PET.prd._ndjson.UINT32_MAX
   pytomography.io.PET.prd._ndjson.INT64_MIN
   pytomography.io.PET.prd._ndjson.INT64_MAX
   pytomography.io.PET.prd._ndjson.UINT64_MAX
   pytomography.io.PET.prd._ndjson.MISSING_SENTINEL
   pytomography.io.PET.prd._ndjson.T
   pytomography.io.PET.prd._ndjson.T_NP
   pytomography.io.PET.prd._ndjson.bool_converter
   pytomography.io.PET.prd._ndjson.int8_converter
   pytomography.io.PET.prd._ndjson.uint8_converter
   pytomography.io.PET.prd._ndjson.int16_converter
   pytomography.io.PET.prd._ndjson.uint16_converter
   pytomography.io.PET.prd._ndjson.int32_converter
   pytomography.io.PET.prd._ndjson.uint32_converter
   pytomography.io.PET.prd._ndjson.int64_converter
   pytomography.io.PET.prd._ndjson.uint64_converter
   pytomography.io.PET.prd._ndjson.size_converter
   pytomography.io.PET.prd._ndjson.float32_converter
   pytomography.io.PET.prd._ndjson.float64_converter
   pytomography.io.PET.prd._ndjson.complexfloat32_converter
   pytomography.io.PET.prd._ndjson.complexfloat64_converter
   pytomography.io.PET.prd._ndjson.string_converter
   pytomography.io.PET.prd._ndjson.date_converter
   pytomography.io.PET.prd._ndjson.time_converter
   pytomography.io.PET.prd._ndjson.datetime_converter
   pytomography.io.PET.prd._ndjson.TEnum
   pytomography.io.PET.prd._ndjson.TFlag
   pytomography.io.PET.prd._ndjson.TKey
   pytomography.io.PET.prd._ndjson.TKey_NP
   pytomography.io.PET.prd._ndjson.TValue
   pytomography.io.PET.prd._ndjson.TValue_NP


.. py:data:: CURRENT_NDJSON_FORMAT_VERSION
   :type: int
   :value: 1

   

.. py:data:: INT8_MIN
   :type: int

   

.. py:data:: INT8_MAX
   :type: int

   

.. py:data:: UINT8_MAX
   :type: int

   

.. py:data:: INT16_MIN
   :type: int

   

.. py:data:: INT16_MAX
   :type: int

   

.. py:data:: UINT16_MAX
   :type: int

   

.. py:data:: INT32_MIN
   :type: int

   

.. py:data:: INT32_MAX
   :type: int

   

.. py:data:: UINT32_MAX
   :type: int

   

.. py:data:: INT64_MIN
   :type: int

   

.. py:data:: INT64_MAX
   :type: int

   

.. py:data:: UINT64_MAX
   :type: int

   

.. py:data:: MISSING_SENTINEL

   

.. py:class:: NDJsonProtocolWriter(stream, schema)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: close()


   .. py:method:: _end_stream()


   .. py:method:: _write_json_line(value)



.. py:class:: NDJsonProtocolReader(stream, schema)

   .. py:method:: close()


   .. py:method:: _read_json_line(stepName, required)



.. py:data:: T

   

.. py:data:: T_NP

   

.. py:class:: JsonConverter(dtype)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`pytomography.io.PET.prd.yardl_types.ABC`

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: overall_dtype()


   .. py:method:: to_json(value)
      :abstractmethod:


   .. py:method:: numpy_to_json(value)
      :abstractmethod:


   .. py:method:: from_json(json_object)
      :abstractmethod:


   .. py:method:: from_json_to_numpy(json_object)
      :abstractmethod:


   .. py:method:: supports_none()



.. py:class:: BoolConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`bool`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.bool_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: bool_converter

   

.. py:class:: Int8Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.int8`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: int8_converter

   

.. py:class:: UInt8Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.uint8`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: uint8_converter

   

.. py:class:: Int16Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.int16`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: int16_converter

   

.. py:class:: UInt16Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.uint16`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: uint16_converter

   

.. py:class:: Int32Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.int32`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: int32_converter

   

.. py:class:: UInt32Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.uint32`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: uint32_converter

   

.. py:class:: Int64Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.int64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: int64_converter

   

.. py:class:: UInt64Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.uint64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: uint64_converter

   

.. py:class:: SizeConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`int`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.uint64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: size_converter

   

.. py:class:: Float32Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`float`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.float32`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: float32_converter

   

.. py:class:: Float64Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`float`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.float64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: float64_converter

   

.. py:class:: Complex32Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`complex`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.complex64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: complexfloat32_converter

   

.. py:class:: Complex64Converter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`complex`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.complex128`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: complexfloat64_converter

   

.. py:class:: StringConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`str`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: string_converter

   

.. py:class:: DateConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.yardl_types.datetime.date`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.datetime64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: date_converter

   

.. py:class:: TimeConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.yardl_types.Time`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.timedelta64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: time_converter

   

.. py:class:: DateTimeConverter

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.yardl_types.DateTime`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.datetime64`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: datetime_converter

   

.. py:data:: TEnum

   

.. py:class:: EnumConverter(enum_type, numpy_type, name_to_value, value_to_name)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`TEnum`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`TEnum`\ , :py:obj:`T_NP`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: TFlag

   

.. py:class:: FlagsConverter(enum_type, numpy_type, name_to_value, value_to_name)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`TFlag`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`TFlag`\ , :py:obj:`T_NP`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:class:: OptionalConverter(element_converter)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`Optional`\ [\ :py:obj:`T`\ ]\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.void`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)


   .. py:method:: supports_none()



.. py:class:: UnionConverter(union_type, cases, simple)

   Bases: :py:obj:`JsonConverter`\ [\ :py:obj:`T`\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)


   .. py:method:: supports_none()



.. py:class:: VectorConverter(element_converter)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`list`\ [\ :py:obj:`T`\ ]\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:class:: FixedVectorConverter(element_converter, length)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`list`\ [\ :py:obj:`T`\ ]\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:data:: TKey

   

.. py:data:: TKey_NP

   

.. py:data:: TValue

   

.. py:data:: TValue_NP

   

.. py:class:: MapConverter(key_converter, value_converter)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`TKey`\ , :py:obj:`TKey_NP`\ , :py:obj:`TValue`\ , :py:obj:`TValue_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`dict`\ [\ :py:obj:`TKey`\ , :py:obj:`TValue`\ ]\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:class:: NDArrayConverterBase(overall_dtype, element_converter, dtype)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`JsonConverter`\ [\ :py:obj:`numpy.typing.NDArray`\ [\ :py:obj:`Any`\ ]\ , :py:obj:`pytomography.io.PET.prd.yardl_types.np.object_`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: _get_dtype_and_subarray_shape(dtype)
      :staticmethod:


   .. py:method:: check_dtype(input_dtype)


   .. py:method:: _read(shape, json_object)



.. py:class:: FixedNDArrayConverter(element_converter, shape)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`NDArrayConverterBase`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:class:: DynamicNDArrayConverter(element_serializer)

   Bases: :py:obj:`NDArrayConverterBase`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



.. py:class:: NDArrayConverter(element_converter, ndims)

   Bases: :py:obj:`pytomography.io.PET.prd.yardl_types.Generic`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ], :py:obj:`NDArrayConverterBase`\ [\ :py:obj:`T`\ , :py:obj:`T_NP`\ ]

   Abstract base class for generic types.

   A generic type is typically declared by inheriting from
   this class parameterized with one or more type variables.
   For example, a generic mapping type might be defined as::

     class Mapping(Generic[KT, VT]):
         def __getitem__(self, key: KT) -> VT:
             ...
         # Etc.

   This class can then be used as follows::

     def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
         try:
             return mapping[key]
         except KeyError:
             return default

   .. py:method:: to_json(value)


   .. py:method:: numpy_to_json(value)


   .. py:method:: from_json(json_object)


   .. py:method:: from_json_to_numpy(json_object)



