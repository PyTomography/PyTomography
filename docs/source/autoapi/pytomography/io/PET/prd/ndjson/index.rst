:py:mod:`pytomography.io.PET.prd.ndjson`
========================================

.. py:module:: pytomography.io.PET.prd.ndjson


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.ndjson._CoincidenceEventConverter
   pytomography.io.PET.prd.ndjson._SubjectConverter
   pytomography.io.PET.prd.ndjson._InstitutionConverter
   pytomography.io.PET.prd.ndjson._ExamInformationConverter
   pytomography.io.PET.prd.ndjson._DetectorConverter
   pytomography.io.PET.prd.ndjson._ScannerInformationConverter
   pytomography.io.PET.prd.ndjson._HeaderConverter
   pytomography.io.PET.prd.ndjson._TimeBlockConverter
   pytomography.io.PET.prd.ndjson._TimeIntervalConverter
   pytomography.io.PET.prd.ndjson._TimeFrameInformationConverter
   pytomography.io.PET.prd.ndjson.NDJsonPrdExperimentWriter
   pytomography.io.PET.prd.ndjson.NDJsonPrdExperimentReader




.. py:class:: _CoincidenceEventConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.CoincidenceEvent`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _SubjectConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Subject`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _InstitutionConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Institution`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _ExamInformationConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.ExamInformation`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _DetectorConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Detector`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _ScannerInformationConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.ScannerInformation`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _HeaderConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Header`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _TimeBlockConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeBlock`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _TimeIntervalConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeInterval`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: _TimeFrameInformationConverter

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.JsonConverter`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeFrameInformation`\ , :py:obj:`pytomography.io.PET.prd.protocols.np.void`\ ]

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



.. py:class:: NDJsonPrdExperimentWriter(stream)

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.NDJsonProtocolWriter`, :py:obj:`pytomography.io.PET.prd.protocols.PrdExperimentWriterBase`

   NDJson writer for the PrdExperiment protocol.

   .. py:method:: _write_header(value)


   .. py:method:: _write_time_blocks(value)



.. py:class:: NDJsonPrdExperimentReader(stream)

   Bases: :py:obj:`pytomography.io.PET.prd._ndjson.NDJsonProtocolReader`, :py:obj:`pytomography.io.PET.prd.protocols.PrdExperimentReaderBase`

   NDJson writer for the PrdExperiment protocol.

   .. py:method:: _read_header()


   .. py:method:: _read_time_blocks()



