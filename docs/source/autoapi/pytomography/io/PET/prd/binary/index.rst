:py:mod:`pytomography.io.PET.prd.binary`
========================================

.. py:module:: pytomography.io.PET.prd.binary


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.binary.BinaryPrdExperimentWriter
   pytomography.io.PET.prd.binary.BinaryPrdExperimentReader
   pytomography.io.PET.prd.binary._CoincidenceEventSerializer
   pytomography.io.PET.prd.binary._SubjectSerializer
   pytomography.io.PET.prd.binary._InstitutionSerializer
   pytomography.io.PET.prd.binary._ExamInformationSerializer
   pytomography.io.PET.prd.binary._DetectorSerializer
   pytomography.io.PET.prd.binary._ScannerInformationSerializer
   pytomography.io.PET.prd.binary._HeaderSerializer
   pytomography.io.PET.prd.binary._TimeBlockSerializer
   pytomography.io.PET.prd.binary._TimeIntervalSerializer
   pytomography.io.PET.prd.binary._TimeFrameInformationSerializer




.. py:class:: BinaryPrdExperimentWriter(stream)

   Bases: :py:obj:`pytomography.io.PET.prd._binary.BinaryProtocolWriter`, :py:obj:`pytomography.io.PET.prd.protocols.PrdExperimentWriterBase`

   Binary writer for the PrdExperiment protocol.

   .. py:method:: _write_header(value)


   .. py:method:: _write_time_blocks(value)



.. py:class:: BinaryPrdExperimentReader(stream)

   Bases: :py:obj:`pytomography.io.PET.prd._binary.BinaryProtocolReader`, :py:obj:`pytomography.io.PET.prd.protocols.PrdExperimentReaderBase`

   Binary writer for the PrdExperiment protocol.

   .. py:method:: _read_header()


   .. py:method:: _read_time_blocks()



.. py:class:: _CoincidenceEventSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.CoincidenceEvent`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _SubjectSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Subject`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _InstitutionSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Institution`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _ExamInformationSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.ExamInformation`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _DetectorSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Detector`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _ScannerInformationSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.ScannerInformation`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _HeaderSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.Header`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _TimeBlockSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeBlock`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _TimeIntervalSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeInterval`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



.. py:class:: _TimeFrameInformationSerializer

   Bases: :py:obj:`pytomography.io.PET.prd._binary.RecordSerializer`\ [\ :py:obj:`pytomography.io.PET.prd.protocols.TimeFrameInformation`\ ]

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

   .. py:method:: write(stream, value)


   .. py:method:: write_numpy(stream, value)


   .. py:method:: read(stream)



