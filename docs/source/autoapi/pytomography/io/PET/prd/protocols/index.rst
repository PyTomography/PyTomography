:py:mod:`pytomography.io.PET.prd.protocols`
===========================================

.. py:module:: pytomography.io.PET.prd.protocols


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.protocols.PrdExperimentWriterBase
   pytomography.io.PET.prd.protocols.PrdExperimentReaderBase




.. py:class:: PrdExperimentWriterBase

   Bases: :py:obj:`abc.ABC`

   Abstract writer for the PrdExperiment protocol.

   .. py:attribute:: schema
      :value: '{"protocol":{"name":"PrdExperiment","sequence":[{"name":"header","type":"Prd.Header"},{"name":"ti...'

      

   .. py:method:: __enter__()


   .. py:method:: __exit__(exc_type, exc, traceback)


   .. py:method:: write_header(value)

      Ordinal 0


   .. py:method:: write_time_blocks(value)

      Ordinal 1


   .. py:method:: _write_header(value)
      :abstractmethod:


   .. py:method:: _write_time_blocks(value)
      :abstractmethod:


   .. py:method:: close()
      :abstractmethod:


   .. py:method:: _end_stream()
      :abstractmethod:


   .. py:method:: _raise_unexpected_state(actual)


   .. py:method:: _state_to_method_name(state)



.. py:class:: PrdExperimentReaderBase

   Bases: :py:obj:`abc.ABC`

   Abstract reader for the PrdExperiment protocol.

   .. py:attribute:: schema

      

   .. py:attribute:: T

      

   .. py:method:: __enter__()


   .. py:method:: __exit__(exc_type, exc, traceback)


   .. py:method:: close()
      :abstractmethod:


   .. py:method:: read_header()

      Ordinal 0


   .. py:method:: read_time_blocks()

      Ordinal 1


   .. py:method:: copy_to(writer)


   .. py:method:: _read_header()
      :abstractmethod:


   .. py:method:: _read_time_blocks()
      :abstractmethod:


   .. py:method:: _wrap_iterable(iterable, final_state)


   .. py:method:: _raise_unexpected_state(actual)


   .. py:method:: _state_to_method_name(state)



