:py:mod:`pytomography.io.PET.prd.types`
=======================================

.. py:module:: pytomography.io.PET.prd.types


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.types.CoincidenceEvent
   pytomography.io.PET.prd.types.Subject
   pytomography.io.PET.prd.types.Institution
   pytomography.io.PET.prd.types.ExamInformation
   pytomography.io.PET.prd.types.Detector
   pytomography.io.PET.prd.types.ScannerInformation
   pytomography.io.PET.prd.types.Header
   pytomography.io.PET.prd.types.TimeBlock
   pytomography.io.PET.prd.types.TimeInterval
   pytomography.io.PET.prd.types.TimeFrameInformation



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.types._mk_get_dtype



Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.prd.types.get_dtype


.. py:class:: CoincidenceEvent(*, detector_1_id = 0, detector_2_id = 0, tof_idx = 0, energy_1_idx = 0, energy_2_idx = 0)

   All information about a coincidence event specified as identifiers or indices (i.e. discretized).
   TODO: this might take up too much space, so some/all of these could be combined in a single index if necessary.

   .. py:attribute:: detector_1_id
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: detector_2_id
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: tof_idx
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: energy_1_idx
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: energy_2_idx
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: Subject(*, name = None, id = '')

   .. py:attribute:: name
      :type: Optional[str]

      

   .. py:attribute:: id
      :type: str

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: Institution(*, name = '', address = '')

   .. py:attribute:: name
      :type: str

      

   .. py:attribute:: address
      :type: str

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: ExamInformation(*, subject = None, institution = None, protocol = None, start_of_acquisition = None)

   Items describing the exam (incomplete)

   .. py:attribute:: subject
      :type: Subject

      

   .. py:attribute:: institution
      :type: Institution

      

   .. py:attribute:: protocol
      :type: Optional[str]

      

   .. py:attribute:: start_of_acquisition
      :type: Optional[pytomography.io.PET.prd.yardl_types.DateTime]

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: Detector(*, id = 0, x = 0.0, y = 0.0, z = 0.0)

   Detector ID and location. Units are in mm
   TODO: this is currently just a sample implementation with "point" detectors.
   We plan to have full shape information here.

   .. py:attribute:: id
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: x
      :type: pytomography.io.PET.prd.yardl_types.Float32

      

   .. py:attribute:: y
      :type: pytomography.io.PET.prd.yardl_types.Float32

      

   .. py:attribute:: z
      :type: pytomography.io.PET.prd.yardl_types.Float32

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: ScannerInformation(*, model_name = None, detectors = None, tof_bin_edges = None, tof_resolution = 0.0, energy_bin_edges = None, energy_resolution_at_511 = 0.0, listmode_time_block_duration = 0)

   .. py:attribute:: model_name
      :type: Optional[str]

      

   .. py:attribute:: detectors
      :type: list[Detector]

      

   .. py:attribute:: tof_bin_edges
      :type: numpy.typing.NDArray[numpy.float32]

      edge information for TOF bins in mm (given as from first to last edge, so there is one more edge than the number of bins)
      TODO: this currently assumes equal size for each TOF bin, but some scanners "stretch" TOF bins depending on length of LOR

   .. py:attribute:: tof_resolution
      :type: pytomography.io.PET.prd.yardl_types.Float32

      TOF resolution in mm

   .. py:attribute:: energy_bin_edges
      :type: numpy.typing.NDArray[numpy.float32]

      edge information for energy windows in keV (given as from first to last edge, so there is one more edge than the number of bins)

   .. py:attribute:: energy_resolution_at_511
      :type: pytomography.io.PET.prd.yardl_types.Float32

      FWHM of photopeak for incoming gamma of 511 keV, expressed as a ratio w.r.t. 511

   .. py:attribute:: listmode_time_block_duration
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      duration of each time block in ms

   .. py:method:: number_of_detectors()


   .. py:method:: number_of_tof_bins()


   .. py:method:: number_of_energy_bins()


   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: Header(*, scanner = None, exam = None)

   .. py:attribute:: scanner
      :type: ScannerInformation

      

   .. py:attribute:: exam
      :type: Optional[ExamInformation]

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: TimeBlock(*, id = 0, prompt_events = None, delayed_events = None)

   .. py:attribute:: id
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      number of the block. Multiply with listmodeTimeBlockDuration to get time since startOfAcquisition

   .. py:attribute:: prompt_events
      :type: list[CoincidenceEvent]

      list of prompts in this time block
      TODO might be better to use !array

   .. py:attribute:: delayed_events
      :type: Optional[list[CoincidenceEvent]]

      list of delayed coincidences in this time block

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: TimeInterval(*, start = 0, stop = 0)

   Time interval in milliseconds since start of acquisition

   .. py:attribute:: start
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:attribute:: stop
      :type: pytomography.io.PET.prd.yardl_types.UInt32

      

   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: TimeFrameInformation(*, time_frames = None)

   A sequence of time intervals (could be consecutive)

   .. py:attribute:: time_frames
      :type: list[TimeInterval]

      

   .. py:method:: number_of_time_frames()


   .. py:method:: __eq__(other)

      Return self==value.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).



.. py:function:: _mk_get_dtype()


.. py:data:: get_dtype

   

