"""System matrices built on another system matrix, for any modality.

* :class:`KEMSystemMatrix`: the kernel method, an image represented as a kernel matrix times coefficients;
* :class:`MotionSystemMatrix`: one system matrix and motion transform per gate, for motion correction.
"""
from .kem_system_matrix import KEMSystemMatrix
from .motion_correction_system_matrix import MotionSystemMatrix