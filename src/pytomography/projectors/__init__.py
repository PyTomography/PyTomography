r"""This module contains classes/functionality for operators that map between distinct vector spaces. One (very important) operator of this form is the system matrix :math:`H:\mathbb{U} \to \mathbb{V}`, which maps from object space :math:`\mathbb{U}` to image space :math:`\mathbb{V}`"""
from .system_matrix import SystemMatrix, ExtendedSystemMatrix
from .SPECT import SPECTSystemMatrix, SPECTSystemMatrixMaskedSegments
from .PET import PETLMSystemMatrix
from .shared import KEMSystemMatrix, MotionSystemMatrix
