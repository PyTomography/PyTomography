r"""System matrices: the operators :math:`H:\mathbb{U} \to \mathbb{V}` from object space to projection space.

Each system matrix projects an object (``forward``) and back projects projections (``backward``), with the transforms
(:mod:`pytomography.transforms`) that model the physics. Build one from the ``object_meta`` and ``proj_meta`` a
reader returns:

* :mod:`pytomography.projectors.SPECT`: :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix` (parallel-hole
  cameras) and :class:`~pytomography.projectors.SPECT.StarGuideSystemMatrix`;
* :mod:`pytomography.projectors.PET`: :class:`~pytomography.projectors.PET.PETLMSystemMatrix` (list mode) and
  :class:`~pytomography.projectors.PET.PETSinogramSystemMatrix`;
* :mod:`pytomography.projectors.CT`: :class:`~pytomography.projectors.CT.CTGen3SystemMatrix` (helical) and
  :class:`~pytomography.projectors.CT.CTConeBeamFlatPanelSystemMatrix`;
* :mod:`pytomography.projectors.shared`: :class:`~pytomography.projectors.shared.KEMSystemMatrix` (kernel method)
  and :class:`~pytomography.projectors.shared.MotionSystemMatrix` (motion correction).

:class:`SystemMatrix` is the base class of all of them, and :class:`ExtendedSystemMatrix` stacks several.
"""
from .system_matrix import SystemMatrix, ExtendedSystemMatrix