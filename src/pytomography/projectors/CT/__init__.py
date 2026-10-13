"""System matrices of CT scans, which project with parallelproj 2 (Joseph's method).

* :class:`~pytomography.projectors.CT.CTGen3SystemMatrix`: third generation (fan beam, cylindrical detector) scans,
  helical or axial, with :class:`~pytomography.metadata.CT.CTGen3ProjMeta`.
* :class:`~pytomography.projectors.CT.CTConeBeamFlatPanelSystemMatrix`: cone-beam scans with a flat panel detector,
  with :class:`~pytomography.metadata.CT.CTConeBeamFlatPanelProjMeta`.

Both also reconstruct by filtered back projection through :class:`~pytomography.algorithms.FilteredBackProjection`.
Importing this package raises if parallelproj 2 (its ``parallelproj_core`` module) is not installed.
"""
try:
    import parallelproj_core
except ImportError:
    raise Exception(
        'CT functionality in PyTomography requires parallelproj 2, which provides the parallelproj_core module. '
        'It is distributed through conda-forge:\n'
        '    conda install -c conda-forge parallelproj\n'
        '(choose a libparallelproj build matching your CUDA version, e.g. libparallelproj=*=cuda130*). '
        'See https://parallelproj.readthedocs.io/en/stable/'
    )
from .ct_conebeam_flatpanel_system_matrix import CTConeBeamFlatPanelSystemMatrix
from .ct_gen3_system_matrix import CTGen3SystemMatrix