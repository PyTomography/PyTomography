"""PET system matrices, which project with parallelproj 2 (conda-forge's ``parallelproj`` package).

* :class:`PETLMSystemMatrix`: list mode, with or without time of flight;
* :class:`PETSinogramSystemMatrix`: sinograms of a polygonal scanner, with or without time of flight;
* :func:`create_sinogramSM_from_LMSM`: the sinogram system matrix of the scanner of a list mode one.

Importing this package raises an error that says how to install parallelproj when it is missing.
"""
try:
    import parallelproj_core
except ImportError:
    raise Exception(
        'PET functionality in PyTomography requires parallelproj 2, which provides the parallelproj_core module. '
        'It is distributed through conda-forge:\n'
        '    conda install -c conda-forge parallelproj\n'
        '(choose a libparallelproj build matching your CUDA version, e.g. libparallelproj=*=cuda130*). '
        'See https://parallelproj.readthedocs.io/en/stable/'
    )
from .petlm_system_matrix import PETLMSystemMatrix
from .pet_sinogram_system_matrix import PETSinogramSystemMatrix, create_sinogramSM_from_LMSM
