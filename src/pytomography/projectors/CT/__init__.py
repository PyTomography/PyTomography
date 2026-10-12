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