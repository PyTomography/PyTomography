try:
    import parallelproj
except:
    raise Exception('PET functionality in PyTomography requires the parallelproj package to be installed. Please install it at https://parallelproj.readthedocs.io/en/stable/')
from .petlm_system_matrix import PETLMSystemMatrix
from .pet_sinogram_system_matrix import PETSinogramSystemMatrix, create_sinogramSM_from_LMSM