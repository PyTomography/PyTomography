"""A Gaussian filter of the object, the usual resolution model of PET system matrices."""
from __future__ import annotations
import torch
import numpy as np
from torch.nn.functional import conv1d
import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.transforms import Transform
from pytomography.utils._renamed import renamed_arguments

class GaussianFilter(Transform):
    """Object-to-object transform that smooths the object with a Gaussian of a given FWHM, in mm.

    The filter is separable: a normalised 1D Gaussian along x, y and z in turn, cut off at ``n_sigmas``, with zeros
    beyond the object's edges. It is symmetric, so :meth:`backward` is :meth:`forward`. It can also be called on an
    object directly (``filter(object)``) once configured. A SPECT grid warns when the FWHM is less than a voxel, which
    usually means a FWHM in cm.

    Args:
        fwhm (float | list): The full width at half maximum, in mm (SPECT, PET and CT alike since PyTomography 4.0):
            one number, or a list of three (x, y, z). ``FWHM=`` (PyTomography 3) still works, with a
            DeprecationWarning, until PyTomography 5.
        n_sigmas (float, optional): Half-width of the kernels, in standard deviations (FWHM / 2.355). Defaults to 3.

    Example:
        .. code-block:: python

            psf_transform = GaussianFilter(4.5)            # 4.5 mm FWHM
            system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform])
    """
    @renamed_arguments(FWHM='fwhm')
    def __init__(self, fwhm: float | list, n_sigmas: float = 3):
        if type(fwhm) == list:
            self.sigma = [fwhm[i] / (2*np.sqrt(2*np.log(2))) for i in range(3)]
        else:
            self.sigma = [fwhm / (2*np.sqrt(2*np.log(2))) for i in range(3)]
        self.n_sigmas = n_sigmas
        
    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        """Builds the kernels for the object grid's voxel sizes; the system matrix calls it.

        Args:
            object_meta (ObjectMeta): The object grid.
            proj_meta (ProjMeta): The projection space.
        """
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        from pytomography.metadata.SPECT import SPECTObjectMeta
        fwhm = [s * 2 * np.sqrt(2 * np.log(2)) for s in self.sigma]
        if isinstance(object_meta, SPECTObjectMeta) and any(f < d for f, d in zip(fwhm, object_meta.dr)):
            import warnings
            warnings.warn(f'GaussianFilter: a FWHM of {[round(f, 3) for f in fwhm]} mm is less than one voxel '
                          f'({object_meta.dr} mm); SPECT lengths are in mm since PyTomography 4.0 (2 cm is 20)',
                          stacklevel=2)
        self._get_kernels()
        
    def _get_kernels(self):
        """Obtains required kernels for smoothing
        """
        self.kernels = []
        for i in range(3):
            sigma = self.sigma[i]
            dx = self.object_meta.dr[i]
            kernel_size = int(2*np.ceil(self.n_sigmas*sigma/dx)+1)
            x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1).to(pytomography.device)*dx
            k = torch.exp(-x**2/(2*sigma**2)).reshape(1,1,-1)
            self.kernels.append(k/k.sum())
            
    def __call__(self, object):
        """Same as :meth:`forward`."""
        return self.forward(object)

    def forward(self, object):
        """Smooths the object.

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: The smoothed object, [Lx, Ly, Lz].
        """
        for i in [0,1,2]:
            object = object.swapaxes(i,2)
            new_shape = object.shape
            object = object.reshape(-1,1,new_shape[-1])
            object = conv1d(object, self.kernels[i], padding='same')
            object = object.reshape(new_shape)
            object= object.swapaxes(i,2)
        return object
            
    def backward(self, object, norm_constant=None):
        """Smooths the object in back projection: the same as :meth:`forward`, as the filter is symmetric.

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].
            norm_constant (torch.Tensor, optional): A normalization image (e.g. :math:`H^T 1`) to smooth too.
                Defaults to None.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The smoothed object, and the smoothed ``norm_constant``
            when one is given.
        """

        object = self.forward(object)
        if norm_constant is not None:
            norm_constant = self.forward(norm_constant)
            return object, norm_constant
        else:
            return object

    