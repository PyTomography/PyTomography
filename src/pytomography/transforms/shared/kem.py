from __future__ import annotations
import numpy as np
from pytomography.utils import get_object_nearest_neighbour
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta


class KEMTransform(Transform):
    r"""Object to object transform used to take in a coefficient image :math:`\alpha` and return an image estimate :math:`f = K\alpha`. This transform implements the matrix :math:`K`.

    Args:
        support_objects (Sequence[torch.tensor]): Objects used for support when building each basis function. These may correspond to PET/CT/MRI images, for example.
        support_kernels (Sequence[Callable], optional): A list of functions corresponding to the support kernel of each support object. If none, defaults to :math:`k(v_i, v_j; \sigma) = \exp\left(-\frac{(v_i-v_j)^2}{2\sigma^2} \right)` for each support object. Defaults to None.
        support_kernels_params (Sequence[Sequence[float]], optional): A list of lists, where each sublist contains the additional parameters corresponding to each support kernel (parameters that follow the semi-colon in the expression above). As an example, if using the default configuration for ``support_kernels`` for two different support objects (say CT and PET), one could given ``support_kernel_params=[[40],[5]]`` If none then defaults to a list of `N*[[1]]` where `N` is the number of support objects. Defaults to None.
        distance_kernel (Callable, optional): Kernel used to weight based on voxel-voxel distance. If none, defaults to :math:`k(x_i, x_j; \sigma) = \exp\left(-\frac{(x_i-x_j)^2}{2\sigma^2} \right) Defaults to None.
        distance_kernel_params (_type_, optional): A list of parameters corresponding to additional parameters for the ``distance_kernel`` (i.e. the parameters that follow the semi-colon in the expression above). If none, then defaults to :math:`\sigma=1`. Defaults to None.
        size (int, optional): The size of each kernel. Defaults to 5.
    """
    
    def __init__(
        self,
        support_objects,
        support_kernels = None,
        support_kernels_params = None,
        distance_kernel = None,
        distance_kernel_params = None,
        size: int = 5,
        top_N: int | None = None,
        kernel_on_gpu: bool = False
    ) -> None:
        
        super(KEMTransform, self).__init__()
        self.support_objects = support_objects
        if support_kernels is None:
            # If not given, all default to Gaussian functions
            self.support_kernels = [lambda obj_f, obj_j, sigma: torch.exp(-(obj_f - obj_j)**2 / (2*sigma**2)) for _ in range(len(support_objects))]
        else:
            self.support_kernels = support_kernels
        if support_kernels_params is None:
            # If not given, parameters default to sigma=1 for each kernel
            self.support_kernel_params = [[1] for _ in range(len(support_objects))]
        else:
            self.support_kernel_params = support_kernels_params
        if distance_kernel is None:
            # If not given, defaults to Gaussian function
            self.distance_kernel = lambda d, sigma: np.exp(-d**2 / (2*sigma**2))
        else:
            self.distance_kernel = distance_kernel
        if distance_kernel_params is None:
            # If not given, defaults to sigma = 1cm
            self.distance_kernel_params = [1]
        else:
            self.distance_kernel_params = distance_kernel_params
        self.size = size
        self.idx_max = int((size - 1) / 2)
        self.idxs = torch.arange(-self.idx_max, self.idx_max+1)
        self.top_N = top_N
        self.kernel_on_gpu = kernel_on_gpu
        
    
    def compute_kernel(self):
        """Computes the kernel required for the KEM transform and stores internally
        """
        shape = self.support_objects[0].shape
        # Keep kernel on CPU until its used (its too big for GPU)
        self.kernel = torch.ones((self.size, self.size, self.size, *shape)).to(pytomography.dtype)
        for i in self.idxs:
            for j in self.idxs:
                for k in self.idxs:
                    kernel_component = 1
                    # All support objects:
                    for l in range(len(self.support_objects)):
                        neighbour_support_object = get_object_nearest_neighbour(self.support_objects[l].cpu(), (i,j,k))
                        kernel_component *= self.support_kernels[l](self.support_objects[l].cpu(), neighbour_support_object, *self.support_kernel_params[l])
                    self.kernel[i+self.idx_max,j+self.idx_max,k+self.idx_max] = kernel_component
        # Get only top N entries (based on anatomical support)
        if self.top_N is not None:
            self.kernel *= (torch.argsort(torch.argsort(self.kernel.reshape((self.size**3,*shape)), dim=0), dim=0)>=self.size**3 - self.top_N).reshape((self.size, self.size, self.size,*shape))
        # Scale by distance
        xv, yv, zv = torch.meshgrid([self.object_meta.dx*self.idxs, self.object_meta.dy*self.idxs, self.object_meta.dz*self.idxs])
        d = torch.sqrt(xv**2+yv**2+zv**2)
        self.kernel *= self.distance_kernel(d, *self.distance_kernel_params).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Normalize
        self.kernel /= self.kernel.sum(dim=(0,1,2))
        # Put on GPU (defaults to false since very large)
        if self.kernel_on_gpu:
            self.kernel = self.kernel.to(pytomography.device)
            
    def configure(
        self,
        object_meta: ObjectMeta,
        proj_meta: ProjMeta
    ) -> None:
        """Function used to initalize the transform using corresponding object and projection metadata

        Args:
            object_meta (SPECTObjectMeta): Object metadata.
            proj_meta (SPECTProjMeta): Projections metadata.
        """
        super(KEMTransform, self).configure(object_meta, proj_meta)
        self.compute_kernel()
    
    @torch.no_grad()
    def forward(
		self,
		object: torch.Tensor,
	) -> torch.tensor:
        r"""Forward transform corresponding to :math:`K\alpha`

        Args:
            object (torch.Tensor): Coefficient image :math:`\alpha`

        Returns:
            torch.tensor: Image :math:`K\alpha`
        """
        object_return = torch.zeros(object.shape).to(self.device)
        for i in self.idxs:
            for j in self.idxs:
                for k in self.idxs:
                    neighbour = get_object_nearest_neighbour(object, (i,j,k))
                    object_return += self.kernel[i+self.idx_max,j+self.idx_max,k+self.idx_max].to(pytomography.device) * neighbour
        return object_return
        
    @torch.no_grad()
    def backward(
		self,
		object: torch.Tensor,
	) -> torch.tensor:
        r"""Backward transform corresponding to :math:`K^T\alpha`. Since the matrix is symmetric, the implementation is the same as forward.

        Args:
            object (torch.Tensor): Coefficient image :math:`\alpha`

        Returns:
            torch.tensor: Image :math:`K^T\alpha`
        """
        return self.forward(object)