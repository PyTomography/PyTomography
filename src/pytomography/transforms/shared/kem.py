"""The kernel matrix of the kernel method (KEM): an image as :math:`f = K\\alpha`, from anatomical support images."""
from __future__ import annotations
import numpy as np
from pytomography.utils import get_object_nearest_neighbour
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta


class KEMTransform(Transform):
    r"""Object-to-object transform that maps a coefficient image :math:`\alpha` to the image :math:`f = K\alpha`.

    Each voxel of :math:`f` is a weighted mean of the coefficients in the ``size`` x ``size`` x ``size`` neighbourhood
    around it. The weight of a neighbour is the product of the support kernels of every support image (how alike the
    two voxels are in it) and the distance kernel, normalised so that each voxel's weights sum to 1. The kernel is
    computed in :meth:`configure`; it takes ``size**3`` images of memory. :meth:`backward` applies the same
    operation as :meth:`forward` (:math:`K` is taken as symmetric). Used by
    :class:`~pytomography.projectors.shared.KEMSystemMatrix` and :class:`~pytomography.algorithms.KEM`.

    Args:
        support_objects (Sequence[torch.Tensor]): The support images (e.g. CT, MRI or PET images), [Lx, Ly, Lz] each,
            on the object grid.
        support_kernels (Sequence[Callable], optional): The support kernel of each support image, a function of the
            voxel's value, its neighbour's value and the kernel's parameters. Defaults to None: the Gaussian
            :math:`k(v_i, v_j; \sigma) = \exp\left(-\frac{(v_i-v_j)^2}{2\sigma^2} \right)` for every support image.
        support_kernels_params (Sequence[Sequence[float]], optional): The parameters of each support kernel (those
            after the semicolon above), e.g. ``[[40], [5]]`` for the default kernels of two support images (in the
            units of their values). Defaults to None: ``[1]`` for every support image.
        distance_kernel (Callable, optional): The kernel of the distance between the voxels, a function of the distance
            in mm and its parameters. Defaults to None: the Gaussian
            :math:`k(d; \sigma) = \exp\left(-\frac{d^2}{2\sigma^2} \right)`.
        distance_kernel_params (Sequence[float], optional): The parameters of ``distance_kernel``; distances are in
            mm. Defaults to None: :math:`\sigma = 10` mm.
        size (int, optional): The size of the neighbourhood along each axis, in voxels (odd). Defaults to 5.
        top_N (int | None, optional): Keep only the ``top_N`` largest support weights of each voxel (before the
            distance kernel; of equal weights, the nearest neighbours), and set the others to zero. Defaults to None:
            keep them all.
        kernel_on_gpu (bool, optional): Keep the kernel on ``pytomography.device`` instead of the CPU (it is large).
            Defaults to False.
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
            # If not given, defaults to sigma = 10 mm (1 cm, as before lengths were mm for every modality)
            self.distance_kernel_params = [10]
        else:
            self.distance_kernel_params = distance_kernel_params
        self.size = size
        self.idx_max = int((size - 1) / 2)
        self.idxs = torch.arange(-self.idx_max, self.idx_max+1)
        self.top_N = top_N
        self.kernel_on_gpu = kernel_on_gpu
        
    
    def compute_kernel(self):
        """Computes the kernel :math:`K` and keeps it in ``kernel``, [size, size, size, Lx, Ly, Lz].

        :meth:`configure` calls it. The kernel stays on the CPU unless ``kernel_on_gpu`` is set.
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
        xv, yv, zv = torch.meshgrid([self.object_meta.dx*self.idxs, self.object_meta.dy*self.idxs, self.object_meta.dz*self.idxs], indexing='ij')
        d = torch.sqrt(xv**2+yv**2+zv**2)
        # Get only top N entries (based on anatomical support)
        if self.top_N is not None and self.top_N < self.size**3:
            self._keep_top_N(d)
        # Scale by distance
        self.kernel *= self.distance_kernel(d, *self.distance_kernel_params).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Normalize
        self.kernel /= self.kernel.sum(dim=(0,1,2))
        # Put on GPU (defaults to false since very large)
        if self.kernel_on_gpu:
            self.kernel = self.kernel.to(pytomography.device)

    def _keep_top_N(self, d: torch.Tensor, chunk_bytes: float = 2**28) -> None:
        """Keeps the ``top_N`` largest kernel entries of each voxel and sets the others to zero, in place. Of equal entries
        (frequent: neighbours with the same support values have equal entries), the nearest neighbours are kept. The
        voxels are taken in chunks, so that this needs about ``chunk_bytes`` more memory however large the image is
        (ranking every entry of the whole kernel at once took about 8 times the kernel's memory).

        Args:
            d (torch.Tensor): Distance of each neighbour from the centre, ``[size, size, size]``.
            chunk_bytes (float, optional): Memory for each chunk of voxels. Defaults to 256 MB.
        """
        n = self.size**3
        flat = self.kernel.reshape((n, -1))
        nearest_first = torch.argsort(d.reshape(-1), stable=True)
        chunk = max(1, int(chunk_bytes // (n * 22)))  # bytes per voxel: a copy, the sort's values and indices, two masks
        for start in range(0, flat.shape[1], chunk):
            part = flat[nearest_first, start:start+chunk]
            # a stable sort keeps the nearer of equal entries first
            top = torch.sort(part, dim=0, descending=True, stable=True).indices[:self.top_N]
            keep = torch.zeros_like(part, dtype=torch.bool).scatter_(0, top, True)
            keep_flat = torch.empty_like(keep)
            keep_flat[nearest_first] = keep
            flat[:, start:start+chunk] *= keep_flat
            
    def configure(
        self,
        object_meta: ObjectMeta,
        proj_meta: ProjMeta
    ) -> None:
        """Computes the kernel for the object grid (its voxel sizes set the distances); the system matrix calls it.

        Args:
            object_meta (ObjectMeta): The object grid.
            proj_meta (ProjMeta): The projection space.
        """
        super(KEMTransform, self).configure(object_meta, proj_meta)
        self.compute_kernel()
    
    @torch.no_grad()
    def forward(
		self,
		object: torch.Tensor,
	) -> torch.tensor:
        r"""The image :math:`K\alpha` of a coefficient image.

        Args:
            object (torch.Tensor): The coefficient image :math:`\alpha`, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: The image :math:`K\alpha`, [Lx, Ly, Lz].
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
        r"""The transpose :math:`K^T`, applied as :meth:`forward` (:math:`K` is taken as symmetric).

        Args:
            object (torch.Tensor): An image, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: :math:`K^T` applied to it, [Lx, Ly, Lz].
        """
        return self.forward(object)