r"""The system matrix of the kernel method (:class:`KEMSystemMatrix`): a system matrix applied to :math:`K\alpha`."""
from __future__ import annotations
import math
import pytomography
from pytomography.utils.memory import MemoryPart, nbytes
from ..system_matrix import SystemMatrix, on_device

class KEMSystemMatrix(SystemMatrix):
    r"""The system matrix :math:`HK` of the kernel method: a system matrix :math:`H` applied to the image :math:`K\alpha`.

    Its objects are the coefficients :math:`\alpha`, [Lx, Ly, Lz] on the grid of ``system_matrix``: the forward
    projection is :math:`H K \alpha` and the back projection :math:`K^T H^T g`. The subsets (``set_n_subsets``,
    ``get_projection_subset``, ``get_weighting_subset``) are those of ``system_matrix``. Reconstruct with
    :class:`~pytomography.algorithms.KEM`, which returns the image :math:`K\hat{\alpha}`.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H` of the acquisition.
        kem_transform (KEMTransform): The kernel :math:`K` (:class:`~pytomography.transforms.shared.KEMTransform`),
            from coefficients to image; it is configured, and its kernel computed, with ``system_matrix``'s metadata.

    Attributes:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        kem_transform (KEMTransform): The kernel :math:`K`.
    """
    def __init__(self, system_matrix, kem_transform):
        self.object_meta = system_matrix.object_meta
        self.proj_meta = system_matrix.proj_meta
        self.system_matrix = system_matrix
        self.kem_transform = kem_transform
        self.kem_transform.configure(system_matrix.object_meta, system_matrix.proj_meta)
        # Inherit required functions from system matrix
        self.set_n_subsets = self.system_matrix.set_n_subsets
        self.get_projection_subset = self.system_matrix.get_projection_subset
        self.get_weighting_subset = self.system_matrix.get_weighting_subset

    @property
    def subset_indices_array(self):
        """The subsets of the system matrix it wraps (set by ``set_n_subsets``)."""
        return self.system_matrix.subset_indices_array

    @property
    def memory_allocator_fraction(self) -> float | None:
        """The Windows allowance of the inner system matrix's memory estimates (see :func:`~pytomography.utils.memory.memory_estimate`)."""
        return getattr(self.system_matrix, 'memory_allocator_fraction', None)

    @property
    def memory_cuda_projector(self) -> bool:
        """Whether the system matrix it wraps projects with parallelproj (see :func:`~pytomography.utils.memory.memory_estimate`)."""
        return getattr(self.system_matrix, 'memory_cuda_projector', False)

    def _max_subsets(self) -> int:
        return self.system_matrix._max_subsets()

    def _memory_settings(self) -> dict[str, tuple]:
        """The settings of the inner system matrix (the kernel transform has none)."""
        return self.system_matrix._memory_settings()

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list:
        return self.system_matrix._memory_alternatives(n_subsets, N_splits)

    def _memory_parts(self, n_subsets: int, N_splits: int | None) -> list[MemoryPart]:
        """What a reconstruction with this system matrix holds at its peak (see
        :meth:`~pytomography.projectors.SystemMatrix.estimate_memory`): the arrays of the inner system matrix, the
        kernel :math:`K` (``size**3`` images, where the transform keeps it: on the host unless ``kernel_on_gpu``) and the
        support images, and the temporaries of applying :math:`K` on the device (the result, a neighbour, a slice of the
        kernel and their product: four images).

        Args:
            n_subsets (int): Number of subsets.
            N_splits (int | None): Projector splits, or None for the inner system matrix's own.

        Returns:
            list[MemoryPart]: The arrays.
        """
        image = 4 * math.prod(self.object_meta.shape)
        parts = list(self.system_matrix._memory_parts(n_subsets, N_splits))
        parts.append(MemoryPart('KEM kernel', *nbytes(getattr(self.kem_transform, 'kernel', None)), scope='held'))
        support = [nbytes(x) for x in getattr(self.kem_transform, 'support_objects', [])]
        if support:
            parts.append(MemoryPart('KEM support images', sum(r for r, _ in support), sum(g for _, g in support), scope='held'))
        parts.append(MemoryPart('applying the KEM kernel: result, neighbour, kernel slice, product',
                                **on_device(pytomography.device, 4 * image), scope='chunk'))
        return parts

    def compute_normalization_factor(self, subset_idx : int | None = None):
        """The normalization :math:`K^T H_m^T 1` of subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            torch.Tensor: :math:`K^T H_m^T 1`, [Lx, Ly, Lz].
        """
        object = self.system_matrix.compute_normalization_factor(subset_idx)
        return self.kem_transform.backward(object)
        
    def forward(self, object, subset_idx=None):
        r"""Forward projection :math:`H_m K \alpha` of coefficients :math:`\alpha`.

        Args:
            object (torch.Tensor): The coefficients :math:`\alpha`, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            torch.Tensor: The projections, as ``system_matrix`` gives them.
        """
        object = self.kem_transform.forward(object)
        return self.system_matrix.forward(object, subset_idx)
    
    def backward(self, proj, subset_idx=None):
        r"""Back projection :math:`K^T H_m^T g_m` of projections, into coefficient space.

        Args:
            proj (torch.Tensor): The projections, as ``system_matrix`` takes them (only the subset's with
                ``subset_idx``).
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            torch.Tensor: The coefficients, [Lx, Ly, Lz].
        """
        object = self.system_matrix.backward(proj, subset_idx)
        return self.kem_transform.backward(object)
        