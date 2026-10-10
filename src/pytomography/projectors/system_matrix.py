from __future__ import annotations
from typing import Mapping, Sequence
import abc
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.utils.memory import MemoryEstimate, MemoryPart, memory_estimate, nbytes
from copy import copy

class SystemMatrix():
    r"""Abstract class for a general system matrix :math:`H:\mathbb{U} \to \mathbb{V}` which takes in an object :math:`f \in \mathbb{U}` and maps it to corresponding projections :math:`g \in \mathbb{V}` that would be produced by the imaging system. A system matrix consists of sequences of object-to-object and proj-to-proj transforms that model various characteristics of the imaging system, such as attenuation and blurring. While the class implements the operator :math:`H:\mathbb{U} \to \mathbb{V}` through the ``forward`` method, it also implements :math:`H^T:\mathbb{V} \to \mathbb{U}` through the `backward` method, required during iterative reconstruction algorithms such as OSEM.
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            im2im_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
            object_meta (ObjectMeta): Object metadata.
            proj_meta (ProjMeta): Projection metadata.
    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: ProjMeta,
        obj2obj_transforms: list[Transform] = [],
        proj2proj_transforms: list[Transform] = [],
    ) -> None:
        self.obj2obj_transforms = obj2obj_transforms
        self.proj2proj_transforms = proj2proj_transforms
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self.initialize_transforms()
             
    def __mul__(self, scalar):
        old_forward = copy(self.forward)
        old_backward = copy(self.backward)
        def new_forward(*args, **kwargs):
            return old_forward(*args, **kwargs) * scalar
        def new_backward(*args, **kwargs):
            return old_backward(*args, **kwargs) * scalar
        self.forward = new_forward
        self.backward = new_backward
        return self
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def initialize_transforms(self):
        """Initializes all transforms used to build the system matrix
        """
        for transform in self.obj2obj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
        for transform in self.proj2proj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
            
    def _get_object_initial(self, device=None):
        """Returns an initial object estimate used in reconstruction algorithms. By default, this is a tensor of ones with the same shape as the object metadata.

        Returns:
            torch.Tensor: Initial object used in image reconstruction algorithms.
        """
        if device is None:
            device = pytomography.device
        return torch.ones(self.object_meta.shape).to(device)
    
    def _get_prior_FOV_scale(self):
        """Sets scaling for the prior within the FOV.

        Returns:
            torch.Tensor: Prior scaling
        """
        return torch.ones(self.object_meta.shape).to(pytomography.device)

    def _fbp(self, projections: torch.Tensor, filter, **kwargs) -> torch.Tensor:
        """Filtered back projection in the geometry of this system matrix, onto its object grid. It is called by
        :class:`pytomography.algorithms.FilteredBackProjection`, which is how it should be used. System matrices whose
        geometry supports an analytic reconstruction override it; this one raises.

        Args:
            projections (torch.Tensor): Projections to reconstruct.
            filter (FBPFilter): Window applied on top of the ramp filter (see :func:`pytomography.utils.get_fbp_filter`).

        Raises:
            NotImplementedError: This system matrix has no filtered back projection.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support filtered back projection')

    def _fbp_memory_parts(self, projections: torch.Tensor, filter, **kwargs) -> list[MemoryPart]:
        """The arrays :meth:`_fbp` would take with the same arguments, without running it and without the overheads
        every estimate adds (:func:`pytomography.utils.memory.memory_estimate`): the projections, the image, and the
        most its steps hold at once. It is called by
        :meth:`pytomography.algorithms.FilteredBackProjection.estimate_memory`. System matrices with a :meth:`_fbp`
        override it; this one raises.

        Raises:
            NotImplementedError: This system matrix has no filtered back projection.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support filtered back projection')

    def _fbp_memory_alternatives(self, projections: torch.Tensor, filter, **kwargs) -> list[tuple[str, list[MemoryPart]]]:
        """Other settings of :meth:`_fbp` that take less memory, with their arrays, as ``(label, parts)``, for the "To use
        less" line of :meth:`pytomography.algorithms.FilteredBackProjection.estimate_memory`. None by default."""
        return []

    def _memory_parts(self, n_subsets: int, N_splits: int | None) -> list[MemoryPart]:
        """The arrays a reconstruction with this system matrix holds at its peak, without the overheads every estimate
        adds (:func:`pytomography.utils.memory.memory_estimate`): what it holds for the whole run, one subset's arrays and
        the projector's temporaries. System matrices that can estimate their memory override it.

        Args:
            n_subsets (int): Number of subsets.
            N_splits (int | None): Projector splits, or None for this system matrix's own.

        Returns:
            list[MemoryPart]: The arrays.
        """
        raise NotImplementedError(f"{type(self).__name__} does not estimate its memory yet")

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list[tuple[str, int, int | None]]:
        """The settings the "To use less" line of :meth:`estimate_memory` shows, as ``(label, n_subsets, N_splits)``:
        twice the subsets, and twice the projector splits if this system matrix has them. System matrices for which
        other settings use less (e.g. fewer subsets) override it.

        Args:
            n_subsets (int): Number of subsets of the estimate.
            N_splits (int | None): Projector splits of the estimate.

        Returns:
            list[tuple[str, int, int | None]]: The settings.
        """
        alternatives = [(f"{2 * n_subsets} subsets", 2 * n_subsets, N_splits)]
        splits = N_splits if N_splits is not None else getattr(self, 'N_splits', None)
        if splits is not None:
            alternatives.append((f"N_splits {2 * splits}", n_subsets, 2 * splits))
        return alternatives

    def estimate_memory(self, n_subsets: int = 1, N_splits: int | None = None, held: Sequence | Mapping = ()) -> MemoryEstimate:
        """The predicted peak RAM and GPU memory of a reconstruction with this system matrix, for ``n_subsets`` subsets
        and ``N_splits`` projector splits, under the current memory budget (:func:`pytomography.set_memory_budget`).
        ``print`` it to see the peak, what it is made of, and what more subsets or splits would need ("To use less");
        ``ram_gb`` and ``gpu_gb`` give the numbers. On Windows, GPU memory counts as RAM too.

        Args:
            n_subsets (int, optional): Number of subsets. Defaults to 1.
            N_splits (int | None, optional): Projector splits, for system matrices that take them. Defaults to None: this
                system matrix's own.
            held (Sequence | Mapping, optional): Other arrays the reconstruction holds for the whole run, such as its
                projections and additive term (tensors or :class:`~pytomography.io.PET.shared.LazySinogram`), as a list
                or as a dict of names to arrays. Their size is added. Defaults to none.

        Returns:
            MemoryEstimate: The estimate.

        Example:
            >>> estimate = system_matrix.estimate_memory(n_subsets=14, held={'data': sinogram, 'additive term': additive_term})
            >>> print(estimate)
        """
        items = held.items() if isinstance(held, Mapping) else ((f"array {k + 1}", x) for k, x in enumerate(held))
        held_parts = [MemoryPart(name, *nbytes(x), scope='held') for name, x in items]
        title = (f"{type(self).__name__}: {n_subsets} subset{'' if n_subsets == 1 else 's'}"
                 + (f", N_splits {N_splits}" if N_splits is not None else "")
                 + (f", memory budget {pytomography.memory_budget / 1e9:g} GB" if pytomography.memory_budget is not None else ""))
        alternatives = [(label, list(self._memory_parts(k, s)) + held_parts) for label, k, s in self._memory_alternatives(n_subsets, N_splits)]
        return memory_estimate(title, list(self._memory_parts(n_subsets, N_splits)) + held_parts, alternatives,
                               allocator_fraction=getattr(self, 'memory_allocator_fraction', None))

    def fewest_subsets(self, ram_gb: float, N_splits: int | None = None, held: Sequence | Mapping = (), max_subsets: int = 1024) -> int | None:
        """The fewest subsets for which :meth:`estimate_memory` predicts at most ``ram_gb`` GB of RAM, or None if no
        number up to ``max_subsets`` does. The numbers are tried in order (an estimate is arithmetic), since memory need
        not fall with more subsets (a CT reconstruction keeps a normalisation image per subset).

        Args:
            ram_gb (float): RAM available, in GB.
            N_splits (int | None, optional): Projector splits. Defaults to None: this system matrix's own.
            held (Sequence | Mapping, optional): As for :meth:`estimate_memory`. Defaults to none.
            max_subsets (int, optional): Most subsets to consider. Defaults to 1024.

        Returns:
            int | None: Number of subsets.
        """
        for n_subsets in range(1, max_subsets + 1):
            if self.estimate_memory(n_subsets, N_splits, held).ram_gb <= ram_gb:
                return n_subsets
        return None

    @abc.abstractmethod
    def forward(self, object: torch.tensor, **kwargs):
        r"""Implements forward projection :math:`Hf` on an object :math:`f`.

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected proj where Ltheta is specified by `self.proj_meta` and `angle_subset`.
        """
        ...
    @abc.abstractmethod
    def backward(
        self,
        proj: torch.tensor,
        angle_subset: list | None = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        r"""Implements back projection :math:`H^T g` on a set of projections :math:`g`.

        Args:
            proj (torch.Tensor): proj which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            return_norm_constant (bool): Whether or not to return :math:`1/\sum_j H_{ij}` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        ...
    @abc.abstractmethod
    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        """Returns a list of subsets corresponding to a partition of the projection data used in a reconstruction algorithm.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 

        Returns:
            list: list of index arrays for each subset
        """
        ...

class ExtendedSystemMatrix(SystemMatrix):
    def __init__(
        self,
        system_matrices: Sequence[SystemMatrix],
        obj2obj_transforms: Sequence[Transform] = None,
        proj2proj_transforms: Sequence[Transform] = None,
        ) -> None:
        r"""System matrix that supports the extension of projection space. Maps to an extended image space :math:`\mathcal{V}^{*}` where projections have shape ``[N,...]`` where ``...`` is the regular projeciton size. As such, this projector only supports objects with a batch size of 1. The forward transform is given by :math:`H' = \sum_n v_n \otimes B_n H_n A_n` where :math:`\left\{A_n\right\}` are object-to-object space transforms, :math:`\left\{H_n\right\}` are a sequence of system matrices, :math:`\left\{B_n\right\}` are a sequence of projection-to-projection space transforms, :math:`v_n` is a basis vector of length :math:`N` with a value of 1 in component :math:`n` and :math:`\otimes` is a tensor product. 

        Args:
            system_matrices (Sequence[SystemMatrix]): List of system matrices corresponding to each dimension :math:`n`. 
            obj2obj_transforms (Sequence[Transform]): List of object to object transforms corresponding to each dimension :math:`n`. 
            proj2proj_transforms (Sequence[Transform]): List of projection to projection transforms corresponding to each dimension :math:`n`. 
        """
        self.object_meta = system_matrices[0].object_meta
        self.proj_meta = system_matrices[0].proj_meta
        self.system_matrices = system_matrices
        self.obj2obj_transforms = obj2obj_transforms
        self.proj2proj_transforms = proj2proj_transforms
        for i in range(len(self.system_matrices)):
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    self.obj2obj_transforms[i].configure(self.system_matrices[i].object_meta, self.system_matrices[i].proj_meta)
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    self.proj2proj_transforms[i].configure(self.system_matrices[i].object_meta, self.system_matrices[i].proj_meta)
        
    def forward(self, object, subset_idx: int | None = None):
        r"""Forward transform :math:`H' = \sum_n v_n \otimes B_n H_n A_n`, This adds an additional dimension to the projection space.

        Args:
            object (torch.Tensor[1,Lx,Ly,Lz]): Object to be forward projected. Must have a batch size of 1.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
           torch.Tensor[N_gates,...]: Forward projection.
        """
        projs = []
        for i in range(len(self.system_matrices)):
            object_i = object.clone()
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    object_i = self.obj2obj_transforms[i].forward(object)
            proj_i = self.system_matrices[i].forward(object_i, subset_idx)
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    proj_i = self.proj2proj_transforms[i].forward(proj_i)
            projs.append(proj_i.clone())
        return torch.stack(projs)
    
    def backward(self, proj, subset_idx: int | None =None):
        r"""Back projection :math:`H' = \sum_n v_n^T \otimes A_n^T H_n^T B_n^T`. This maps an extended projection back to the original object space.

        Args:
            proj (torch.Tensor[N,...]): Projection data to be back-projected.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.. Defaults to None.

        Returns:
            torch.Tensor[1,Lx,Ly,Lz]: Back projection.
        """
        objects = []
        for i in range(len(self.system_matrices)):
            proj_i = proj[i]
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    proj_i = self.proj2proj_transforms[i].backward(proj_i)
            object_i = self.system_matrices[i].backward(proj_i, subset_idx)
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    object_i = self.obj2obj_transforms[i].backward(object_i)
            objects.append(object_i.clone())
        return torch.stack(objects).sum(axis=0)
    
    def set_n_subsets(
        self,
        n_subsets: int
    ) -> list:
        # assumes all are similar
        for system_matrix in self.system_matrices:
            system_matrix.set_n_subsets(n_subsets)
        self.subset_indices_array = self.system_matrices[0].subset_indices_array
        
    def get_projection_subset(
        self,
        projections: torch.tensor,
        subset_idx: int
    ) -> torch.tensor: 
        return self.system_matrices[0].get_projection_subset(projections, subset_idx)
    
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> torch.tensor:
        return self.system_matrices[0].get_weighting_subset(subset_idx)
    
    def compute_normalization_factor(self, subset_idx: int | None = None):
        r"""Function called by reconstruction algorithms to get the normalization factor :math:`H' = \sum_n v_n^T \otimes A_n^T H_n^T B_n^T` 1.

        Returns:
           torch.Tensor[1,Lx,Ly,Lz]: Normalization factor.
        """
        norm_proj = torch.ones((len(self.system_matrices), *self.proj_meta.shape)).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = norm_proj[:,self.subset_indices_array[subset_idx]]
        return self.backward(norm_proj, subset_idx)
