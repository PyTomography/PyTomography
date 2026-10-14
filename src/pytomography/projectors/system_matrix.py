from __future__ import annotations
from typing import Callable, Mapping, Sequence
import abc
import math
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.utils.memory import (MemoryEstimate, MemoryPart, MemoryPlan, memory_estimate, nbytes, plan_settings,
                                       split_candidates, tensor_bytes, PLAN_MAX_SUBSETS)
from copy import copy


def held_parts(held: Sequence | Mapping) -> list[MemoryPart]:
    """The arrays a computation holds for its whole run, as memory parts: ``held`` is a list of arrays (tensors or
    :class:`~pytomography.io.PET.shared.LazySinogram`) or a dict of names to arrays; a :class:`MemoryPart` among them
    (for arrays not made yet) is taken as it is."""
    items = held.items() if isinstance(held, Mapping) else ((f"array {k + 1}", x) for k, x in enumerate(held))
    return [x if isinstance(x, MemoryPart) else MemoryPart(name, *nbytes(x), scope='held') for name, x in items]


def on_device(device, nbytes: float) -> dict:
    """``{'gpu_bytes': nbytes}`` for a CUDA device, ``{'ram_bytes': nbytes}`` otherwise: the keyword of a
    :class:`MemoryPart` for an array on ``device``."""
    return {'gpu_bytes': float(nbytes)} if torch.device(device).type == 'cuda' else {'ram_bytes': float(nbytes)}

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

    def _fbp_memory_settings(self, projections: torch.Tensor, filter, **kwargs) -> dict[str, tuple]:
        """The settings of :meth:`_fbp` with the same arguments that
        :meth:`pytomography.algorithms.FilteredBackProjection.plan_memory` chooses, as ``{name: (present value,
        candidates)}`` in the order they are searched, each list fastest first. A name is a keyword of :meth:`_fbp`, or
        ``memory_budget_gb`` for the memory budget (:func:`pytomography.set_memory_budget`). None by default: the
        reconstruction has nothing to choose."""
        return {}

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

    def _saving_memory(self, candidates: list, n_subsets: int, N_splits: int | None) -> list:
        """The ``(label, n_subsets, N_splits)`` ``candidates`` with which :meth:`_memory_parts` saves RAM or GPU memory
        compared with ``n_subsets`` and ``N_splits``: at least 5% of the two together, so that a few kilobytes of one are
        not offered against far more of the other. For system matrices whose memory can fall with fewer subsets as well
        as more (one :math:`H_m^T 1` image per subset), whose :meth:`_memory_alternatives` offer both."""
        def totals(n, s):
            parts = self._memory_parts(n, s)
            return sum(p.ram_bytes for p in parts), sum(p.gpu_bytes for p in parts)
        ram, gpu = totals(n_subsets, N_splits)
        least = 0.05 * (ram + gpu)
        return [c for c in candidates if c[1] >= 1 and (lambda r, g: ram - r >= least or gpu - g >= least)(*totals(c[1], c[2]))]

    def _max_subsets(self) -> int:
        """Most subsets :meth:`plan_memory` tries: one view (or angle) each, at most
        :data:`~pytomography.utils.memory.PLAN_MAX_SUBSETS`."""
        meta = self.proj_meta
        n = getattr(meta, 'N_angles', None) or getattr(meta, 'num_projections', None)
        return max(1, min(PLAN_MAX_SUBSETS, int(n))) if n else PLAN_MAX_SUBSETS

    def _memory_settings(self) -> dict[str, tuple]:
        """The settings of a reconstruction with this system matrix that :meth:`plan_memory` chooses, in one place, as
        ``{name: (present value, candidates)}``: in the order they are searched (the first is the outer loop), each
        list fastest first. Here: the projector splits ``N_splits`` (1, 2, 4, ... 256 and the present value), if this
        system matrix has them, then the number of subsets ``n_subsets`` (every number from 1 to the number of views or
        angles, at most :data:`~pytomography.utils.memory.PLAN_MAX_SUBSETS`). Each name is a keyword of
        :meth:`estimate_memory`. System matrices with other settings or limits override it.

        Returns:
            dict[str, tuple]: The settings.
        """
        settings = {}
        splits = getattr(self, 'N_splits', None)
        if splits is not None:
            settings['N_splits'] = (splits, split_candidates(splits))
        present = len(self.subset_indices_array) if hasattr(self, 'subset_indices_array') else 1
        settings['n_subsets'] = (present, list(range(1, self._max_subsets() + 1)))
        return settings

    def _memory_estimate(self, settings: Mapping, held: Sequence[MemoryPart] = (), alternatives: bool = True,
                         extra: Callable[[int], list] | None = None, title: str | None = None,
                         title_settings: Mapping | None = None) -> MemoryEstimate:
        """The estimate of a reconstruction with every setting of :meth:`_memory_settings` given (``settings``), with
        the memory parts ``held`` added, and those of ``extra(n_subsets)`` (arrays of a reconstruction algorithm that
        depend on the subsets). :meth:`estimate_memory` and :meth:`plan_memory` are built on it, and so are the
        algorithms' ``estimate_memory`` and ``plan_memory``.

        Args:
            settings (Mapping): The settings, by name (``n_subsets``, and ``N_splits`` if this system matrix has it).
            held (Sequence[MemoryPart], optional): Parts to add. Defaults to none.
            alternatives (bool, optional): Whether to evaluate the "To use less" settings. Defaults to True.
            extra (Callable[[int], list] | None, optional): More parts, as a function of the number of subsets.
                Defaults to none.
            title (str | None, optional): What is estimated. Defaults to the name of this system matrix.
            title_settings (Mapping | None, optional): The settings the title names. Defaults to ``settings``.

        Returns:
            MemoryEstimate: The estimate.
        """
        n_subsets = int(settings.get('n_subsets', 1))
        N_splits = settings.get('N_splits')
        shown = settings if title_settings is None else title_settings
        name = title if title is not None else type(self).__name__
        title = (f"{name}: {n_subsets} subset{'' if n_subsets == 1 else 's'}"
                 + (f", N_splits {shown['N_splits']}" if shown.get('N_splits') is not None else "")
                 + (f", memory budget {pytomography.memory_budget / 1e9:g} GB" if pytomography.memory_budget is not None else ""))
        more = (lambda n: list(extra(n))) if extra is not None else (lambda n: [])
        others = []
        if alternatives:
            for label, k, s in self._memory_alternatives(n_subsets, N_splits):
                alternative = {key: value for key, value in (('n_subsets', k), ('N_splits', s if s is not None else N_splits)) if key in settings}
                others.append((label, list(self._memory_parts(k, s)) + list(held) + more(k), alternative))
        return memory_estimate(title, list(self._memory_parts(n_subsets, N_splits)) + list(held) + more(n_subsets), others,
                               allocator_fraction=getattr(self, 'memory_allocator_fraction', None), settings=dict(settings),
                               cuda_projector=getattr(self, 'memory_cuda_projector', False))

    def estimate_memory(self, n_subsets: int = 1, N_splits: int | None = None, held: Sequence | Mapping = ()) -> MemoryEstimate:
        """The predicted peak RAM and GPU memory of a reconstruction with this system matrix, for ``n_subsets`` subsets
        and ``N_splits`` projector splits, under the current memory budget (:func:`pytomography.set_memory_budget`).
        ``print`` it to see the peak, what it is made of, and what more subsets or splits would need ("To use less");
        ``ram_gb`` and ``gpu_gb`` give the numbers, ``to_dict()`` all of it as JSON. On Windows, GPU memory counts as
        RAM too. For the opposite question, which settings fit in the memory there is, see :meth:`plan_memory`.

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
        settings = {name: present for name, (present, _) in self._memory_settings().items()}
        settings['n_subsets'] = n_subsets
        if N_splits is not None and 'N_splits' in settings:
            settings['N_splits'] = N_splits
        return self._memory_estimate(settings, held_parts(held), title_settings={'N_splits': N_splits})

    def plan_memory(self, ram_gb: float | None = None, gpu_gb: float | None = None, held: Sequence | Mapping = (), **fixed) -> MemoryPlan:
        """The fastest settings of a reconstruction with this system matrix that fit in ``ram_gb`` of RAM and ``gpu_gb`` of
        GPU memory, as a :class:`~pytomography.utils.memory.MemoryPlan` (``plan.settings``, ``plan.fits``,
        ``plan.estimate``; ``print`` it, or ``plan.to_dict()`` for JSON). This is the opposite question to
        :meth:`estimate_memory`: memory in, settings out.

        Settings passed by name are kept (e.g. ``n_subsets=14``); the others are chosen. The fewest projector splits
        ``N_splits`` come first, then the fewest subsets that fit with them: every number of subsets is tried in order,
        since memory need not fall with more subsets (a CT or SPECT reconstruction keeps an :math:`H_m^T 1` image per
        subset). The settings and the values tried are those of :meth:`_memory_settings`. Without budgets, the plan is
        for what is free on this machine now (:func:`~pytomography.utils.memory.available_memory`, which keeps a margin
        free). On Windows GPU memory counts as RAM too, so the RAM budget covers both. If nothing fits, the plan has
        ``fits`` False and the settings that come closest; it does not raise.

        Args:
            ram_gb (float | None, optional): RAM to fit in, in GB. Defaults to None: what is available now.
            gpu_gb (float | None, optional): GPU memory to fit in, in GB. Defaults to None: what is available now (0
                without a CUDA device, where the arrays are in RAM).
            held (Sequence | Mapping, optional): As for :meth:`estimate_memory`: the projections, additive term and
                other arrays the reconstruction holds. Defaults to none.
            **fixed: Settings to keep, by name: ``n_subsets``, and ``N_splits`` for system matrices that take it.

        Returns:
            MemoryPlan: The plan.

        Raises:
            ValueError: A setting in ``fixed`` that this system matrix does not have.

        Example:
            >>> plan = system_matrix.plan_memory(ram_gb=25, held={'data': sinogram}, n_subsets=14)
            >>> print(plan)
            >>> system_matrix.N_splits = plan.settings['N_splits']
            >>> recon = OSEM(likelihood)(n_iters=4, n_subsets=plan.settings['n_subsets'])
        """
        parts = held_parts(held)
        return plan_settings(type(self).__name__, self._memory_settings(), lambda s, alternatives: self._memory_estimate(s, parts, alternatives),
                             fixed, ram_gb, gpu_gb)

    def fewest_subsets(self, ram_gb: float, N_splits: int | None = None, held: Sequence | Mapping = (), max_subsets: int = 1024) -> int | None:
        """The fewest subsets for which :meth:`estimate_memory` predicts at most ``ram_gb`` GB of RAM, or None if no
        number up to ``max_subsets`` does. The numbers are tried in order (an estimate is arithmetic), since memory need
        not fall with more subsets (a CT reconstruction keeps a normalisation image per subset). It is
        :meth:`plan_memory` with the projector splits kept and no GPU limit.

        Args:
            ram_gb (float): RAM available, in GB.
            N_splits (int | None, optional): Projector splits. Defaults to None: this system matrix's own.
            held (Sequence | Mapping, optional): As for :meth:`estimate_memory`. Defaults to none.
            max_subsets (int, optional): Most subsets to consider. Defaults to 1024.

        Returns:
            int | None: Number of subsets.
        """
        settings = self._memory_settings()
        settings['n_subsets'] = (settings['n_subsets'][0], list(range(1, max_subsets + 1)))
        fixed = {}
        if 'N_splits' in settings:
            fixed['N_splits'] = settings['N_splits'][0] if N_splits is None else N_splits
        parts = held_parts(held)
        plan = plan_settings(type(self).__name__, settings, lambda s, alternatives: self._memory_estimate(s, parts, False),
                             fixed, ram_gb=ram_gb, gpu_gb=math.inf)
        return plan.settings['n_subsets'] if plan.fits else None

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

    def get_weighting_subset(self, subset_idx: int | None) -> float:
        r"""The share of the projections in subset :math:`m`, which scales parameters relative to :math:`H_m^T 1` in reconstruction algorithms (a prior's :math:`\beta`, BSREM's step); 1 without subsets. The subsets ``set_n_subsets`` makes partition the projections, so this is the number of projections in the subset over the number in all of them: for StarGuide, whose subsets are groups of angles, the views of the subset's angles over all views. System matrices whose projections do not all weigh the same override it.

        Args:
            subset_idx (int | None): Subset index, or None for all the projections.

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        return len(self.subset_indices_array[subset_idx]) / sum(len(subset) for subset in self.subset_indices_array)

    @staticmethod
    def _check_n_subsets(n_subsets: int, n_views: int, views: str = 'angles') -> None:
        """Stops a partition into subsets that would leave a subset empty, with the most there can be.

        Args:
            n_subsets (int): Number of subsets asked for.
            n_views (int): Number of views the projections are partitioned by (angles, projections).
            views (str, optional): What the views are, for the message. Defaults to 'angles'.

        Raises:
            ValueError: ``n_subsets`` is less than 1 or more than ``n_views``.
        """
        if not 1 <= n_subsets <= n_views:
            raise ValueError(f"{n_subsets} subsets of {n_views} {views}: each subset needs at least one, so use 1 to {n_views} subsets")


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
    ) -> None:
        """Sets the same subsets on every system matrix, and their ``subset_indices_array`` (that of the first) on this one.

        Args:
            n_subsets (int): Number of subsets.
        """
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
    
    @property
    def memory_allocator_fraction(self) -> float | None:
        """The Windows allowance of the first system matrix's memory estimates (see :func:`~pytomography.utils.memory.memory_estimate`)."""
        return getattr(self.system_matrices[0], 'memory_allocator_fraction', None)

    @property
    def memory_cuda_projector(self) -> bool:
        """Whether the first system matrix projects with parallelproj (see :func:`~pytomography.utils.memory.memory_estimate`)."""
        return getattr(self.system_matrices[0], 'memory_cuda_projector', False)

    def _max_subsets(self) -> int:
        return self.system_matrices[0]._max_subsets()

    def _memory_settings(self) -> dict[str, tuple]:
        """The settings of the first system matrix (they are all assumed alike, as :meth:`set_n_subsets` assumes)."""
        return self.system_matrices[0]._memory_settings()

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list:
        return self.system_matrices[0]._memory_alternatives(n_subsets, N_splits)

    def _memory_parts(self, n_subsets: int, N_splits: int | None) -> list[MemoryPart]:
        """What a reconstruction with this system matrix holds at its peak (see :meth:`SystemMatrix.estimate_memory`):
        the arrays of the first system matrix, with its projections of a subset (the parts named ``projections ...``
        or ``data ...``) counted twice for each of the :math:`N` system matrices (each one's projection is kept, then
        they are stacked); the arrays every other distinct system matrix keeps of its own (its transforms', its
        sensitivity, its events; not the images and caches of the reconstruction, which are shared); the arrays the
        transforms keep and their temporaries (twelve images, rounded up from a deformation's coordinates and
        resampling); and the projections of ones of all :math:`N` that :meth:`compute_normalization_factor` back
        projects.

        Args:
            n_subsets (int): Number of subsets.
            N_splits (int | None): Projector splits, or None for the system matrices' own.

        Returns:
            list[MemoryPart]: The arrays.
        """
        N = len(self.system_matrices)
        first = self.system_matrices[0]
        parts = []
        for part in first._memory_parts(n_subsets, N_splits):
            if part.scope == 'subset' and part.name.startswith(('projections', 'data')):
                parts.append(MemoryPart(f"{part.name}, for each of {N}, kept and stacked", 2 * N * part.ram_bytes, 2 * N * part.gpu_bytes, 'subset'))
            else:
                parts.append(part)
        shared = ('image', 'H_m^T 1', 'rotation grids', 'normalisation images', 'sensitivity and normalisation images')
        seen = {id(first)}
        for i, system_matrix in enumerate(self.system_matrices[1:], start=2):
            if id(system_matrix) in seen:
                continue
            seen.add(id(system_matrix))
            held = [p for p in system_matrix._memory_parts(n_subsets, N_splits) if p.scope == 'held' and not p.name.startswith(shared)]
            if not held:
                continue
            parts.append(MemoryPart(f"system matrix {i}: what it holds", sum(p.ram_bytes for p in held), sum(p.gpu_bytes for p in held), 'held'))
        transforms = [t for t in list(self.obj2obj_transforms or []) + list(self.proj2proj_transforms or []) if t is not None]
        if transforms:
            ram = sum(tensor_bytes(t)[0] for t in transforms)
            gpu = sum(tensor_bytes(t)[1] for t in transforms)
            parts.append(MemoryPart(f"arrays of the {len(transforms)} transforms", ram, gpu, 'held'))
            image = 4 * math.prod(self.object_meta.shape)
            parts.append(MemoryPart("a transform's temporaries", **on_device(pytomography.device, 12 * image), scope='chunk'))
        shape = getattr(self.proj_meta, 'shape', None)
        if shape is not None:
            parts.append(MemoryPart(f"projections of ones of all {N} (for H^T 1)", **on_device(pytomography.device, 4 * N * math.prod(shape)), scope='subset'))
        return parts

    def compute_normalization_factor(self, subset_idx: int | None = None):
        r"""Function called by reconstruction algorithms to get the normalization factor :math:`H' = \sum_n v_n^T \otimes A_n^T H_n^T B_n^T` 1.

        Returns:
           torch.Tensor[1,Lx,Ly,Lz]: Normalization factor.
        """
        norm_proj = torch.ones((len(self.system_matrices), *self.proj_meta.shape)).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = norm_proj[:,self.subset_indices_array[subset_idx]]
        return self.backward(norm_proj, subset_idx)
