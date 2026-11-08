"""The base classes of the system matrices: the interface every projector implements, and several stacked into one.

:class:`SystemMatrix` is what the likelihoods and reconstruction algorithms call; :class:`ExtendedSystemMatrix`
stacks several system matrices (e.g. one per gate or per time frame) into one projection space.
"""
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
    """The arrays a computation holds for its whole run, as memory parts for the memory estimates.

    Args:
        held (Sequence | Mapping): A list of arrays (tensors or :class:`~pytomography.io.PET.LazySinogram`), or a dict
            of names to arrays. A :class:`~pytomography.utils.memory.MemoryPart` among them (for arrays not made yet)
            is taken as it is.

    Returns:
        list[MemoryPart]: One part per array, of scope ``'held'``.
    """
    items = held.items() if isinstance(held, Mapping) else ((f"array {k + 1}", x) for k, x in enumerate(held))
    return [x if isinstance(x, MemoryPart) else MemoryPart(name, *nbytes(x), scope='held') for name, x in items]


def on_device(device, nbytes: float) -> dict:
    """The keyword argument of a :class:`~pytomography.utils.memory.MemoryPart` for an array on ``device``.

    Args:
        device (str | torch.device): Where the array is.
        nbytes (float): Its size, in bytes.

    Returns:
        dict: ``{'gpu_bytes': nbytes}`` for a CUDA device, ``{'ram_bytes': nbytes}`` otherwise.
    """
    return {'gpu_bytes': float(nbytes)} if torch.device(device).type == 'cuda' else {'ram_bytes': float(nbytes)}

class SystemMatrix():
    r"""The base class of every system matrix :math:`H:\mathbb{U} \to \mathbb{V}`, from objects to projections.

    A system matrix maps an object :math:`f \in \mathbb{U}` to the projections :math:`g \in \mathbb{V}` the imaging
    system would measure (:meth:`forward`), and projections back to object space with the transpose :math:`H^T`
    (:meth:`backward`), which iterative reconstruction algorithms such as OSEM need. Object-to-object and
    projection-to-projection transforms (:mod:`pytomography.transforms`) model the system's physics, such as
    attenuation and blurring.

    Subclasses implement :meth:`forward` and :meth:`backward` (with an optional subset index ``subset_idx``), and, for
    reconstructions with subsets, ``set_n_subsets``, ``get_projection_subset``, ``get_weighting_subset`` and
    ``compute_normalization_factor`` (:math:`H_m^T 1`). Multiplying a system matrix by a number (``2 * H``) gives a
    new system matrix whose forward and back projections are scaled; ``H`` itself is unchanged.

    Args:
        object_meta (ObjectMeta): The object grid.
        proj_meta (ProjMeta): The projection space.
        obj2obj_transforms (list[Transform], optional): Object-to-object transforms applied before forward projection
            (and their transposes after back projection), in order. Defaults to none.
        proj2proj_transforms (list[Transform], optional): Projection-to-projection transforms applied after forward
            projection (and their transposes before back projection), in order. Defaults to none.
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
        r"""The system matrix scaled by a number, :math:`cH`, as a new system matrix; this one is unchanged.

        The scaled system matrix is a shallow copy: it shares this one's transforms, metadata and data, so it takes no
        memory of its own, and it keeps its own subsets. Its forward and back projections are this class's, times
        ``scalar``; scaling it again multiplies the factors.

        Args:
            scalar (float | torch.Tensor): The factor :math:`c`, e.g. a sensitivity in counts per second per MBq.

        Returns:
            SystemMatrix: The scaled system matrix.
        """
        scaled = copy(self)
        scaled._scale = getattr(self, '_scale', 1) * scalar
        forward, backward = type(self).forward, type(self).backward
        scaled.forward = lambda *args, **kwargs: forward(scaled, *args, **kwargs) * scaled._scale
        scaled.backward = lambda *args, **kwargs: backward(scaled, *args, **kwargs) * scaled._scale
        return scaled

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def initialize_transforms(self):
        """Configures every transform of the system matrix with its ``object_meta`` and ``proj_meta``."""
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
        """The predicted peak RAM and GPU memory of a reconstruction with this system matrix.

        The estimate is for ``n_subsets`` subsets and ``N_splits`` projector splits, under the current memory budget
        (:func:`pytomography.set_memory_budget`). ``print`` it to see the peak, what it is made of, and what more
        subsets or splits would need ("To use less"); ``ram_gb`` and ``gpu_gb`` give the numbers, ``to_dict()`` all of
        it as JSON. On Windows, GPU memory counts as RAM too. For the opposite question, which settings fit in the
        memory there is, see :meth:`plan_memory`.

        Args:
            n_subsets (int, optional): Number of subsets. Defaults to 1.
            N_splits (int | None, optional): Projector splits, for system matrices that take them. Defaults to None: this
                system matrix's own.
            held (Sequence | Mapping, optional): Other arrays the reconstruction holds for the whole run, such as its
                projections and additive term (tensors or :class:`~pytomography.io.PET.LazySinogram`), as a list
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
        """The fastest settings of a reconstruction with this system matrix that fit in the given RAM and GPU memory.

        The result is a :class:`~pytomography.utils.memory.MemoryPlan` (``plan.settings``, ``plan.fits``,
        ``plan.estimate``; ``print`` it, or ``plan.to_dict()`` for JSON). This is the opposite question to
        :meth:`estimate_memory`: memory in, settings out.

        Settings passed by name are kept (e.g. ``n_subsets=14``); the others are chosen. The fewest projector splits
        ``N_splits`` come first, then the fewest subsets that fit with them: every number of subsets is tried in order,
        since memory need not fall with more subsets (a CT or SPECT reconstruction keeps an :math:`H_m^T 1` image per
        subset). The settings tried are, unless a system matrix has others, the number of subsets (one to the number of
        views or angles, at most :data:`~pytomography.utils.memory.PLAN_MAX_SUBSETS`) and, for system matrices that
        take them, the projector splits ``N_splits`` (1, 2, 4... 256 and the present value). Without budgets, the plan
        is for what is free on this machine now (:func:`~pytomography.utils.memory.available_memory`, which keeps a margin
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
            ValueError: If ``fixed`` has a setting that this system matrix does not have.

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
        """The fewest subsets for which :meth:`estimate_memory` predicts at most ``ram_gb`` GB of RAM.

        The numbers are tried in order (an estimate is arithmetic), since memory need not fall with more subsets (a CT
        reconstruction keeps a normalisation image per subset). It is :meth:`plan_memory` with the projector splits
        kept and no GPU limit.

        Args:
            ram_gb (float): RAM available, in GB.
            N_splits (int | None, optional): Projector splits. Defaults to None: this system matrix's own.
            held (Sequence | Mapping, optional): As for :meth:`estimate_memory`. Defaults to none.
            max_subsets (int, optional): Most subsets to consider. Defaults to 1024.

        Returns:
            int | None: Number of subsets, or None if no number up to ``max_subsets`` fits.
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
        r"""Forward projection :math:`H_m f` of an object :math:`f`, implemented by each subclass.

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].
            **kwargs: The subclasses take a subset index ``subset_idx`` (:math:`m`, see ``set_n_subsets``; None for
                all the projections).

        Returns:
            torch.Tensor: The projections, in the shape of ``proj_meta`` (only the subset's with ``subset_idx``).
        """
        ...
    @abc.abstractmethod
    def backward(
        self,
        proj: torch.tensor,
        angle_subset: list | None = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        r"""Back projection :math:`H_m^T g` of projections :math:`g`, implemented by each subclass.

        Args:
            proj (torch.Tensor): The projections to back project (only the subset's with a subset index).
            angle_subset (list, optional): The subset: the subclasses take a subset index ``subset_idx`` here (see
                ``set_n_subsets``); None for all the projections. Defaults to None.
            return_norm_constant (bool, optional): Also return the normalization :math:`H_m^T 1`, for the subclasses
                that support it. Defaults to False.

        Returns:
            torch.Tensor: The object, [Lx, Ly, Lz] (and :math:`H_m^T 1` with ``return_norm_constant``).
        """
        ...
    @abc.abstractmethod
    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        """A partition of the projections into subsets; no system matrix implements or calls it.

        PyTomography's system matrices partition their projections with ``set_n_subsets(n_subsets)`` instead, which
        the likelihoods call.

        Args:
            n_subsets (int): Number of subsets.

        Returns:
            list: One array of projection indices per subset.
        """
        ...

    def get_weighting_subset(self, subset_idx: int | None) -> float:
        r"""The share of the projections in subset :math:`m`; 1 without subsets.

        It scales parameters relative to :math:`H_m^T 1` in reconstruction algorithms (a prior's :math:`\beta`,
        BSREM's step). The subsets ``set_n_subsets`` makes partition the projections, so this is the number of
        projections in the subset over the number in all of them: for StarGuide, whose subsets are groups of angles, the
        views of the subset's angles over all views. System matrices whose projections do not all weigh the same
        override it.

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
    r"""Several system matrices stacked into one, whose projections have an extra first axis: [N, ...].

    The forward projection is :math:`H' = \sum_n v_n \otimes B_n H_n A_n`, with :math:`H_n` the system matrices,
    :math:`A_n` object-to-object transforms, :math:`B_n` projection-to-projection transforms, :math:`v_n` the basis
    vector of length :math:`N` with 1 in component :math:`n` and :math:`\otimes` the tensor product. It models, for
    example, gated or dynamic data, or the same object seen through several motion states
    (:class:`~pytomography.transforms.shared.DVFMotionTransform`). The system matrices are assumed to share their
    object grid and their subsets (the first one's are used).

    Args:
        system_matrices (Sequence[SystemMatrix]): The system matrix :math:`H_n` of each component :math:`n`.
        obj2obj_transforms (Sequence[Transform], optional): The object-to-object transform :math:`A_n` of each
            component (None in the list for none). Defaults to None (no transforms).
        proj2proj_transforms (Sequence[Transform], optional): The projection-to-projection transform :math:`B_n` of
            each component (None in the list for none). Defaults to None (no transforms).
    """
    def __init__(
        self,
        system_matrices: Sequence[SystemMatrix],
        obj2obj_transforms: Sequence[Transform] = None,
        proj2proj_transforms: Sequence[Transform] = None,
        ) -> None:
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
        r"""Forward projection :math:`H' f = \sum_n v_n \otimes B_n H_n A_n f`, with an extra first axis.

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`; None for all the projections. Defaults to None.

        Returns:
            torch.Tensor: The projections of every component, [N, ...].
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
        r"""Back projection :math:`H'^T g = \sum_n A_n^T H_n^T B_n^T g_n`: every component back projected and summed.

        Args:
            proj (torch.Tensor): The projections of every component, [N, ...].
            subset_idx (int | None, optional): Subset index :math:`m`; None for all the projections. Defaults to None.

        Returns:
            torch.Tensor: The object, [Lx, Ly, Lz].
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
        """Partitions the projections of every system matrix into ``n_subsets`` subsets.

        The system matrices are assumed to partition alike: the first one's ``subset_indices_array`` is used.

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
        """The projections of subset ``subset_idx``, as the first system matrix takes them.

        Args:
            projections (torch.Tensor): The projections.
            subset_idx (int): Subset index :math:`m`.

        Returns:
            torch.Tensor: The subset's projections.
        """
        return self.system_matrices[0].get_projection_subset(projections, subset_idx)

    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> torch.tensor:
        """The relative weight of subset ``subset_idx`` (its share of the projections), from the first system matrix.

        Args:
            subset_idx (int): Subset index :math:`m`.

        Returns:
            float: The weight.
        """
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
        r"""The normalization :math:`H_m'^T 1 = \sum_n A_n^T H_n^T B_n^T 1`, which the reconstruction algorithms use.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`; None for all the projections. Defaults to None.

        Returns:
            torch.Tensor: The normalization, [Lx, Ly, Lz].
        """
        norm_proj = torch.ones((len(self.system_matrices), *self.proj_meta.shape)).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = norm_proj[:,self.subset_indices_array[subset_idx]]
        return self.backward(norm_proj, subset_idx)
