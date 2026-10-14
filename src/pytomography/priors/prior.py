"""The base class of the priors."""
from __future__ import annotations
from collections.abc import Sequence
import torch
import abc
import pytomography
from pytomography.metadata import ObjectMeta

class Prior():
    r"""The base class of the priors :math:`V(f)`, the penalty in the log-posterior :math:`\ln L(g|f) - \beta V(f)`.

    Subclasses implement ``__call__``, which gives the prior or its derivatives at the object set with
    :meth:`set_object`. The reconstruction algorithm gives the prior its object grid (:meth:`set_object_meta`), the
    scale of the field of view (:meth:`set_FOV_scale`), the current object and the subset's share of the data
    (:meth:`set_beta_scale`).

    Args:
        beta (float): The weight :math:`\beta` of the prior.
        obj2obj_transforms (Sequence[Transform], optional): Transforms applied, in order, to the prior's value and
            derivatives at every voxel (for example a :class:`~pytomography.transforms.shared.GaussianFilter`),
            configured with the object grid in :meth:`set_object_meta`. Defaults to none.
    """
    @abc.abstractmethod
    def __init__(self, beta: float, obj2obj_transforms: Sequence[Transform] = ()):
        self.beta = beta
        self.obj2obj_transforms = list(obj2obj_transforms)
        self.device = pytomography.device

    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Gives the prior the object grid, and configures its transforms with it.

        Args:
            object_meta (ObjectMeta): The object grid.
        """
        self.object_meta = object_meta
        for transform in self.obj2obj_transforms:
            transform.configure(object_meta, None)

    def set_beta_scale(self, factor: float) -> None:
        r"""Sets the factor that scales :math:`\beta` in the derivatives: the subset's share of the data, with subsets.

        Args:
            factor (float): The factor (``system_matrix.get_weighting_subset(subset_idx)``).
        """
        self.beta_scale_factor = factor
        
    def set_FOV_scale(self, FOV_scale: torch.Tensor) -> None:
        r"""Sets a scale of the prior that depends on the position in the field of view.

        Args:
            FOV_scale (torch.Tensor): The scale, [Lx, Ly, Lz] (from the system matrix: ones, or the axial scale of a PET
                list mode system matrix).
        """
        self.FOV_scale = FOV_scale
                

    def set_object(self, object: torch.Tensor) -> None:
        r"""Sets the object :math:`f` at which the prior and its derivatives are computed (a copy is kept).

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].
        """
        self.object = object.clone()

    @abc.abstractmethod
    def __call__(self):
        """Computes the prior or its derivatives at the object set with :meth:`set_object`; subclasses implement it."""
        ...