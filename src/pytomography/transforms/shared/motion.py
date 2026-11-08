"""Motion of an object by a deformation vector field (:class:`DVFMotionTransform`), for motion correction."""
from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform

class DVFMotionTransform(Transform):
    r"""Object-to-object transform that deforms an object by a deformation vector field, conserving its activity.

    The forward transform takes the object to the deformed frame: its value at voxel :math:`x` is the object at
    :math:`\phi(x) = x + u_f(x)` (trilinear interpolation, zero outside the grid), times the volume change of the
    deformation there, :math:`J(x) = |\det(I + \nabla u_f(x))|` (central differences, one-sided at the edges of the
    grid). A region that expands keeps its activity and its concentration drops, so the total activity is conserved.
    The backward transform is the adjoint of the forward one: the value at :math:`y` is the deformed object at
    :math:`\psi(y) = y + u_b(y)`, where :math:`\psi = \phi^{-1}` (the Jacobian cancels in the change of variables,
    so the backward transform has none). Give one per gate to
    :class:`~pytomography.projectors.shared.MotionSystemMatrix`; it applies to whole objects (not to the rotated
    objects of a SPECT system matrix's views).

    Args:
        dvf_forward (torch.Tensor | None, optional): Displacement :math:`u_f` of the forward transform, [Lx, Ly, Lz, 3],
            in voxels, with the components along x, y and z. Defaults to None: the forward transform is the identity.
        dvf_backward (torch.Tensor | None, optional): Displacement :math:`u_b` of the backward transform,
            [Lx, Ly, Lz, 3], in voxels: the inverse deformation of ``dvf_forward``
            (:math:`u_b(y) = -u_f(\psi(y))`). Defaults to None: the backward transform is the identity.
    """
    def __init__(
        self,
        dvf_forward: torch.Tensor | None = None,
        dvf_backward: torch.Tensor | None = None,
        )-> None:
        self.dvf_forward = None if dvf_forward is None else dvf_forward.to(pytomography.device).to(pytomography.dtype)
        self.dvf_backward = None if dvf_backward is None else dvf_backward.to(pytomography.device).to(pytomography.dtype)
        self.dvf_forward_vol_ratio = None if self.dvf_forward is None else self._get_vol_ratio(self.dvf_forward)
        super(DVFMotionTransform, self).__init__()  ## go to the _init_ in Class Transform

    @staticmethod
    def _get_vol_ratio(DVF: torch.Tensor) -> torch.Tensor:
        r"""Volume change :math:`|\det(I + \nabla u)|` of the deformation :math:`x \to x + u(x)` at each voxel, with the
        derivatives taken by central differences (one-sided at the edges of the grid).

        Args:
            DVF (torch.Tensor[Lx,Ly,Lz,3]): Displacement :math:`u` in voxels.

        Returns:
            torch.Tensor[Lx,Ly,Lz]: Volume ratio.
        """
        def derivative(u, axis):
            return torch.gradient(u, dim=axis)[0] if u.shape[axis] > 1 else torch.zeros_like(u)
        # J[i][j] = d(x_i + u_i)/dx_j
        J = [[derivative(DVF[..., i], j) + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
        det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) \
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) \
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
        return torch.abs(det)

    def _get_old_coordinates(self):
        """Obtain meshgrid of coordinates corresponding to the object

        Returns:
            torch.Tensor: Tensor of coordinates corresponding to input object
        """
        dim_x, dim_y, dim_z = self.object_meta.shape
        coordinates=torch.stack(torch.meshgrid(torch.arange(dim_x),torch.arange(dim_y), torch.arange(dim_z), indexing='ij')).permute((1,2,3,0)).to(pytomography.device).to(pytomography.dtype)
        return coordinates

    def _get_new_coordinates(self, old_coordinates: torch.Tensor, DVF: torch.Tensor):
        """Obtain the new coordinates of each voxel based on the DVF.

        Args:
            old_coordinates (torch.Tensor): Old coordinates of each voxel
            DVF (torch.Tensor): Deformation vector field.

        Returns:
            torch.Tensor: New coordinates, normalised to [-1, 1] for ``grid_sample``.
        """
        dimensions = torch.tensor(self.object_meta.shape).to(pytomography.device)
        new_coordinates = old_coordinates + DVF
        new_coordinates = 2/(dimensions-1)*new_coordinates - 1
        return new_coordinates

    def _apply_dvf(self, DVF: torch.Tensor, vol_ratio: torch.Tensor | None, object_i: torch.Tensor):
        """Applies the deformation vector field to the object

        Args:
            DVF (torch.Tensor): Deformation vector field
            vol_ratio (torch.Tensor | None): Volume ratio to multiply the deformed object by, or None.
            object_i (torch.Tensor): Old object.

        Returns:
            torch.Tensor: Deformed object.
        """
        old_coordinates = self._get_old_coordinates()
        new_coordinates = self._get_new_coordinates(old_coordinates, DVF)
        deformed = torch.nn.functional.grid_sample(object_i.unsqueeze(0).unsqueeze(0), new_coordinates.unsqueeze(0).flip(dims=[-1]), align_corners=True)[0, 0]
        # Adjust for stretching of object
        return deformed if vol_ratio is None else deformed * vol_ratio

    def forward(
        self,
        object_i: torch.Tensor,
    )-> torch.Tensor:
        """Deforms the object by ``dvf_forward``, times the volume change of the deformation (activity conserved).

        Args:
            object_i (torch.Tensor): The object, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: The deformed object, [Lx, Ly, Lz] (``object_i`` itself without ``dvf_forward``).
        """
        if self.dvf_forward is None:
            return object_i
        else:
            return self._apply_dvf(self.dvf_forward, self.dvf_forward_vol_ratio, object_i)

    def backward(
        self,
        object_i: torch.Tensor,
    )-> torch.Tensor:
        """Deforms the object by ``dvf_backward``: the adjoint of :meth:`forward` when it is the inverse deformation.

        Args:
            object_i (torch.Tensor): The object, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: The deformed object, [Lx, Ly, Lz] (``object_i`` itself without ``dvf_backward``).
        """
        if self.dvf_backward is None:
            return object_i
        else:
            return self._apply_dvf(self.dvf_backward, None, object_i)
