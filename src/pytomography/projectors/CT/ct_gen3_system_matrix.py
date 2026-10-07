from __future__ import annotations
import torch
import pytomography
from pytomography.projectors import SystemMatrix
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
import parallelproj_core

#: Most rays :class:`CTGen3SystemMatrix` hands to parallelproj in one call. Their coordinates are built on the GPU, at
#: about 28 bytes per ray, so a projection of every view of a clinical scan at once (half a billion rays, 15 GB) is
#: split into groups of views below this (about 1 GB); a subset of a typical reconstruction is a single call.
_MAX_RAYS_PER_CALL = 2**25

def _float32(x, device) -> torch.Tensor:
    """Contiguous float32 tensor on ``device``: the array form every parallelproj kernel expects."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device=device, dtype=torch.float32).contiguous()

def _pad(object: torch.Tensor) -> torch.Tensor:
    """The object with one voxel of zeros on every face, which is how the CT system matrices hand it to parallelproj."""
    return torch.nn.functional.pad(object, (1, 1, 1, 1, 1, 1))

def _crop(object: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pad`: removes the outer voxel on every face."""
    return object[1:-1, 1:-1, 1:-1].contiguous()

class CTGen3SystemMatrix(SystemMatrix):
    """System matrix for 3rd generation clinical DICOM scanners with cylindrical detector panels. For more information, see the DICOM-CTPD user manual.

        Args:
            object_meta (ObjectMeta): Metadata for object space
            proj_meta (CTConeBeamFlatPanelProjMeta): Projection metadata for the CT system
            N_splits (int, optional): Splits up computation of forward/back projection to save GPU memory. A projection is also split whenever needed to keep below ``_MAX_RAYS_PER_CALL`` (2**25) rays per call. Defaults to 1.
            device (str, optional): Device on which projections are output. Defaults to pytomography.device.
            fov_mask (bool, optional): Model only the voxels inside the scan field of view, the cylinder (radius ``fov_radius``) that the fan of every view covers. Voxels outside it are seen by some views only; a reconstruction cannot determine them, and left in the model they come out as large spurious values. With the mask they are kept at zero. Turn it off to forward project an object that extends past the field of view. Defaults to True.
    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: CTGen3ProjMeta,
        N_splits: int = 1,
        device: str = pytomography.device,
        fov_mask: bool = True
    ) -> None:
        super(CTGen3SystemMatrix, self).__init__(object_meta, proj_meta)
        # the geometry every kernel call needs, in the form it needs (float32, on the projection device).
        # parallelproj 2 clips each ray to the faces of the image it is given, which drops the half voxel of
        # interpolation outside the outermost voxel centres that parallelproj 1 included, so rays leaving the image
        # through a face where the object is not zero come out short (by 4.3 mm on average for the rays leaving a
        # helical chest scan through the axial faces). The object is therefore projected with one voxel of zeros on
        # every face (_pad), which gives the parallelproj 1 line integrals; origin is that of the padded grid.
        self.origin = _float32(-(torch.tensor(object_meta.shape)/2+0.5) * torch.tensor(object_meta.dr), pytomography.device)
        self.voxel_size = _float32(object_meta.dr, pytomography.device)
        self.N_splits = N_splits
        self.device = device
        # the scan field of view: the cylinder covered by the fan of every view (its outermost channels, from the focal
        # spot path's smallest radius)
        fan = proj_meta.phis_det[:, 0]
        self.fov_radius = float(proj_meta.source_rhos.min()) * float(torch.sin(torch.minimum(fan[0].abs(), fan[-1].abs())))
        self._fov = None
        if fov_mask:
            (Nx, Ny, _), (dx, dy, _) = object_meta.shape, object_meta.dr
            x = (torch.arange(Nx) - (Nx - 1) / 2) * dx
            y = (torch.arange(Ny) - (Ny - 1) / 2) * dy
            self._fov = _float32((x[:, None] ** 2 + y[None, :] ** 2 <= self.fov_radius ** 2)[:, :, None], pytomography.device)

    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`\tilde{H}_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

        Args:
            subset_idx (int): Subset index

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.N_angles
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. CT conebeam flat panel partitions projections based on angle.

        Args:
            projections (torch.Tensor): total projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.Tensor: subsampled projections :math:`g_m`.
        """
        if subset_idx is None:
            return projections
        else:
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices.to(projections.device)]
            return proj_subset
        
    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(self.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def _angle_indices(self, subset_idx: int | None) -> torch.Tensor:
        """Indices of the views in subset :math:`m`, or of every view when ``subset_idx`` is None."""
        if subset_idx is None:
            return torch.arange(self.proj_meta.N_angles)
        return self.subset_indices_array[subset_idx]

    def _splits(self, angle_indices: torch.Tensor) -> list:
        """The groups of views the projection is computed in, as (start, end, view indices), where start:end is the position of the group in the projections: ``N_splits`` groups, or more if needed to keep each below ``_MAX_RAYS_PER_CALL`` rays."""
        rays_per_view = self.proj_meta.shape[0] * self.proj_meta.shape[1]
        n_splits = max(self.N_splits, -(-angle_indices.shape[0] * rays_per_view // _MAX_RAYS_PER_CALL))
        splits, start = [], 0
        for idxs in torch.tensor_split(angle_indices, n_splits):
            if idxs.numel():
                splits.append((start, start + idxs.shape[0], idxs))
            start += idxs.shape[0]
        return splits

    def _ray_coordinates(self, idxs: torch.Tensor) -> tuple:
        """Coordinates of the focal spot (start) and the detector element (end) of every ray of the views ``idxs``, shaped (views, columns, rows, 3). They are computed on the device the projector runs on: building them on the host and copying them across took longer than the forward projection itself."""
        idxs = idxs.to(pytomography.device)
        xend = _float32(self.proj_meta.get_detector_coordinates(idxs), pytomography.device)
        xstart = _float32(self.proj_meta.source_focal_spots, pytomography.device)[idxs][:,None,None].expand(xend.shape)
        return xstart.contiguous(), xend

    def compute_normalization_factor(self, subset_idx: int | None = None):
        r"""Computes the normalization factor :math:`H^T 1`

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Normalization factor.
        """
        n_views = self._angle_indices(subset_idx).shape[0]
        # Put BP on cpu since we could potentially have a lot of them
        return self.backward(torch.ones(n_views, *self.proj_meta.shape, device=self.device), subset_idx).cpu()

    def _coverage(self) -> torch.Tensor:
        r""":math:`H^T 1` over every view, back projected one group of views at a time (without forming the projections of ones of every view, 2 GB for a clinical scan)."""
        BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        for start, end, idxs in self._splits(self._angle_indices(None)):
            xstart, xend = self._ray_coordinates(idxs)
            ones = torch.ones(xend.shape[:-1], dtype=torch.float32, device=pytomography.device)
            parallelproj_core.joseph3d_back(xstart, xend, BP, self.origin, self.voxel_size, ones)
        BP = _crop(BP)
        return BP if self._fov is None else BP * self._fov

    def _get_object_initial(self, device=None) -> torch.Tensor:
        """Initial object of reconstruction algorithms: ones where rays reach and zeros elsewhere (outside the field of view, and in end slices beyond the axial reach of the scan). Voxels no ray reaches are never updated, so they used to keep the value 1."""
        if device is None:
            device = pytomography.device
        return (self._coverage() > 0).to(torch.float32).to(device)

    def forward(self, object, subset_idx=None, *args, **kwargs):
        """Computes forward projection

        Args:
            object (torch.Tensor): Object to be forward projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.

        Returns:
            torch.Tensor: Projections corresponding to :math:`\int \mu dx` along all LORs.
        """
        angle_indices = self._angle_indices(subset_idx)
        object = _float32(object, pytomography.device)
        if self._fov is not None:
            object = object * self._fov
        object = _pad(object)
        # Project into one buffer: parallelproj writes the line integrals of each split into the array it is given
        proj = torch.zeros((angle_indices.shape[0], *self.proj_meta.shape), dtype=torch.float32, device=self.device)
        for start, end, idxs in self._splits(angle_indices):
            xstart, xend = self._ray_coordinates(idxs)
            proj_i = torch.zeros(xend.shape[:-1], dtype=torch.float32, device=pytomography.device)
            parallelproj_core.joseph3d_fwd(xstart, xend, object, self.origin, self.voxel_size, proj_i)
            proj[start:end] = proj_i.to(self.device)
        return proj
    
    def backward(self, proj, subset_idx=None, *args, **kwargs):
        """Computes back projection

        Args:
            proj (torch.Tensor): Projections to be back projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.

        Returns:
            torch.Tensor: Back projection, on ``pytomography.device``.
        """
        angle_indices = self._angle_indices(subset_idx)
        # parallelproj adds into the image it is given, so every split accumulates into one (padded) buffer
        BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        for start, end, idxs in self._splits(angle_indices):
            xstart, xend = self._ray_coordinates(idxs)
            proj_i = _float32(proj[start:end], pytomography.device)
            parallelproj_core.joseph3d_back(xstart, xend, BP, self.origin, self.voxel_size, proj_i)
        BP = _crop(BP)
        return BP if self._fov is None else BP * self._fov