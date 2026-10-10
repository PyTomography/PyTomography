from __future__ import annotations
import numpy as np
import torch
import pytomography
from pytomography.projectors import SystemMatrix
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
import parallelproj_core
from . import _wfbp

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

def _saving_memory(system_matrix, candidates: list, n_subsets: int, N_splits: int | None) -> list:
    """The (label, n_subsets, N_splits) ``candidates`` with which ``system_matrix._memory_parts`` saves RAM or GPU
    memory compared with ``n_subsets`` and ``N_splits``: at least 5% of the two together, so that a few kilobytes of
    one are not offered against far more of the other."""
    def totals(n, s):
        parts = system_matrix._memory_parts(n, s)
        return sum(p.ram_bytes for p in parts), sum(p.gpu_bytes for p in parts)
    ram, gpu = totals(n_subsets, N_splits)
    least = 0.05 * (ram + gpu)
    return [c for c in candidates if (lambda r, g: ram - r >= least or gpu - g >= least)(*totals(c[1], c[2]))]

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

    def _fbp(self, projections: torch.Tensor, filter, slice_thickness: float | None = None, gpu_budget: float | None = None,
             Q: float = 0.6, k_range: int | None = None, stats: dict | None = None, backend: str = 'auto',
             column_weights: torch.Tensor | None = None) -> torch.Tensor:
        r"""Helical filtered back projection onto the object grid of this system matrix, called by
        :class:`pytomography.algorithms.FilteredBackProjection`. The fan projections are rebinned to parallel beams and
        reconstructed by weighted filtered back projection (WFBP, Stierstorfer et al. 2004), which also handles circular
        scans. With a flying focal spot, each group of views sharing a focal spot offset is reconstructed on its own and
        the groups are averaged. Voxels outside the field of view are zero, as in :meth:`forward`.

        Host memory: besides the projections and the image, the reconstruction holds the rebinned projections of a chunk
        of views at a time, at most an eighth of the memory budget (:func:`pytomography.set_memory_budget`), or 1 GB
        without one. A smaller budget means more chunks, which is slower; the image is the same.

        Args:
            projections (torch.Tensor): Line integrals (views, columns, rows), on any device.
            filter (FBPFilter): Window applied on top of the ramp filter.
            slice_thickness (float, optional): Average every slice over a box of this width (mm) along z, sampled every
                0.25 mm, on top of the reconstruction's own axial resolution. Defaults to None (no averaging).
            gpu_budget (float, optional): Bytes the reconstruction may hold on the device; never more than a quarter of
                the free GPU memory (see :func:`pytomography.utils.gpu_budget`). Defaults to 1.5 GB.
            Q (float, optional): WFBP row weighting: rows within the central fraction Q of the detector get full
                weight, falling to zero at its edges. Defaults to 0.6.
            k_range (int, optional): Half turns either side searched for rays through the same voxel. Defaults to
                what the pitch and cone angle allow.
            stats (dict, optional): Receives the time and peak GPU memory of each focal spot group.
            backend (str, optional): ``'auto'`` back projects with a fused CUDA kernel when CuPy is installed and the
                device is a GPU (much faster, and lighter on memory), and with PyTorch otherwise; ``'cuda'`` or
                ``'torch'`` force one. Defaults to ``'auto'``.
            column_weights (torch.Tensor, optional): A weight for each detector column, multiplying the line
                integrals as they are read: the reconstruction of the weighted projections, without a weighted copy of
                them (:func:`pytomography.io.CT.preprocessing.fit_column_scale` uses it). Defaults to None.

        Returns:
            torch.Tensor: Attenuation per mm on the object grid, on ``pytomography.device``.
        """
        X, Y, z, z_offsets = self._fbp_grid(slice_thickness)
        image = _wfbp.fbp_helical(projections, self.proj_meta, X, Y, z, window=filter, z_offsets=z_offsets, Q_weight=Q,
                                  k_range=k_range, budget=gpu_budget, device=pytomography.device, stats=stats, backend=backend,
                                  column_weights=column_weights)
        return image if self._fov is None else image.mul_(self._fov.to(image.device))

    def _fbp_grid(self, slice_thickness: float | None) -> tuple:
        """The points :meth:`_fbp` reconstructs, X and Y (Nx, Ny) and z (Nz), at the voxel centres as the projector
        places them (see ``origin``), and the offsets along z of the sub-slices that make up ``slice_thickness``."""
        (Nx, Ny, Nz), (dx, dy, dz) = self.object_meta.shape, self.object_meta.dr
        x = (torch.arange(Nx) - (Nx - 1) / 2) * dx
        y = (torch.arange(Ny) - (Ny - 1) / 2) * dy
        z = (np.arange(Nz) - (Nz - 1) / 2) * dz
        X, Y = torch.meshgrid(x, y, indexing='ij')
        if slice_thickness:
            n = max(1, int(np.ceil(slice_thickness / 0.25 - 1e-9)))
            z_offsets = tuple(((np.arange(n) + 0.5) / n - 0.5) * slice_thickness)
        else:
            z_offsets = (0.0,)
        return X, Y, z, z_offsets

    def _fbp_memory_parts(self, projections: torch.Tensor, filter, slice_thickness: float | None = None,
                          gpu_budget: float | None = None, Q: float = 0.6, k_range: int | None = None,
                          stats: dict | None = None, backend: str = 'auto', column_weights: torch.Tensor | None = None) -> list:
        """The memory :meth:`_fbp` takes with the same arguments, without running it, for
        :meth:`pytomography.algorithms.FilteredBackProjection.estimate_memory` (see :func:`._wfbp.memory_parts`)."""
        X, Y, z, z_offsets = self._fbp_grid(slice_thickness)
        return _wfbp.memory_parts(projections, self.proj_meta, X, Y, z, z_offsets=z_offsets, k_range=k_range,
                                  budget=gpu_budget, device=pytomography.device, backend=backend)

    def _fbp_memory_alternatives(self, projections: torch.Tensor, filter, gpu_budget: float | None = None, **options) -> list:
        """Ways to run :meth:`_fbp` in less memory, as (label, memory parts): a lower memory budget, which halves the
        chunk of rebinned views on the host, and half the GPU budget. Both take longer and give the same image; each is
        offered only if it saves memory."""
        from pytomography.utils.memory import memory_budget_set, DEFAULT_BUDGET
        total = lambda parts: sum(p.ram_bytes + p.gpu_bytes for p in parts)
        parts = self._fbp_memory_parts(projections, filter, gpu_budget=gpu_budget, **options)
        chunk = sum(p.ram_bytes for p in parts if p.name == 'rebinned views of one chunk')
        alternatives = []
        gb = float(f'{4 * chunk / 1e9:.1g}')                   # chunks of at most budget / 8: half the present chunk
        if gb > 0:
            with memory_budget_set(gb):
                fewer = self._fbp_memory_parts(projections, filter, gpu_budget=gpu_budget, **options)
            if total(fewer) < total(parts):
                alternatives.append((f'set_memory_budget({gb:g})', fewer))
        half = (DEFAULT_BUDGET if gpu_budget is None else gpu_budget) / 2
        smaller = self._fbp_memory_parts(projections, filter, gpu_budget=half, **options)
        if total(smaller) < total(parts):
            alternatives.append((f'gpu_budget={half / 1e9:g}e9', smaller))
        return alternatives

    def _memory_parts(self, n_subsets: int = 1, N_splits: int | None = None) -> list:
        r"""The arrays of an ordered-subset reconstruction with this system matrix, as OS-SART takes them, for
        :meth:`estimate_memory` (which adds the projections, passed as ``held``). Held for the whole reconstruction:
        the image, and one :math:`H_m^T 1` per subset, which :meth:`compute_normalization_factor` keeps on the host, an
        image each, so that more subsets take more memory. For each subset: its projections (measured, predicted, of
        ones, and their ratio) and the images of the update. For each projector call: the end points and values of its
        rays (28 bytes a ray, at most :data:`_MAX_RAYS_PER_CALL` rays) and the padded image. The largest calls are those
        of the initial object (:meth:`_get_object_initial`), which back projects every view in ``N_splits`` calls."""
        from pytomography.utils.memory import MemoryPart
        N_splits = self.N_splits if N_splits is None else N_splits
        n = max(1, n_subsets)
        image = int(np.prod(self.object_meta.shape)) * 4
        padded = int(np.prod([s + 2 for s in self.object_meta.shape])) * 4
        rays_per_view = int(np.prod(self.proj_meta.shape))
        views = -(-self.proj_meta.N_angles // n)                    # in the largest subset
        calls = lambda v: max(N_splits, -(-v * rays_per_view // _MAX_RAYS_PER_CALL))
        rays = max(-(-v // calls(v)) * rays_per_view for v in (views, self.proj_meta.N_angles))
        on = lambda device, b: dict(gpu_bytes=b) if torch.device(device).type == 'cuda' else dict(ram_bytes=b)
        return [MemoryPart('image', **on(pytomography.device, image), scope='held'),
                MemoryPart(f'H_m^T 1 of {n} subset{"s" if n > 1 else ""}', ram_bytes=n * image, scope='held'),
                MemoryPart('projections of a subset: measured, predicted, of ones, ratio',
                           **on(self.device, 4 * views * rays_per_view * 4), scope='subset'),
                MemoryPart('images of an update', **on(pytomography.device, 3 * image), scope='subset'),
                MemoryPart('projector call: rays and the padded image', **on(pytomography.device, 28 * rays + padded + image),
                           scope='chunk')]

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list:
        """Half and twice the subsets, and twice ``N_splits``, where they take less memory. Each subset keeps an
        image on the host while its projections shrink with more subsets, so which way saves memory depends on the
        scan; more projector calls take less on the GPU."""
        N_splits = self.N_splits if N_splits is None else N_splits
        candidates = [(f'{n_subsets // 2} subsets', n_subsets // 2, N_splits)] if n_subsets >= 2 else []
        candidates += [(f'{2 * n_subsets} subsets', 2 * n_subsets, N_splits), (f'N_splits {2 * N_splits}', n_subsets, 2 * N_splits)]
        return _saving_memory(self, candidates, n_subsets, N_splits)

    def forward(self, object, subset_idx=None):
        r"""Computes forward projection

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
    
    def backward(self, proj, subset_idx=None):
        """Computes back projection :math:`H^T g` (for filtered back projection, use
        :class:`pytomography.algorithms.FilteredBackProjection`)

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