"""System matrix of cone-beam CT scans with a flat panel detector."""
from __future__ import annotations
import torch
import pytomography
import numpy as np
from pytomography.projectors import SystemMatrix
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
from pytomography.utils.fourier_filters import get_fbp_filter, ramp_filter
import parallelproj_core
from .ct_gen3_system_matrix import _float32, _pad, _crop, _saving_memory

class CTConeBeamFlatPanelSystemMatrix(SystemMatrix):
    """System matrix of a cone-beam CT scan with a flat panel detector.

    The forward projection of an object (linear attenuation coefficients in 1/mm) is its line integrals along the
    rays from each view's source to the centres of the detector pixels, computed by Joseph's method with parallelproj
    on ``pytomography.device``, one view at a time. The geometry is that of
    :class:`~pytomography.metadata.CT.CTConeBeamFlatPanelProjMeta`; the object grid is centred on the origin of its
    frame.

    :class:`~pytomography.algorithms.FilteredBackProjection` reconstructs circular scans (fixed z) with this system
    matrix by FDK: each projection is cosine weighted, ramp filtered along u (times the window), back projected, and
    weighted with the FDK distance weight. It needs views evenly spaced over a full turn at one z (there are no
    short-scan weights) and raises otherwise. It takes no options besides ``filter``.

    Args:
        object_meta (ObjectMeta): Grid of the object.
        proj_meta (CTConeBeamFlatPanelProjMeta): Geometry of the scan.
        N_splits (int, optional): Number of parallelproj calls the forward and the back projection of each view are
            split into. The rays of the whole view are built either way, so it does not lower the memory. Defaults to
            1.
        device (str, optional): Device the projections are returned on; the projector itself runs on
            ``pytomography.device``. Defaults to ``pytomography.device`` (its value when this module was imported).
    """
    #: Memory estimates: this system matrix projects with parallelproj, whose CUDA build makes a CUDA context (and
    #: runs on the GPU) also when PyTomography's device is the CPU, if a CUDA device is visible
    #: (:func:`~pytomography.utils.memory.projector_uses_gpu`).
    memory_cuda_projector = True

    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: CTConeBeamFlatPanelProjMeta,
        N_splits: int = 1,
        device: str = pytomography.device
    ) -> None:
        super(CTConeBeamFlatPanelSystemMatrix, self).__init__(object_meta, proj_meta)
        # the geometry every kernel call needs, in the form it needs (float32, on the projection device). The object is
        # projected with one voxel of zeros on every face (see CTGen3SystemMatrix); origin is that of the padded grid.
        self.origin = _float32(-(torch.tensor(object_meta.shape)/2+0.5) * torch.tensor(object_meta.dr), pytomography.device) # + proj_meta.COR
        self.voxel_size = _float32(object_meta.dr, pytomography.device)
        self.N_splits = N_splits
        self.device = device
        self._FBP_postweight_component1 = None
        self._FBP_preweight = None
    
    def _get_FBP_scale(self):
        return 0.5 * (2 * np.pi/ self.proj_meta.N_angles) * (self.proj_meta.DSD/self.proj_meta.DSO)
    
    def _get_FBP_preweight(self, idx):
        if self._FBP_preweight is None:
            s, v = self.proj_meta._get_detector_pixel_s_v(self.device)
            self._FBP_preweight = (self.proj_meta.DSD / torch.sqrt(s**2 + v**2 + self.proj_meta.DSD**2)).to(self.device)
        return self._FBP_preweight
    
    def _get_FBP_postweight(self, idx):
        # Postweight put on pytomography.device, not self.device (otherwise too slow)
        Nx, Ny, Nz = self.object_meta.shape
        dx, dy, dz = self.object_meta.dr
        du, dv = self.proj_meta.dr
        # the voxel centres, and the rotation axis, which is at COR[:2] = (0, -COR) of the object frame
        ax, ay = self.proj_meta.COR[:2].to(pytomography.device)
        x = (torch.arange(-Nx/2+0.5, Nx/2+0.5, 1)*dx).to(pytomography.device)
        y = (torch.arange(-Ny/2+0.5, Ny/2+0.5, 1)*dy).to(pytomography.device)
        z = (torch.arange(-Nz/2+0.5, Nz/2+0.5, 1)*dz).to(pytomography.device)
        # Typical post-weight from FDK algorithm: (DSO / U)^2, with U the voxel's distance from the source along the
        # central ray, DSO + (x - ax) cos(theta) + (y - ay) sin(theta) measured from the rotation axis. (Before
        # PyTomography 4 the axis was added instead of subtracted, which put U off by 2 COR sin(theta).)
        if self._FBP_postweight_component1 is None:
            xv, yv = torch.meshgrid(x - ax, y - ay, indexing='ij')
            post_weight = (self.proj_meta.DSO / (self.proj_meta.DSO + yv.unsqueeze(0) * torch.sin(self.proj_meta.angles.to(pytomography.device)).reshape((-1,1,1)) + xv.unsqueeze(0) * torch.cos(self.proj_meta.angles.to(pytomography.device)).reshape((-1,1,1))))**2
            self._FBP_postweight_component1 = post_weight.unsqueeze(-1)
        # Weight that removes length scaling Joseph projector to make projector "unmatched" (see Ander Biguri thesis chapter 4).
        # The length of l = (lx, ly, lz) and its product with d are summed over broadcast axes: three images at most,
        # not the (Nx, Ny, Nz, 3) vectors and their products.
        d = -self.proj_meta.detector_orientations[idx].to(pytomography.device)
        source_pos = self.proj_meta.beam_locations[idx].to(pytomography.device)
        lx = (x - source_pos[0])[:, None, None]
        ly = (y - source_pos[1])[None, :, None]
        lz = (z - source_pos[2])[None, None, :]
        w = (lx ** 2 + ly ** 2 + lz ** 2).sqrt_()
        w.div_((lx * d[0] + ly * d[1] + lz * d[2]).pow_(3)).mul_(self.proj_meta.DSD**2 * dx*dy*dz / (du*dv))
        return self._FBP_postweight_component1[idx] / w
    
    def set_n_subsets(self, n_subsets: int) -> None:
        r"""Partition the views into ``n_subsets`` subsets for ordered-subset reconstruction.

        Subset :math:`m` holds views :math:`m, m + n, m + 2n, \dots` (:math:`n` = ``n_subsets``). The indices are kept
        in ``subset_indices_array``, on ``device``.

        Args:
            n_subsets (int): Number of subsets, from 1 to the number of views.

        Raises:
            ValueError: If ``n_subsets`` is less than 1 or more than the number of views.
        """
        self._check_n_subsets(n_subsets, self.proj_meta.N_angles)
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(self.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
        """Select the projections :math:`g_m` of the views in subset :math:`m`.

        Args:
            projections (torch.Tensor): Projections :math:`g` of every view, [views, u, v].
            subset_idx (int | None): Subset index :math:`m`, or None for every view.

        Returns:
            torch.Tensor: Projections of the subset's views, [views in the subset, u, v], on the device of
            ``projections`` (``projections`` itself for None).
        """
        if subset_idx is None:
            return projections
        else:
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices.to(projections.device)]   # projections may stay on the CPU
            return proj_subset
        
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""Compute the fraction of the views in subset :math:`m`.

        Reconstruction algorithms scale parameters that are relative to :math:`H_m^T 1`, such as the prior weight
        :math:`\beta`, by it.

        Args:
            subset_idx (int | None): Subset index :math:`m`, or None for all the views.

        Returns:
            float: Number of views in the subset over the number of views (1 for None).
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.N_angles
        
    def compute_normalization_factor(self, subset_idx: int = None):
        r"""Compute the normalization factor :math:`H_m^T 1`, the back projection of ones over subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: every view.

        Returns:
            torch.Tensor: The normalization factor, [Lx, Ly, Lz], float32, on ``pytomography.device``.
        """
        n_views = self.proj_meta.N_angles if subset_idx is None else len(self.subset_indices_array[subset_idx])
        return self.backward(torch.ones(n_views, *self.proj_meta.shape, device=self.device), subset_idx)
    
    def forward(
        self, object: torch.Tensor,
        subset_idx: int | None = None,
    ) -> torch.Tensor:
        r"""Forward project an object: :math:`H_m f`, the line integrals :math:`\int \mu\,dl` along every ray.

        Args:
            object (torch.Tensor): Linear attenuation coefficients :math:`\mu` on the object grid, [Lx, Ly, Lz], in
                1/mm.
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: every view.

        Returns:
            torch.Tensor: Line integrals, [views, u, v] (the subset's views), float32, on ``device``.
        """
        angle_indices = self._angle_indices(subset_idx)
        object_i = _pad(_float32(object, pytomography.device))
        # Project into one buffer: parallelproj writes the line integrals of each angle into the array it is given
        proj = torch.zeros((len(angle_indices), *self.proj_meta.shape), dtype=torch.float32, device=self.device)
        for i in range(len(angle_indices)):
            beam_coordinate, detector_coordinates = self._rays_of_view(angle_indices[i])
            proj_i = torch.zeros(detector_coordinates.shape[0], dtype=torch.float32, device=pytomography.device)
            # the rays of the view in N_splits kernel calls, as back projection does; each ray is computed on its own
            for beam_coordinate_s, detector_coordinates_s, proj_s in zip(torch.tensor_split(beam_coordinate, self.N_splits), torch.tensor_split(detector_coordinates, self.N_splits), torch.tensor_split(proj_i, self.N_splits)):
                parallelproj_core.joseph3d_fwd(beam_coordinate_s, detector_coordinates_s, object_i, self.origin, self.voxel_size, proj_s)
            proj[i] = proj_i.reshape(self.proj_meta.shape).to(self.device)
        return proj
    
    def _angle_indices(self, subset_idx: int | None) -> torch.Tensor:
        """Indices of the views in subset :math:`m`, or of every view when ``subset_idx`` is None."""
        if subset_idx is None:
            return torch.arange(self.proj_meta.N_angles).to(pytomography.device)
        return self.subset_indices_array[subset_idx]

    def _rays_of_view(self, idx) -> tuple:
        """Focal spot and detector element coordinates of every ray of view ``idx``, flattened, on ``pytomography.device``."""
        detector_coordinates = _float32(self.proj_meta._get_detector_coordinates(idx), pytomography.device).flatten(end_dim=1)
        beam_coordinate = _float32(self.proj_meta.beam_locations[idx], pytomography.device).expand(detector_coordinates.shape).contiguous()
        return beam_coordinate, detector_coordinates

    def _back_project_view(self, BP: torch.Tensor, idx, proj_i: torch.Tensor) -> None:
        """Adds the (Joseph) back projection of the projection ``proj_i`` of view ``idx`` into the padded image ``BP``."""
        beam_coordinate, detector_coordinates = self._rays_of_view(idx)
        proj_i = _float32(proj_i.flatten(), pytomography.device)
        for detector_coordinates_s, beam_coordinate_s, proj_s in zip(torch.tensor_split(detector_coordinates, self.N_splits), torch.tensor_split(beam_coordinate, self.N_splits), torch.tensor_split(proj_i, self.N_splits)):
            parallelproj_core.joseph3d_back(beam_coordinate_s, detector_coordinates_s, BP, self.origin, self.voxel_size, proj_s)

    def _fbp(self, projections: torch.Tensor, filter=None) -> torch.Tensor:
        """Filtered back projection of a circular scan (FDK) onto the object grid of this system matrix, called by
        :class:`pytomography.algorithms.FilteredBackProjection`. Each projection is cosine weighted and ramp filtered
        along the detector rows (times the window ``filter``), back projected, and weighted with the FDK distance weight.

        Args:
            projections (torch.Tensor): Line integrals (views, u, v), on any device: one view at a time is moved to
                the projection device.
            filter (FBPFilter, optional): Window applied on top of the ramp filter. Defaults to None (Ram-Lak).

        Returns:
            torch.Tensor: Attenuation on the object grid, on ``pytomography.device``.

        Raises:
            ValueError: If the scan is not a full turn of evenly spaced views at one z. Every view is weighted by
                2 pi / N and there are no short-scan (Parker) weights, so a shorter arc, or one with views missing,
                would be reconstructed with the wrong weights.
        """
        self._check_fdk_scan()
        window = get_fbp_filter(filter)
        BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        for idx in range(self.proj_meta.N_angles):
            proj_i = projections[idx].to(self.device) * self._get_FBP_preweight(idx)   # one view at a time on the device
            proj_i = ramp_filter(proj_i, self.proj_meta.dr[0], window, dim=0)
            # each view's back projection is weighted on its own before it is accumulated (in place: one image more)
            BP_i = torch.zeros_like(BP)
            self._back_project_view(BP_i, idx, proj_i)
            weighted = self._get_FBP_postweight(idx)
            weighted.mul_(BP_i[1:-1, 1:-1, 1:-1]).mul_(self._get_FBP_scale())
            BP[1:-1, 1:-1, 1:-1] += weighted
            del BP_i, weighted
        return _crop(BP)

    def _check_fdk_scan(self) -> None:
        """Raises unless the views are evenly spaced over a full turn (every gap between neighbouring angles within 10 %
        of 360 / N degrees: measured stage angles jitter, by up to 0.8 % of the step for the SophiaBeads scan) at one z."""
        n = self.proj_meta.N_angles
        angles = np.sort(np.mod(self.proj_meta.angles.detach().cpu().double().numpy(), 2 * np.pi))
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        step = 2 * np.pi / n
        if n < 2 or np.abs(gaps - step).max() > 0.1 * step:
            raise ValueError(f'FDK needs views evenly spread over a full turn ({n} views {np.degrees(step):.3g} degrees '
                             f'apart); the largest gap between these is {np.degrees(gaps.max()):.3g} degrees. A short '
                             'scan needs short-scan (Parker) weights, which this FDK does not have: reconstruct it '
                             'iteratively (e.g. SART)')
        z = self.proj_meta.z_locations.detach().cpu()
        if (z.max() - z.min()).item() > 1e-3 * self.object_meta.dz:
            raise ValueError('FDK needs a circular scan, at one z; this one moves from z = '
                             f'{z.min().item():.4g} to {z.max().item():.4g} mm')

    def _fbp_memory_parts(self, projections: torch.Tensor, filter=None) -> list:
        """The memory :meth:`_fbp` takes, without running it, for
        :meth:`pytomography.algorithms.FilteredBackProjection.estimate_memory`. Held: the projections, the padded
        image, and the in-plane distance weight of every view, which the system matrix keeps for later calls. For each
        view: its back projection (a second padded image), its rays, its ramp filter's spectra, and its weights in 3D
        (three images, measured); on the first view, while the in-plane weights of every view are computed, two more
        arrays of their size."""
        from pytomography.utils.memory import MemoryPart, nbytes
        image = int(np.prod(self.object_meta.shape)) * 4
        padded = int(np.prod([s + 2 for s in self.object_meta.shape])) * 4
        Nx, Ny, _ = self.object_meta.shape
        cols, rows = self.proj_meta.shape
        rays = cols * rows
        spectra = 16 * int(2 ** np.ceil(np.log2(2 * cols))) * rows               # ramp_filter's, zero padded
        weights = self.proj_meta.N_angles * Nx * Ny * 4
        on = (lambda b: dict(gpu_bytes=b)) if torch.device(pytomography.device).type == 'cuda' else (lambda b: dict(ram_bytes=b))
        ram, gpu = nbytes(projections)
        return [MemoryPart('projections', ram_bytes=ram, gpu_bytes=gpu, scope='held'),
                MemoryPart('image', **on(padded + image), scope='held'),
                MemoryPart('distance weights of every view', **on(weights), scope='held'),
                MemoryPart('one view: back projection, rays, distance weight',
                           **on(max(padded + 28 * rays + spectra + 3 * image, 2 * weights)), scope='chunk')]

    def _memory_parts(self, n_subsets: int = 1, N_splits: int | None = None) -> list:
        r"""The arrays of an ordered-subset reconstruction with this system matrix, as OS-SART takes them, for
        :meth:`estimate_memory` (which adds the projections, passed as ``held``). Held for the whole reconstruction:
        the image, and one :math:`H_m^T 1` per subset, kept on the device (an image each), so that more subsets take
        more memory. For each subset: its projections (measured, predicted, of ones, and their ratio) and the images of
        the update. For each view: the end points and values of its rays (28 bytes a ray) and the padded image.
        ``N_splits`` splits the kernel calls of a view but not these arrays, so it does not change the memory."""
        from pytomography.utils.memory import MemoryPart
        n = max(1, n_subsets)
        image = int(np.prod(self.object_meta.shape)) * 4
        padded = int(np.prod([s + 2 for s in self.object_meta.shape])) * 4
        rays = int(np.prod(self.proj_meta.shape))
        views = -(-self.proj_meta.N_angles // n)                    # in the largest subset
        on = lambda device, b: dict(gpu_bytes=b) if torch.device(device).type == 'cuda' else dict(ram_bytes=b)
        return [MemoryPart('image', **on(pytomography.device, image), scope='held'),
                MemoryPart(f'H_m^T 1 of {n} subset{"s" if n > 1 else ""}', **on(pytomography.device, n * image), scope='held'),
                MemoryPart('projections of a subset: measured, predicted, of ones, ratio',
                           **on(self.device, 4 * views * rays * 4), scope='subset'),
                MemoryPart('images of an update', **on(pytomography.device, 3 * image), scope='subset'),
                MemoryPart('one view: rays and the padded image', **on(pytomography.device, 28 * rays + padded + image), scope='chunk')]

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list:
        """Half and twice the subsets, where they take less memory: each subset keeps an image on the device while
        its projections shrink with more subsets, so which way saves memory depends on the scan. ``N_splits`` does
        not change the memory."""
        candidates = [(f'{n_subsets // 2} subsets', n_subsets // 2, N_splits)] if n_subsets >= 2 else []
        return _saving_memory(self, candidates + [(f'{2 * n_subsets} subsets', 2 * n_subsets, N_splits)], n_subsets, N_splits)

    def backward(
        self,
        proj: torch.Tensor,
        subset_idx: int | None = None,
    ) -> torch.Tensor:
        r"""Back project projections: :math:`H_m^T g`.

        This is the adjoint of :meth:`forward`; for filtered back projection (FDK), use
        :class:`~pytomography.algorithms.FilteredBackProjection`.

        Args:
            proj (torch.Tensor): Projections of the subset's views, [views, u, v], on any device.
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: every view.

        Returns:
            torch.Tensor: Back projection on the object grid, [Lx, Ly, Lz], float32, on ``pytomography.device``.
        """
        # parallelproj adds into the image it is given, so every angle accumulates into one (padded) buffer
        BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        angle_indices = self._angle_indices(subset_idx)
        for i in range(len(angle_indices)):
            self._back_project_view(BP, angle_indices[i], proj[i])
        return _crop(BP)