"""The system matrix of parallel-hole SPECT cameras, which projects by rotating the object and summing it.

:class:`SPECTSystemMatrix` takes the ``object_meta`` and ``proj_meta`` of :func:`pytomography.io.SPECT.read_dicom` or
:func:`~pytomography.io.SPECT.read_simind`, and transforms such as
:class:`~pytomography.transforms.SPECT.SPECTAttenuationTransform` and
:class:`~pytomography.transforms.SPECT.SPECTPSFTransform`. :class:`MonteCarloHybridSPECTSystemMatrix` forward projects
with SIMIND instead.
"""
from __future__ import annotations
import math
import warnings
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.transforms.shared import RotationTransform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.utils import pad_object, unpad_object, pad_proj, unpad_proj, rotate_detector_z
from pytomography.utils.fourier_filters import get_fbp_filter, ramp_filter
from pytomography.utils.memory import MemoryPart, nbytes, tensor_bytes
import numpy as np
from ..system_matrix import SystemMatrix, on_device
from pytomography.utils import simind_mc
from copy import copy
from typing import Sequence
import shutil

def _transform_memory(transform, object_shape: Sequence[int], n_views: int) -> tuple[float, float, float]:
    """What an object-to-object transform of a SPECT system matrix keeps, and the most one call holds at once on objects
    of ``object_shape``, its output included, in bytes: (RAM kept, GPU kept, call's peak on ``pytomography.device``).
    Measured on a GPU for 128^3 and 182^3 objects (padded), where noted.

    * Attenuation: its map, and the rotated probability map of every view if it caches them (``cache_probabilities``).
      A call without the cache pads and rotates the map, scales it, sums it along the projection axis (flipped, summed,
      flipped back, less half of itself) and takes the exponential, then multiplies: six objects (measured); with the
      cache, the product only.
    * PSF with Gaussian layers: their kernels (one per radius); a call takes three objects (measured: the copy a
      convolution is given, its output and the next's).
    * PSF from an operator (:class:`~pytomography.transforms.SPECT.psf.PSF2D`): its grids, and the kernel spectrum of
      every radius if it caches them (``cache_kernel``); a call takes eight arrays of the zero padded object (signal, its
      spectrum, the kernel's, their product and the result, rounded up for the first call, which also runs the
      operator's own convolution to check the fast one). Not measured.
    * Any other transform: the tensors it keeps, and two objects per call.
    """
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    from pytomography.transforms.SPECT.psf import Seperable1DBlurNet, PSF2D
    obj = 4 * math.prod(object_shape)
    on_gpu = torch.device(pytomography.device).type == 'cuda'
    if isinstance(transform, SPECTAttenuationTransform):
        ram, gpu = nbytes(getattr(transform, 'attenuation_map', None))
        if transform.cache_probabilities:
            cache = n_views * obj
            ram, gpu = (ram, gpu + cache) if on_gpu else (ram + cache, gpu)
            return ram, gpu, obj
        return ram, gpu, 6 * obj
    if isinstance(transform, SPECTPSFTransform):
        ram = gpu = temporaries = 0.0
        for layer in getattr(transform, 'layers', {}).values():
            if isinstance(layer, Seperable1DBlurNet):
                for conv in (layer.layer_r, layer.layer_z):
                    if conv is not None:
                        r, g = nbytes(conv.weight)
                        ram, gpu = ram + r, gpu + g
                temporaries = max(temporaries, 3 * obj)
            elif isinstance(layer, PSF2D):
                r, g = tensor_bytes(layer)
                k = int(layer.kernel_size)
                height = object_shape[1] + k - 1
                width = object_shape[2] + k - 1
                width += width % 2
                if layer.cache_kernel and getattr(layer, '_kernel_spectrum', None) is None:   # cached at the first call
                    spectrum = 8.0 * object_shape[0] * height * (width // 2 + 1)
                    r, g = (r, g + spectrum) if on_gpu else (r + spectrum, g)
                ram, gpu = ram + r, gpu + g
                temporaries = max(temporaries, 8 * 4.0 * object_shape[0] * height * width)
            else:
                temporaries = max(temporaries, 4 * obj)
        return ram, gpu, temporaries
    ram, gpu = tensor_bytes(transform)
    return ram, gpu, 2 * obj

def _view_weights(angles: Sequence[float]) -> list[float]:
    """Angle (radians) that each view stands for in the sum of filtered back projection. A parallel-hole view and the
    view opposite it measure the same lines, so the angles are taken modulo 180 degrees, where each view gets half the
    gaps to its neighbours: N equally spaced views over 360 or over 180 degrees each get pi/N, and lines measured twice
    share their weight. A gap of more than twice the typical one means the views cover less than 180 degrees, which
    filtered back projection cannot reconstruct exactly: this warns, and the views beside the gap get the typical gap.
    """
    a = np.mod(np.asarray(angles, dtype=np.float64), 180.0)
    order = np.argsort(a, kind='stable')
    gaps = np.diff(np.append(a[order], a[order][0] + 180.0))  # from each view (in angle order) to the next
    typical = np.median(gaps[gaps > 1e-6]) if np.any(gaps > 1e-6) else 180.0
    if np.any(gaps > 2 * typical + 1e-6):
        warnings.warn('the views cover less than 180 degrees: filtered back projection cannot reconstruct the object '
                      'exactly', stacklevel=3)
        gaps = np.minimum(gaps, typical)
    weights = np.empty_like(a)
    weights[order] = 0.5 * (gaps + np.roll(gaps, 1))
    return np.deg2rad(weights).tolist()

class SPECTSystemMatrix(SystemMatrix):
    r"""System matrix of a parallel-hole SPECT camera, which projects by rotating the object and summing it.

    For each view, the (padded) object is rotated so that the detector is along +x, the object-to-object transforms
    are applied to the rotated object (with the view's index, e.g. attenuation towards the detector and the
    distance-dependent PSF), and the object is summed along x. The projection-to-projection transforms are then applied
    to the projections of all the views (e.g. an additive term or a cut-off). Back projection does the transposes in
    the reverse order. Projections are [θ, r, z] (one energy window), objects [Lx, Ly, Lz]; with subsets, the views are
    split into interleaved groups (view ``i`` in subset ``i % n_subsets``).

    Args:
        obj2obj_transforms (list[Transform]): Object-to-object transforms, applied to the rotated object of each view.
        proj2proj_transforms (list[Transform]): Projection-to-projection transforms, applied after the sum.
        object_meta (SPECTObjectMeta): The object grid.
        proj_meta (SPECTProjMeta): The projection space.
        object_initial_based_on_camera_path (bool, optional): Make the initial object of a reconstruction zero outside
            the camera's path (voxels further from the axis than the detector at any view). Defaults to False.

    Example:
        .. code-block:: python

            projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM, energy_window=0)
            mu = io.attenuation_map(files_CT, object_meta, proj_meta.energy_windows[0].photon_keV)
            psf_meta = io.SPECT.collimator_psf('SY-ME', proj_meta.energy_windows[0].photon_keV)
            system_matrix = SPECTSystemMatrix(
                obj2obj_transforms=[SPECTAttenuationTransform(mu), SPECTPSFTransform(psf_meta)],
                proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        proj2proj_transforms: list[Transform],
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        object_initial_based_on_camera_path: bool = False
    ) -> None:
        super(SPECTSystemMatrix, self).__init__(object_meta, proj_meta, obj2obj_transforms, proj2proj_transforms)
        self.object_initial_based_on_camera_path = object_initial_based_on_camera_path
        self.rotation_transform = RotationTransform()
        
    def _get_object_initial(self, device=None):
        """Returns an initial object estimate used in reconstruction algorithms. By default, this is a tensor of ones with the same shape as the object metadata.

        Returns:
            torch.Tensor: Initial object used in reconstruction algorithm.
        """
        object_initial = torch.ones(self.object_meta.shape).to(device)
        if self.object_initial_based_on_camera_path:
            for i in range(len(self.proj_meta.angles)):
                cutoff_idx = int(np.ceil(self.object_meta.shape[0]/ 2 - self.proj_meta.radii[i]/self.object_meta.dr[0])) #+ 5 # +5 assumes 5 additional voxel between camera and object
                if cutoff_idx<0:
                    continue
                img_cutoff = torch.ones(self.object_meta.shape).to(device)
                img_cutoff[:cutoff_idx, :, :] = 0
                img_cutoff = pad_object(img_cutoff)
                img_cutoff = self.rotation_transform.backward(img_cutoff, 270-self.proj_meta.angles[i])
                img_cutoff = unpad_object(img_cutoff)
                object_initial *= img_cutoff
        return object_initial
    
    def set_n_subsets(
        self,
        n_subsets: int
    ) -> None:
        """Partitions the views into ``n_subsets`` interleaved subsets: view ``i`` goes in subset ``i % n_subsets``.

        The partition is kept in ``subset_indices_array`` (one index tensor per subset).

        Args:
            n_subsets (int): Number of subsets, from 1 to the number of views.

        Raises:
            ValueError: If ``n_subsets`` is less than 1 or more than the number of views.
        """
        self._check_n_subsets(n_subsets, self.proj_meta.shape[0], 'projections')
        indices = torch.arange(self.proj_meta.shape[0]).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(
        self,
        projections: torch.tensor,
        subset_idx: int
    ) -> torch.tensor: 
        """The projections :math:`g_m` of subset :math:`m`: its views.

        Args:
            projections (torch.Tensor): All the projections :math:`g`, [..., θ, r, z].
            subset_idx (int): Subset index :math:`m`.

        Returns:
            torch.Tensor: The subset's projections :math:`g_m`, [..., θ_m, r, z].
        """
        return projections[...,self.subset_indices_array[subset_idx],:,:]
    
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""The relative weight of subset :math:`m`: its share of the views.

        Reconstruction algorithms scale parameters that are relative to :math:`H_m^T 1` by it, such as the prior
        weight :math:`\beta`.

        Args:
            subset_idx (int | None): Subset index :math:`m`; None for all the views.

        Returns:
            float: The number of views of the subset over the number of views (1 for None).
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.num_projections

    def compute_normalization_factor(self, subset_idx : int | None = None) -> torch.tensor:
        """The normalization :math:`H_m^T 1`: the back projection of ones over the views of subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the views.

        Returns:
            torch.Tensor: :math:`H_m^T 1`, [Lx, Ly, Lz].
        """
        norm_proj = torch.ones(self.proj_meta.shape).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = self.get_projection_subset(norm_proj, subset_idx)
        # an image of its own: back projection returns a view of the padded object it was computed in (about twice the
        # image for a 128^3 object), which a reconstruction would otherwise keep for each of its subsets
        return self.backward(norm_proj, subset_idx).contiguous()

    def _memory_parts(self, n_subsets: int, N_splits: int | None = None) -> list[MemoryPart]:
        r"""What a reconstruction with this system matrix holds at its peak (see
        :meth:`~pytomography.projectors.SystemMatrix.estimate_memory`), all on ``pytomography.device``. Held for the
        whole run: the image, one :math:`H_m^T 1` image per subset (so that more subsets can take more memory), the
        images of an update, the rotation grids of every view (cached by the rotation: two per view, one more per
        attenuation transform and for an initial object from the camera path), and what the transforms keep (see
        :func:`_transform_memory`). For a subset: its projections (measured, expected, additive term, of ones, and the
        padded copy back projection makes; at the start, the projections of ones of every view for :math:`H_m^T 1`),
        one more padded copy per projection transform, and the padded object, support and sum of a projection. For a
        view: the rotated object and the most one of the transforms holds at once (they run one after another, each
        freeing its temporaries: rotation, attenuation and a Gaussian PSF together took seven padded objects, measured).
        ``N_splits`` does not apply.

        Args:
            n_subsets (int): Number of subsets.
            N_splits (int | None): Not used (no projector splits).

        Returns:
            list[MemoryPart]: The arrays.
        """
        from pytomography.transforms.SPECT import SPECTAttenuationTransform
        n = max(1, int(n_subsets))
        device = pytomography.device
        image = 4 * math.prod(self.object_meta.shape)
        padded_shape = tuple(self.object_meta.padded_shape)
        padded = 4 * math.prod(padded_shape)
        n_views = self.proj_meta.num_projections
        views = -(-n_views // n)                                      # in the largest subset
        view_padded = 4 * math.prod(self.proj_meta.padded_shape[1:])
        view = 4 * math.prod(self.proj_meta.shape[1:])
        attenuations = sum(isinstance(t, SPECTAttenuationTransform) for t in self.obj2obj_transforms)
        grids_per_view = 2 + attenuations + (1 if self.object_initial_based_on_camera_path else 0)
        parts = [MemoryPart('image', **on_device(device, image)),
                 MemoryPart(f'H_m^T 1 of {n} subset{"s" if n > 1 else ""}', **on_device(device, n * image)),
                 MemoryPart('images of an update', **on_device(device, 3 * image)),
                 MemoryPart('rotation grids of every view (cached)', **on_device(device, grids_per_view * n_views * padded_shape[0] * padded_shape[1] * 2 * 4))]
        ram = gpu = temporaries = 0.0
        for transform in self.obj2obj_transforms:
            r, g, t = _transform_memory(transform, padded_shape, n_views)
            ram, gpu, temporaries = ram + r, gpu + g, max(temporaries, t)
        if self.obj2obj_transforms:
            parts.append(MemoryPart('what the object transforms keep', ram, gpu))
        if self.proj2proj_transforms:
            kept = [tensor_bytes(t) for t in self.proj2proj_transforms]
            parts.append(MemoryPart('what the projection transforms keep', sum(r for r, _ in kept), sum(g for _, g in kept)))
        projections = max(5 * views * view_padded, n_views * view + 2 * views * view_padded) + len(self.proj2proj_transforms) * views * view_padded
        parts.append(MemoryPart('projections of a subset: measured, expected, additive term, of ones, padded copy', **on_device(device, projections), scope='subset'))
        parts.append(MemoryPart('padded object, support and sum of a projection', **on_device(device, 3 * padded), scope='subset'))
        parts.append(MemoryPart("one view: rotated object and the transforms' temporaries", **on_device(device, padded + max(temporaries, padded)), scope='chunk'))
        return parts

    def _memory_alternatives(self, n_subsets: int, N_splits: int | None) -> list:
        """Half and twice the subsets (at most one view each), where they take less memory: each subset keeps an
        :math:`H_m^T 1` image while its projections shrink with more subsets, so which way saves memory depends on the
        scan."""
        candidates = [(f'{n_subsets // 2} subsets', n_subsets // 2, None)] if n_subsets >= 2 else []
        if 2 * n_subsets <= self.proj_meta.num_projections:
            candidates.append((f'{2 * n_subsets} subsets', 2 * n_subsets, None))
        return self._saving_memory(candidates, n_subsets, N_splits)

    def _host_phis(self) -> list[float]:
        """Python floats of ``270 - proj_meta.angles`` (the rotation applied to the object for each projection), copied from the device once per ``angles`` tensor. The projection loops pass these, and plain ``int`` angle indices, to the rotation and the transforms, so no per-angle device-to-host synchronisation occurs: indexing a device tensor with a 0-d device tensor, or converting a 0-d device tensor to a Python scalar, stalls the CPU until the GPU queue has drained (about 0.25 ms each on Windows), which was most of the projection time. The subtraction is still done in the tensor's dtype, so the values are unchanged."""
        angles = self.proj_meta.angles
        if getattr(self, '_host_phis_src', None) is not angles:
            self._host_phis_cache = (270 - angles).tolist()
            self._host_phis_src = angles
        return self._host_phis_cache

    def _host_angle_indices(self, subset_idx: int | None) -> list[int]:
        """Angle indices of subset ``subset_idx`` (all angles if None) as a list of Python ints."""
        if subset_idx is None:
            return list(range(self.proj_meta.num_projections))
        angle_subset = self.subset_indices_array[subset_idx]
        return angle_subset.tolist() if torch.is_tensor(angle_subset) else list(angle_subset)

    def forward(
        self,
        object: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Forward projection :math:`g_m = H_m f` of an object.

        Args:
            object (torch.Tensor): The object, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`: project onto the subset's views only. Defaults
                to None: all the views.

        Returns:
            torch.Tensor: The projections, [θ, r, z] (the subset's views with ``subset_idx``), on
            ``pytomography.device``.
        """
        # Deal with subset stuff (Python ints/floats: the loop below must not synchronise with the device)
        angle_indices = self._host_angle_indices(subset_idx)
        phis = self._host_phis()
        N_angles = len(angle_indices)
        # Start projection
        object = object.to(pytomography.device)
        proj = torch.zeros((N_angles,*self.proj_meta.padded_shape[1:]), device=pytomography.device)
        # Format Object (once: the rotation returns a new tensor, so the padded object is never modified)
        object_padded = pad_object(object)
        # Loop through all angles
        for i, angle_idx in enumerate(angle_indices):
            # beta = 270 - phi, and backward transform called because projection should be at +beta (requires inverse rotation of object)
            object_i = self.rotation_transform.backward(object_padded, phis[angle_idx])
            # Apply object 2 object transforms
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_idx)
            proj[i] = object_i.sum(axis=0)
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Back projection :math:`H_m^T g_m` of projections.

        Args:
            proj (torch.Tensor): The projections, [θ, r, z] (only the subset's views with ``subset_idx``).
            subset_idx (int | None, optional): Subset index :math:`m` of the projections. Defaults to None: all the
                views.

        Returns:
            torch.Tensor: The object, [Lx, Ly, Lz], on ``pytomography.device``.
        """
        # Deal with subset stuff (Python ints/floats: the loop below must not synchronise with the device)
        angle_indices = self._host_angle_indices(subset_idx)
        phis = self._host_phis()
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones(self.object_meta.shape, device=pytomography.device), mode='back_project')
        # Pad proj and norm_proj (norm_proj used to compute sum_j H_ij)
        proj = pad_proj(proj)
        # First apply proj transforms before back projecting
        for transform in self.proj2proj_transforms[::-1]:
            proj = transform.backward(proj)
        # Setup for back projection
        object = torch.zeros(self.object_meta.padded_shape, device=pytomography.device)
        for i, angle_idx in enumerate(angle_indices):
            # Perform back projection
            object_i = proj[i].unsqueeze(0) * boundary_box_bp
            # Apply object mappings
            for transform in self.obj2obj_transforms[::-1]:
                object_i  = transform.backward(object_i, angle_idx)
            # Rotate all objects by by their respective angle
            object_i = self.rotation_transform.forward(object_i, phis[angle_idx])
            # Add to total
            object += object_i
        # Unpad
        object = unpad_object(object)
        return object

    def _fbp(self, projections: torch.Tensor, filter=None) -> torch.Tensor:
        r"""Filtered back projection of parallel-hole projections onto the object grid of this system matrix, called by
        :class:`pytomography.algorithms.FilteredBackProjection`. Each projection is ramp filtered along :math:`r` (times
        the window ``filter``, which is given the frequency in cycles per mm) and back projected as in :meth:`backward`,
        weighted by the angle it stands for: :math:`\pi/N` for :math:`N` equally spaced views over 360 or over 180
        degrees. Only the geometry is used, not the transforms of the system matrix, so the object is not corrected for
        attenuation, collimator blurring or scatter.

        Args:
            projections (torch.Tensor[N, Lr, Lz]): Projections.
            filter (FBPFilter, optional): Window applied on top of the ramp filter. Defaults to None (Ram-Lak).

        Returns:
            torch.Tensor[Lx, Ly, Lz]: The object, in the units of the object in :meth:`forward`.
        """
        if tuple(projections.shape) != tuple(self.proj_meta.shape):
            raise ValueError(f'projections of shape {tuple(projections.shape)} do not match the system matrix, '
                             f'{tuple(self.proj_meta.shape)}')
        window = get_fbp_filter(filter)
        # The projector samples r on the object grid (y, once rotated) and adds up the voxels along x: the ramp filter,
        # per mm of r, times the voxel length along x gives the object per voxel. Lengths in mm, the unit of the windows.
        dx, dy = self.object_meta.dr[0], self.object_meta.dr[1]
        weights = _view_weights(self.proj_meta.angles.tolist())
        phis = self._host_phis()
        boundary_box_bp = pad_object(torch.ones(self.object_meta.shape, device=pytomography.device), mode='back_project')
        object = torch.zeros(self.object_meta.padded_shape, device=pytomography.device)
        for angle_idx in range(self.proj_meta.num_projections):
            # filtered over the padded width, so that voxels off the axis get the tails of the filter
            proj_filtered = ramp_filter(pad_proj(projections[angle_idx].to(pytomography.device)), dy, window, dim=0)
            object_i = proj_filtered.unsqueeze(0) * boundary_box_bp
            object += self.rotation_transform.forward(object_i, phis[angle_idx]) * (weights[angle_idx] * dx)
        return unpad_object(object)

    def _fbp_memory_parts(self, projections: torch.Tensor, filter=None) -> list:
        """The memory :meth:`_fbp` takes, without running it, for
        :meth:`pytomography.algorithms.FilteredBackProjection.estimate_memory`. Held: the projections, the padded
        object and its support (two images), and the rotation grid of every view, which is cached. For each view: the
        filtered projection spread through the object, its rotated, weighted copy and the filter's FFT workspace (six
        images, measured)."""
        from pytomography.utils.memory import MemoryPart, nbytes
        padded = int(np.prod(self.object_meta.padded_shape)) * 4
        Lx, Ly = self.object_meta.padded_shape[:2]
        on = (lambda b: dict(gpu_bytes=b)) if torch.device(pytomography.device).type == 'cuda' else (lambda b: dict(ram_bytes=b))
        ram, gpu = nbytes(projections)
        return [MemoryPart('projections', ram_bytes=ram, gpu_bytes=gpu, scope='held'),
                MemoryPart('image and its support', **on(2 * padded), scope='held'),
                MemoryPart('rotation grids of every view', **on(self.proj_meta.num_projections * Lx * Ly * 2 * 4), scope='held'),
                MemoryPart('one view: spread and rotated', **on(6 * padded), scope='chunk')]

class MonteCarloHybridSPECTSystemMatrix(SPECTSystemMatrix):
    """A SPECT system matrix whose forward projection is a SIMIND Monte Carlo simulation of the object.

    The forward projection simulates the object (as the source) in the 140 keV attenuation map with SIMIND, for every
    isotope, and returns the total projections of the photopeak window, scaled by the object's sum and the isotope's
    proportion; the projection-to-projection transforms are then applied. The back projection is
    :class:`SPECTSystemMatrix`'s, with the transforms given. SIMIND must be on the PATH. Lengths are in mm, like every length in
    PyTomography: the detector thicknesses are converted to SIMIND's cm only where SIMIND is called
    (:func:`pytomography.utils.simind_mc.get_simind_isotope_detector_params`); their names before PyTomography 4, in
    cm (``crystal_thickness``...), raise a TypeError that gives the new value.

    Args:
        object_meta (SPECTObjectMeta): The object grid.
        proj_meta (SPECTProjMeta): The projection space.
        n_events (int): Number of photons to simulate per view.
        n_parallel (int): Number of SIMIND processes run at once, one per CPU core. Leave a few cores for the rest of
            the computer, e.g. ``max(1, os.cpu_count() - 4)``.
        obj2obj_transforms (Sequence[Transform]): Object-to-object transforms for back projection.
        proj2proj_transforms (Sequence[Transform]): Projection-to-projection transforms, applied after the simulation
            (e.g. an additive term or a cut-off) and before back projection.
        attenuation_map_140keV (torch.Tensor): Attenuation map at 140 keV, in 1/mm, [Lx, Ly, Lz], for the simulation.
        energy_window_params (Sequence[str]): The lines of a SIMIND ``scattwin.win`` file: the energy windows to
            simulate.
        primary_window_idx (int): Index (in ``energy_window_params``) of the photopeak window that is reconstructed.
        isotope_names (Sequence[str]): The isotopes to simulate (SIMIND's names).
        isotope_ratios (Sequence[float]): The proportion of each isotope.
        collimator_type (str): The collimator (SIMIND's name).
        crystal_thickness_mm (float): Crystal thickness in mm (NaI).
        cover_thickness_mm (float): Cover thickness in mm (aluminium).
        backscatter_thickness_mm (float): Equivalent backscatter thickness in mm (pyrex).
        energy_resolution_140keV (float, optional): Energy resolution of the detector at 140 keV, in percent, scaled
            as the square root of the energy. Defaults to 0.
        advanced_energy_resolution_model (str | None, optional): An energy resolution model used instead of
            ``energy_resolution_140keV``; only ``'siemens'``. Defaults to None.
        advanced_collimator_modeling (bool, optional): Model septal penetration and scatter in the collimator. Defaults
            to False.

    Raises:
        TypeError: If a thickness is missing, or given in cm under its name from before PyTomography 4.
        RuntimeError: If SIMIND is not on the PATH.
    """
    def __init__(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        n_events: int,
        n_parallel: int,
        obj2obj_transforms: Sequence[Transform],
        proj2proj_transforms: Sequence[Transform],
        attenuation_map_140keV: torch.Tensor,
        energy_window_params: Sequence[str],
        primary_window_idx: int,
        isotope_names: Sequence[str],
        isotope_ratios: Sequence[float],
        collimator_type: str,
        crystal_thickness_mm: float | None = None,
        cover_thickness_mm: float | None = None,
        backscatter_thickness_mm: float | None = None,
        energy_resolution_140keV: float = 0,
        advanced_energy_resolution_model: str | None = None,
        advanced_collimator_modeling: bool = False,
        **removed,
    ):
        simind_mc.refuse_thicknesses_in_cm(removed)
        missing = [n for n, v in (('crystal_thickness_mm', crystal_thickness_mm), ('cover_thickness_mm', cover_thickness_mm),
                                  ('backscatter_thickness_mm', backscatter_thickness_mm)) if v is None]
        if missing:
            raise TypeError(f'missing {", ".join(missing)}')
        # check if "simind" is PATH variable and if not raise error
        if not shutil.which("simind"):
            raise RuntimeError(
                "SIMIND is not in PATH. You need to install SIMIND to use this projector."
            )
        super().__init__(
            obj2obj_transforms,
            proj2proj_transforms,
            object_meta,
            proj_meta,
            object_initial_based_on_camera_path=True,
        )
        self.attenuation_map_140keV = attenuation_map_140keV
        self.energy_window_params = energy_window_params
        self.isotope_names = isotope_names
        self.isotope_ratios = isotope_ratios
        self.collimator_type = collimator_type
        self.cover_thickness_mm = cover_thickness_mm
        self.crystal_thickness_mm = crystal_thickness_mm
        self.backscatter_thickness_mm = backscatter_thickness_mm
        self.energy_resolution_140keV = energy_resolution_140keV
        self.advanced_energy_resolution_model = advanced_energy_resolution_model
        self.advanced_collimator_modeling = advanced_collimator_modeling
        self.primary_window_idx = primary_window_idx
        self.n_events = n_events
        self.n_parallel = n_parallel

    SIMIND_PROCESS_BYTES = 0.05e9
    """Memory of one SIMIND process besides the maps and projections it keeps, in bytes, for memory estimates.

    Measured with ``tests/calibrate_memory.py`` (case ``simind``: 128^3 maps, 60 views, one window): about 10 MB a
    process, maps and images included; rounded up."""

    def _memory_parts(self, n_subsets: int, N_splits: int | None = None) -> list[MemoryPart]:
        """What a reconstruction with this system matrix holds at its peak: as :class:`SPECTSystemMatrix` (whose back
        projection it uses), and the 140 keV attenuation map; for a subset, the host copies of the maps written for
        SIMIND and the scatter and total projections of every energy window that are read back and added up (four sets);
        and the ``n_parallel`` SIMIND processes, each with its maps and the images of every window (three per window),
        plus :data:`SIMIND_PROCESS_BYTES`. The SIMIND processes are processes of their own: they take the machine's
        RAM, not this process's.

        Args:
            n_subsets (int): Number of subsets.
            N_splits (int | None): Not used.

        Returns:
            list[MemoryPart]: The arrays.
        """
        parts = super()._memory_parts(n_subsets, N_splits)
        n = max(1, int(n_subsets))
        views = -(-self.proj_meta.num_projections // n)
        view = 4 * math.prod(self.proj_meta.shape[1:])
        image = 4 * math.prod(self.object_meta.shape)
        windows = len(self.energy_window_params)
        parts.append(MemoryPart('attenuation map at 140 keV', *nbytes(self.attenuation_map_140keV)))
        parts.append(MemoryPart('SIMIND maps and the projections of every window read back (host)',
                                ram_bytes=2 * image + 4 * windows * views * view, scope='subset'))
        parts.append(MemoryPart(f'SIMIND processes ({self.n_parallel})',
                                ram_bytes=self.n_parallel * (self.SIMIND_PROCESS_BYTES + 2 * image + 3 * windows * views * view), scope='chunk'))
        return parts

    def _get_proj_meta_subset(self, subset_idx: int) -> SPECTProjMeta:
        """Creates a new SPECTProjMeta that corresponds to a subset of projections
        Args:
            subset_idx (int): Index of the subset to use
        Returns:
            SPECTProjMeta: New SPECTProjMeta that corresponds to the subset of projections
        """
        indices_array = self.subset_indices_array[subset_idx]
        proj_meta_new = copy(self.proj_meta)
        proj_meta_new.angles = proj_meta_new.angles[indices_array]
        proj_meta_new.radii = proj_meta_new.radii[indices_array.cpu().numpy()]
        proj_meta_new.shape = (len(indices_array), proj_meta_new.shape[1], proj_meta_new.shape[2])
        proj_meta_new.padded_shape = (len(indices_array), proj_meta_new.padded_shape[1], proj_meta_new.padded_shape[2])
        proj_meta_new.num_projections = len(indices_array)
        return proj_meta_new
        
    def forward(self, object: torch.Tensor, subset_idx: int | None = None):
        """Forward projection by a SIMIND simulation of the object: the photopeak window's total projections.

        Args:
            object (torch.Tensor): The object (the source), [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index: simulate the subset's views only. Defaults to None: all
                the views.

        Returns:
            torch.Tensor: The simulated projections, [θ, r, z], scaled by the object's sum and the isotopes'
            proportions.
        """
        # subsample proj_meta
        if subset_idx is not None:
            proj_meta = self._get_proj_meta_subset(subset_idx)
        else:
            proj_meta = self.proj_meta
        projections_total = 0
        for isotope_name, isotope_ratio in zip(self.isotope_names, self.isotope_ratios):
            index_dict = simind_mc.get_simind_params_from_metadata(self.object_meta, proj_meta)
            index_dict.update(simind_mc.get_simind_isotope_detector_params(
                isotope_name = isotope_name,
                collimator_type= self.collimator_type,
                crystal_thickness_mm=self.crystal_thickness_mm,
                cover_thickness_mm=self.cover_thickness_mm,
                backscatter_thickness_mm=self.backscatter_thickness_mm,
                energy_resolution_140keV=self.energy_resolution_140keV,
                advanced_energy_resolution_model=self.advanced_energy_resolution_model,
                advanced_collimator_modeling=self.advanced_collimator_modeling
            ))
            projections = simind_mc.run_scatter_simulation(
                object,
                self.attenuation_map_140keV,
                self.object_meta,
                proj_meta,
                self.energy_window_params,
                index_dict,
                self.n_events,
                self.n_parallel,
                return_total=True,
            )[self.primary_window_idx]
            projections_total += projections * object.sum() * isotope_ratio
        # still apply proj2proj transforms since these are only additive term and cutoff
        for transform in self.proj2proj_transforms:
            projections_total = transform.forward(projections_total, padded=False)
        return projections_total
        