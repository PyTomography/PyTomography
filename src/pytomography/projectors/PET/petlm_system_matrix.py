from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors import SystemMatrix
import math
import numpy as np
import parallelproj_core
from pytomography.io.PET.shared import crystal_pair_blocks, crystal_pair_index
from pytomography.utils.memory import block_size

#: Factor applied to the memory estimates of :meth:`PETLMSystemMatrix.print_memory_usage`. Summing the arrays a
#: projection allocates underestimates what the device reports, because PyTorch's caching allocator keeps freed
#: blocks; measured peaks of the GATE mMR scan ran 16 to 32 percent above the sum, so the estimates are scaled.
_MEMORY_MARGIN = 1.5

def _float32(x, device) -> torch.Tensor:
    """Contiguous float32 tensor on ``device``: the array form every parallelproj kernel expects."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device=device, dtype=torch.float32).contiguous()

def _pad(object: torch.Tensor) -> torch.Tensor:
    """The object with one voxel of zeros on every face, which is how the PET system matrices hand it to parallelproj.

    parallelproj 1 integrated each ray out to one voxel beyond the outermost voxel centres, where the image (linearly
    interpolated) falls to zero. parallelproj 2 clips rays at the faces of the image, so a ray that leaves through a
    face where the object is not zero loses that last half voxel. Projecting the padded object (with ``object_origin``
    moved one voxel out) gives the parallelproj 1 line integrals; back projections are cropped with :func:`_crop`."""
    return torch.nn.functional.pad(object, (1, 1, 1, 1, 1, 1))

def _crop(object: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pad`: removes the outer voxel on every face."""
    return object[1:-1, 1:-1, 1:-1].contiguous()

def _padded_origin(object_meta: ObjectMeta) -> torch.Tensor:
    """Centre of the first voxel of the padded object (see :func:`_pad`), as parallelproj expects it."""
    return _float32((- np.array(object_meta.shape) / 2 - 0.5) * (np.array(object_meta.dr)), pytomography.device)

class PETLMSystemMatrix(SystemMatrix):
    r"""System matrix of PET list mode data. Forward projections corresponds to computing the expected counts along all LORs specified: in particular it approximates :math:`g_i = \int_{\text{LOR}_i} h(r) f(r) dr` where index :math:`i` corresponds to a particular detector pair and :math:`h(r)` is a Gaussian function that incorporates time-of-flight information (:math:`h(r)=1` for non-time-of-flight). The integral is approximated in the discrete object space using Joseph3D projections. In general, the system matrix implements two different projections, the quantity :math:`H` which projects to LORs corresponding to all detected events, and the quantity :math:`\tilde{H}` which projects to all valid LORs. The quantity :math:`H` is used for standard forward/back projection, while :math:`\tilde{H}` is used to compute the sensitivity image.

        Args:
            object_meta (SPECTObjectMeta): Metadata of object space, containing information on voxel size and dimensions.
            proj_meta (PETLMProjMeta): PET listmode projection space metadata. This information contains the detector ID pairs of all detected events, as well as a scanner lookup table and time-of-flight metadata. In addition, this metadata contains all information regarding event weights, typically corresponding to the effects of attenuation :math:`\mu` and :math:`\eta`. 
            obj2obj_transforms (Sequence[Transform]): Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging.
            attenuation_map (torch.tensor[float] | None, optional): Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of any weights provided in ``proj_meta``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
            scale_projection_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
            N_splits (int): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str): The device on which forward/back projection tensors are output. This is seperate from ``pytomography.device``, which handles internal computations. The reason for having the option of a second device is that the projection space may be very large, and certain GPUs may not have enough memory to store the projections. If ``device`` is not the same as ``pytomography.device``, then one must also specify the same ``device`` in any reconstruction algorithm used. Defaults to ``pytomography.device``.
            lor_device (str): Where the detector IDs of the events and the scanner lookup table are kept. Keeping them on ``pytomography.device`` lets the projector read the LOR coordinates directly, instead of gathering them on the host and copying them across at every projection; that gathering and copying, rather than the projector itself, is most of the cost of a list mode projection. Set this to ``'cpu'`` when the events do not fit in GPU memory, which restores the previous behaviour of copying each chunk across as it is projected. Use :meth:`print_memory_usage` to see what a dataset requires. Defaults to ``pytomography.device``.
            sort_events (bool): Whether to reorder the events so that neighbouring events cross the image in nearly the same place (see :meth:`_sort_events`). This makes the projector's memory accesses local and roughly halves the time of a time of flight list mode projection. It needs the scanner geometry (``proj_meta.info``) and one additional index array per event. Events are only reordered internally: projections are returned, and expected, in the order the events were given. Defaults to True.

    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: PETLMProjMeta,
        obj2obj_transforms: list[Transform] = [],
        attenuation_map: torch.tensor[float] | None = None,
        scale_projection_by_sensitivity: bool = False,
        N_splits: int = 1,
        FOV_scale_enabled: bool = True,
        device: str = pytomography.device,
        lor_device: str = pytomography.device,
        sort_events: bool = True,
    ) -> None:
        super(PETLMSystemMatrix, self).__init__(
            obj2obj_transforms=obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
            )
        self.output_device = device
        if self.proj_meta.tof_meta is not None:
            self.TOF = True
        else:
            self.TOF = False
        self.obj2obj_transforms = obj2obj_transforms
        # the geometry every kernel call needs, in the form it needs (float32, on the projection device); objects are
        # projected with one voxel of zeros on every face (see _pad), so the origin is that of the padded grid
        self.object_origin = _padded_origin(object_meta)
        self.voxel_size = _float32(np.array(object_meta.dr), pytomography.device)
        self.lor_device = lor_device
        self.proj_meta.detector_ids = self.proj_meta.detector_ids.to(lor_device)
        self.proj_meta.scanner_lut = _float32(self.proj_meta.scanner_lut, lor_device)
        self.attenuation_map = attenuation_map
        self.N_splits = N_splits
        self.scale_projection_by_sensitivity = scale_projection_by_sensitivity
        self.sort_events = sort_events
        self._orders = {}
        self.norm_BP = self._backward_full()
        # replace zeros (outside FOV) with small value to avoid NaNs
        self.norm_BP[self.norm_BP < 1e-7] = 1e7
        self.FOV_scale_enabled = FOV_scale_enabled
        
    def _get_object_initial(self, device=pytomography.device):
        # Only consider the space within the FOV
        zmin = (self.object_meta.shape[-1]-1)/2 + float(self.proj_meta.scanner_lut[:,2].min()) /self.object_meta.dr[-1]
        zmax = (self.object_meta.shape[-1]-1)/2 + float(self.proj_meta.scanner_lut[:,2].max()) /self.object_meta.dr[-1]
        zmin = max(0, zmin)
        zmax = max(0,zmax)
        object_initial = torch.ones(self.object_meta.shape).to(device)
        object_initial[:,:,:int(np.ceil(zmin))] = 0
        object_initial[:,:,int(np.floor(zmax)):] = 0
        return object_initial
    
    def _get_prior_FOV_scale(self):
        """Sets scaling for the prior within the FOV.

        Returns:
            torch.Tensor: Prior scaling
        """
        if self.FOV_scale_enabled:
            zmin = (self.object_meta.shape[-1]-1)/2 + float(self.proj_meta.scanner_lut[:,2].min()) /self.object_meta.dr[-1]
            zmax = (self.object_meta.shape[-1]-1)/2 + float(self.proj_meta.scanner_lut[:,2].max()) /self.object_meta.dr[-1]
            zmid = (zmin + zmax) / 2
            zmin = max(0, zmin)
            zmax = max(0,zmax)
            # Set axial FOV scaling
            z = torch.arange(self.object_meta.shape[-1]).to(pytomography.device)
            FOV_scale = (zmid - torch.abs(z - zmid)) / zmid
            FOV_scale[FOV_scale<0] = 0
            FOV_scale = torch.ones(self.object_meta.shape).to(pytomography.device) * FOV_scale
        else:
            FOV_scale = torch.ones(self.object_meta.shape).to(pytomography.device)
        return FOV_scale
    
    def _event_order(self, subset_idx: int | None) -> torch.Tensor | None:
        r"""Permutation putting the events of subset :math:`m` in the order of the sinogram bin they fall in, so that events next to each other cross the image in nearly the same place.

        Events are recorded in the order they were detected, so neighbouring events are unrelated in space, and neighbouring threads of the projector read (and, back projecting, atomically write) unrelated parts of the image. Ordering them makes those accesses local: on an RTX 5090, a 4 million event non time of flight forward projection of a GATE mMR scan takes 9.8 ms as recorded and 2.2 ms ordered, and back projection 22.2 ms against 10.6 ms. Only the order the LORs are handed to the projector in changes; projections are returned, and accepted, in the order the events were given.

        The permutation is built once per subset and kept (4 bytes per event in total). It is None when ``sort_events`` is off or the scanner geometry (``proj_meta.info``) is not available.

        Args:
            subset_idx (int | None): Subset index :math:`m`, or None for all events.

        Returns:
            torch.Tensor | None: The permutation, or None.
        """
        if subset_idx in self._orders:
            return self._orders[subset_idx]
        order = None
        if self.sort_events and getattr(self.proj_meta, 'info', None) is not None:
            ids = self.proj_meta.detector_ids if subset_idx is None else self.proj_meta.detector_ids[self.subset_indices_array[subset_idx].to(self.proj_meta.detector_ids.device)]
            device = ids.device
            # the sinogram lookup tables are built by a loop over all crystal pairs, so build them once
            if getattr(self, '_sinogram_tables', None) is None:
                from pytomography.io.PET.shared import sinogram_coordinates
                lor_coordinates, sinogram_index = sinogram_coordinates(self.proj_meta.info)
                self._sinogram_tables = (lor_coordinates.to(device), sinogram_index.to(device))
            lor_coordinates, sinogram_index = self._sinogram_tables
            crystals_per_ring = self.proj_meta.info['NrCrystalsPerRing']
            # the sort keys are computed a block of events at a time (all 50.8M events of the mMR at once took about
            # 5 GB of temporaries on the GPU, which stayed in PyTorch's cache)
            key = torch.empty(ids.shape[0], dtype=torch.long, device=device)
            for start in range(0, ids.shape[0], 2**22):
                block = ids[start:start + 2**22, :2].to(torch.long)
                within_ring_id, pair_order = (block % crystals_per_ring).sort(dim=1, descending=True)
                ring_ids = (block // crystals_per_ring).gather(1, pair_order)
                angular_radial = lor_coordinates[within_ring_id[:,0], within_ring_id[:,1]]
                plane = sinogram_index[ring_ids[:,0], ring_ids[:,1]]
                key[start:start + block.shape[0]] = (angular_radial[:,0] * lor_coordinates.shape[1] + angular_radial[:,1]) * sinogram_index.numel() + plane
            del ids
            order = torch.argsort(key).to(torch.int32)
        self._orders[subset_idx] = order
        return order

    def print_memory_usage(self, n_subsets: int | None = None) -> None:
        """Prints what this system matrix keeps in memory and what a projection needs on top of it, against the memory the device has free.

        A projection of a subset is reported separately from a projection of every event at once, because they differ by more than the ratio of their sizes: reconstruction algorithms project one subset at a time, while ``compute_normalization_factor`` and a plain ``forward()`` project everything. The first projection of each subset also builds that subset's event ordering, which is the largest temporary of all; it is freed afterwards and not paid again.

        The estimate does not include what the reconstruction algorithm, the likelihood and the object transforms hold (the measured projections, the additive term, the current estimate), so leave headroom.

        Args:
            n_subsets (int | None, optional): Number of subsets to report for. Defaults to the number this system matrix has been configured with, or 1.
        """
        n_events = self.proj_meta.detector_ids.shape[0]
        n_voxels = int(np.prod(self.object_meta.shape))
        n_voxels_padded = int(np.prod([n + 2 for n in self.object_meta.shape]))   # the object as parallelproj gets it (see _pad)
        id_bytes = self.proj_meta.detector_ids.element_size() * self.proj_meta.detector_ids.shape[1]
        if n_subsets is None:
            n_subsets = len(self.subset_indices_array) if hasattr(self, 'subset_indices_array') else 1
        on_lor_device = str(self.lor_device) == str(pytomography.device)
        stored = {'event detector IDs': id_bytes * n_events,
                  'scanner lookup table': self.proj_meta.scanner_lut.element_size() * self.proj_meta.scanner_lut.numel()}
        if not on_lor_device:
            stored = {f'{k} (on {self.lor_device})': v for k, v in stored.items()}
        stored['sensitivity image'] = 4 * n_voxels
        if self.attenuation_map is not None:
            stored['attenuation map'] = 4 * n_voxels
        if self.sort_events:
            stored['event ordering'] = 4 * n_events
        resident = sum(v for k, v in stored.items() if on_lor_device or '(on ' not in k)

        def projection(n: int) -> dict:
            chunk = int(np.ceil(n / self.N_splits))
            need = {'LOR detector IDs for the projector': id_bytes * n if (self.sort_events or not on_lor_device) else 0,
                    'LOR coordinates of one chunk': 2 * 3 * 4 * chunk,
                    'projection values': 4 * n,
                    'object and its transforms': 3 * 4 * n_voxels + 4 * n_voxels_padded}
            return {k: v for k, v in need.items() if v}

        print(f"PETLMSystemMatrix memory: {n_events:,} events, object {tuple(self.object_meta.shape)}, "
              f"N_splits={self.N_splits}, lor_device={self.lor_device}")
        print(f"  kept for the lifetime of the system matrix:")
        for k, v in stored.items():
            print(f"    {k:40s} {v/1e9:8.3f} GB")
        print(f"    {'resident on ' + str(pytomography.device):40s} {resident/1e9:8.3f} GB")
        for label, n in ((f'one subset of {n_subsets}', int(np.ceil(n_events / n_subsets))), ('every event at once', n_events)):
            need = projection(n)
            build = (id_bytes + 8 + 8) * n if self.sort_events else 0   # sorting keys, freed afterwards
            print(f"  projecting {label} ({n:,} events):")
            for k, v in need.items():
                print(f"    {k:40s} {v/1e9:8.3f} GB")
            peak = _MEMORY_MARGIN * (resident + sum(need.values()))
            print(f"    {'estimated peak':40s} {peak/1e9:8.3f} GB"
                  + (f"  ({_MEMORY_MARGIN*(resident + sum(need.values()) + build)/1e9:.3f} GB the first time, while its ordering is built)" if build else ""))
        print(f"  peaks include a {_MEMORY_MARGIN:.1f}x margin: PyTorch's allocator holds freed blocks in its cache, so the "
              f"peak it reports runs above the sum of the arrays above.")
        if torch.cuda.is_available() and str(pytomography.device).startswith('cuda'):
            free, total = torch.cuda.mem_get_info()
            worst = _MEMORY_MARGIN * (resident + sum(projection(n_events).values()) + ((id_bytes + 16) * n_events if self.sort_events else 0))
            print(f"  device has {free/1e9:.3f} GB free of {total/1e9:.3f} GB"
                  + ("" if worst < free else "  -- the all-event projection does not fit; raise N_splits, "
                                             "set lor_device='cpu', or project subsets only"))

    def _n_events(self, subset_idx: int | None) -> int:
        """Number of events of subset ``subset_idx`` (all the events for None)."""
        if subset_idx is None:
            return self.proj_meta.detector_ids.shape[0]
        return self.subset_indices_array[subset_idx].shape[0]

    def _chunk_events(self, subset_idx: int | None, order: torch.Tensor | None, start: int, end: int) -> torch.Tensor:
        """Positions, within subset ``subset_idx``, of the events of chunk ``start:end`` of the projector order (sinogram
        order when ``order`` is given, the order the events were given in otherwise), as a long tensor on the device
        the detector IDs are on."""
        device = self.proj_meta.detector_ids.device
        if order is None:
            return torch.arange(start, end, device=device)
        return order[start:end].to(device=device, dtype=torch.long)

    def _subset_positions(self, subset_idx: int | None) -> torch.Tensor | None:
        """Positions of subset ``subset_idx``'s events among all the events, on the device the detector IDs are on (None
        for all the events). A projection takes them there once and every chunk indexes them (taking them there for
        each chunk copied 29 MB to the GPU ten times per projection of a 14th of the mMR's events)."""
        if subset_idx is None:
            return None
        return self.subset_indices_array[subset_idx].to(self.proj_meta.detector_ids.device)

    def _event_ids(self, positions: torch.Tensor | None, events: torch.Tensor | None) -> torch.Tensor:
        """Detector IDs (and TOF bins) of ``events``, positions within the subset whose events are at ``positions``
        among all the events (from :meth:`_subset_positions`; None for all the events). All of the subset's events, in
        the order they were given, for ``events`` None."""
        ids = self.proj_meta.detector_ids
        if positions is not None:
            return ids[positions if events is None else positions[events]]
        return ids if events is None else ids[events]

    def _chunks(self, n: int) -> list:
        """Start/end of the ``N_splits`` contiguous chunks the projection is computed in."""
        size = int(np.ceil(n / self.N_splits))
        return [(s, min(n, s + size)) for s in range(0, n, size)] if n else []

    def _lor_coordinates(self, idx: torch.Tensor) -> tuple:
        """Coordinates of the two detectors of each LOR of ``idx``, on the device the projector runs on. When the detector IDs and the lookup table already live there, this is a gather with no transfer."""
        lut = self.proj_meta.scanner_lut
        idx = idx.to(lut.device)
        xstart = lut[idx[:,0].to(torch.long)]
        xend = lut[idx[:,1].to(torch.long)]
        if xstart.device != torch.device(pytomography.device):
            xstart, xend = xstart.to(pytomography.device), xend.to(pytomography.device)
        return xstart.contiguous(), xend.contiguous()

    def _tof_arguments(self, idx: torch.Tensor) -> tuple:
        """Time of flight arguments of a chunk, in the form parallelproj expects: float32 arrays, and the bin index of each event as a 0-based int16."""
        tof_meta = self.proj_meta.tof_meta
        sigma = _float32(tof_meta.sigma, pytomography.device).reshape(-1)
        center_offset = _float32(tof_meta.center_offset, pytomography.device).reshape(-1)
        # parallelproj 2 takes the bin number itself (0 to num_bins-1), not an offset from the central bin
        bins = idx[:,2].to(device=pytomography.device, dtype=torch.int16).contiguous()
        return float(tof_meta.bin_width), sigma, center_offset, bins, int(tof_meta.num_bins), float(tof_meta.n_sigmas)

    def _compute_attenuation_probability_projection(self, idx: torch.tensor) -> torch.tensor:
        """Computes probabilities of photons being detected along an LORs corresponding to ``idx``.

        Args:
            idx (torch.tensor): Indices of the detector pairs.

        Returns:
            torch.Tensor: The probabilities of photons being detected along the detector pairs.
        """
        proj = torch.zeros(idx.shape[0], dtype=torch.float32, device=self.output_device)
        attenuation_map = _pad(_float32(self.attenuation_map, pytomography.device))
        for start, end in self._chunks(idx.shape[0]):
            xstart, xend = self._lor_coordinates(idx[start:end])
            chunk = torch.zeros(end - start, dtype=torch.float32, device=pytomography.device)
            parallelproj_core.joseph3d_fwd(xstart, xend, attenuation_map, self.object_origin, self.voxel_size, chunk)
            proj[start:end] = torch.exp(-chunk).to(self.output_device)
        return proj

    def _compute_sensitivity_projection(self, all_ids: bool = True) -> torch.Tensor:
        """Computes the sensitivty projection (when back projected, gives normalization factor)

        Args:
            all_ids (bool, optional): Compute for all detector IDs. Defaults to True.

        Returns:
            torch.Tensor: Sesitivity factor for detector IDs
        """
        if all_ids:
            # every line of response of the sensitivity image, filled a block at a time (all 411 million crystal pairs
            # of the mMR at once took a 3.3 GB list of pairs, plus the temporaries of making it)
            proj = torch.empty(self._n_sensitivity_lors())
            for offset, ids in self._sensitivity_lor_blocks(self._sensitivity_block_size(default=2**24)):
                proj[offset:offset + ids.shape[0]] = self._sensitivity_weights(ids, offset)
            return proj
        detector_ids = self.proj_meta.detector_ids
        proj = torch.ones(detector_ids.shape[0])
        # Load normalization weights for the specific detector IDs: the norm factor of each event's crystal pair, a block
        # of events at a time (all 50.8M events of the mMR at once took about 2.5 GB of host temporaries)
        if self.proj_meta.weights_sensitivity is not None:
            weights = self.proj_meta.weights_sensitivity.cpu()
            n_crystals = self.proj_meta.info['NrCrystalsPerRing'] * self.proj_meta.info['NrRings']
            for start in range(0, detector_ids.shape[0], 2**22):
                ids_sorted, _ = torch.sort(detector_ids[start:start + 2**22, :2], 1)
                index = crystal_pair_index(ids_sorted[:, 0], ids_sorted[:, 1], n_crystals).cpu()
                proj[start:start + index.shape[0]] *= weights[index]
        # Scale the weights by attenuation image if its provided in the system matrix
        if self.attenuation_map is not None:
            proj *= self._compute_attenuation_probability_projection(detector_ids).cpu()
        return proj
        
    def _n_sensitivity_lors(self) -> int:
        """Number of lines of response of the sensitivity image: ``detector_ids_sensitivity``, or every pair of crystals."""
        if self.proj_meta.detector_ids_sensitivity is not None:
            return self.proj_meta.detector_ids_sensitivity.shape[0]
        n_crystals = self.proj_meta.scanner_lut.shape[0]
        return n_crystals * (n_crystals - 1) // 2

    def _sensitivity_block_size(self, default: int) -> int:
        """Lines of response of the sensitivity image handled at once: as many as fit in an eighth of the memory budget (about 48 bytes each on the host; :func:`pytomography.set_memory_budget`), or ``default`` without a budget."""
        return block_size(48, default=default)

    def _sensitivity_lor_blocks(self, lors_per_block: int):
        """The lines of response of the sensitivity image a block at a time, as ``(offset, detector_ids)``: ``offset`` is the index of the block's first line of response in ``detector_ids_sensitivity`` or, without it, among all crystal pairs in ``torch.combinations`` order. That is the order of ``weights_sensitivity``."""
        ids = self.proj_meta.detector_ids_sensitivity
        if ids is not None:
            for start in range(0, ids.shape[0], lors_per_block):
                yield start, ids[start:start + lors_per_block]
        else:
            yield from crystal_pair_blocks(self.proj_meta.scanner_lut.shape[0], lors_per_block)

    def _sensitivity_weights(self, detector_ids: torch.Tensor, offset: int) -> torch.Tensor:
        """Sensitivity weights :math:`w` of a block from ``_sensitivity_lor_blocks``: the normalization weight times the probability that the photons are not attenuated."""
        proj = torch.ones(detector_ids.shape[0])
        if self.proj_meta.weights_sensitivity is not None:
            proj *= self.proj_meta.weights_sensitivity[offset:offset + detector_ids.shape[0]].cpu()
        if self.attenuation_map is not None:
            proj *= self._compute_attenuation_probability_projection(detector_ids).cpu()
        return proj

    def _backward_full(self, N_splits: int = 20):
        r"""Computes full back projection :math:`\tilde{H}^T w g` where :math:`w` is the weighting specified in the projection metadata that accounts for attenuation/normalization correction. If ``proj`` ($g$) is not provided, then uses a tensor of all ones (this is used to compute the normalization factor).

        The lines of response (every crystal pair, or ``detector_ids_sensitivity``) are gone through a block at a time: all 411 million crystal pairs of the mMR took a 3.3 GB list of pairs and 1.6 GB arrays of their weights at once.

        Args:
            N_splits (int, optional): Optionally splits up computation to save memory on GPU. Defaults to 20.
        """
        # parallelproj adds into the image it is given, so every block accumulates into one (padded) buffer
        norm_BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        lors_per_call = math.ceil(self._n_sensitivity_lors() / N_splits)   # what one projector call back projected before
        for offset, detector_ids in self._sensitivity_lor_blocks(min(lors_per_call, self._sensitivity_block_size(default=lors_per_call))):
            proj = self._sensitivity_weights(detector_ids, offset)
            xstart, xend = self._lor_coordinates(detector_ids)
            parallelproj_core.joseph3d_back(
                xstart, xend, norm_BP, self.object_origin, self.voxel_size,
                _float32(proj + pytomography.delta, pytomography.device))
        norm_BP = _crop(norm_BP)
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            norm_BP  = transform.backward(norm_BP)
        return norm_BP.cpu()
    
    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.detector_ids.shape[0]).to(torch.long).cpu()
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        # the projector order of each subset's events belongs to one partition: forget those of another (the order of
        # all the events, key None, stays)
        if getattr(self, '_orders_n_subsets', None) != n_subsets:
            self._orders = {key: order for key, order in self._orders.items() if key is None}
            self._orders_n_subsets = n_subsets
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. For LM PET, its always the case that :math:`g_m=1`, but this function is still required for subsampling scatter :math:`s_m` as is required in certain reconstruction algorithms

        Args:
            projections (torch.Tensor): total projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.Tensor: subsampled projections :math:`g_m`.
        """
        # Needs to consider cases where projection is simply a 1 element tensor in the numerator, but also cases of scatter where it is a longer tensor
        
        if (projections.shape[0]>1)*(subset_idx is not None):
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices]
        else:
            proj_subset = projections
        return proj_subset
    
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
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.detector_ids.shape[0]

    def compute_normalization_factor(self, subset_idx: int | None = None) -> torch.tensor:
        r"""Function called by reconstruction algorithms to get the sensitivty image :math:`\tilde{H}_m^T w`.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`. If none, then considers backprojection over all subsets. Defaults to None.

        Returns:
            torch.tensor: Normalization factor.
        """
        
        if subset_idx is None:
            fraction_considered = 1
        else:
            fraction_considered = self.subset_indices_array[subset_idx].shape[0] / self.proj_meta.detector_ids.shape[0] 
        return fraction_considered * self.norm_BP.to(self.output_device)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int = None,
    ) -> torch.tensor:
        """Computes forward projection. In the case of list mode PET, this corresponds to the expected number of detected counts along each LOR corresponding to a particular object.

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.

        Returns:
            torch.tensor: Projections corresponding to the expected number of counts along each LOR.
        """ 
        # Apply object space transforms
        object = _float32(object, pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        object = _pad(_float32(object, pytomography.device))
        # The LORs go to the projector in sinogram order, a chunk at a time: each chunk's events are taken through the
        # event order, and its projections written back to where those events are (no sorted copy of all the events)
        n_events = self._n_events(subset_idx)
        order = self._event_order(subset_idx)
        positions = self._subset_positions(subset_idx)
        proj = torch.empty(n_events, dtype=torch.float32, device=self.output_device)
        for start, end in self._chunks(n_events):
            events = self._chunk_events(subset_idx, order, start, end)
            idx_partial = self._event_ids(positions, events)
            xstart, xend = self._lor_coordinates(idx_partial)
            chunk = torch.zeros(end - start, dtype=torch.float32, device=pytomography.device)
            if self.TOF:
                bin_width, sigma, center_offset, bins, num_bins, n_sigmas = self._tof_arguments(idx_partial)
                parallelproj_core.joseph3d_tof_lm_fwd(xstart, xend, object, self.object_origin, self.voxel_size,
                                                      chunk, bin_width, sigma, center_offset, bins, num_bins, n_sigmas)
            else:
                parallelproj_core.joseph3d_fwd(xstart, xend, object, self.object_origin, self.voxel_size, chunk)
            proj[events.to(proj.device)] = chunk.to(self.output_device)
        if self.scale_projection_by_sensitivity:
            if self.proj_meta.weights is None:
                if self.attenuation_map is not None:
                    # proj is in the order the events were given, so the factors are computed in that order too
                    proj = proj * self._compute_attenuation_probability_projection(self._event_ids(positions, None)).to(proj.device)
                else:
                    raise Exception('If scaling by sensitivity, then `weights` must be provided in the projection metadata')
            else:
                proj = proj * self.get_projection_subset(self.proj_meta.weights, subset_idx).to(proj.device)
        return proj.to(self.output_device)
            
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: list[int] = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        """Computes back projection. This corresponds to tracing a sequence of LORs into object space.

        Args:
            proj (torch.tensor): Projections to be back projected
            subset_idx (int, optional): Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.
            return_norm_constant (bool, optional): Whether or not to return the normalization constant: useful in reconstruction algorithms that require :math:`H_m^T 1`. Defaults to False.

        Returns:
            torch.tensor: _description_
        """
        positions = self._subset_positions(subset_idx)
        # Normalization/attenuation scaling (if needed); the same factors as forward, so the two stay adjoint
        if self.scale_projection_by_sensitivity:
            if self.proj_meta.weights is None:
                if self.attenuation_map is not None:
                    proj = proj * self._compute_attenuation_probability_projection(self._event_ids(positions, None)).to(proj.device)
                else:
                    raise Exception('If scaling by sensitivity, then `weights` must be provided in the projection metadata')
            else:
                proj = proj * self.get_projection_subset(self.proj_meta.weights, subset_idx).to(proj.device)
        # The LORs go to the projector in sinogram order, a chunk at a time, with their projections (taken through the
        # event order chunk by chunk, so there is no sorted copy of all the events or projections)
        n_events = self._n_events(subset_idx)
        order = self._event_order(subset_idx)
        # parallelproj adds into the image it is given, so every chunk accumulates into one (padded) buffer
        BP = torch.zeros(tuple(n + 2 for n in self.object_meta.shape), dtype=torch.float32, device=pytomography.device)
        for start, end in self._chunks(n_events):
            events = self._chunk_events(subset_idx, order, start, end)
            idx_partial = self._event_ids(positions, events)
            proj_i = _float32(proj[events.to(proj.device)], pytomography.device)
            xstart, xend = self._lor_coordinates(idx_partial)
            if self.TOF:
                bin_width, sigma, center_offset, bins, num_bins, n_sigmas = self._tof_arguments(idx_partial)
                parallelproj_core.joseph3d_tof_lm_back(xstart, xend, BP, self.object_origin, self.voxel_size,
                                                       proj_i, bin_width, sigma, center_offset, bins, num_bins, n_sigmas)
            else:
                parallelproj_core.joseph3d_back(xstart, xend, BP, self.object_origin, self.voxel_size, proj_i)
        BP = _crop(BP)
        # Apply object transforms
        norm_constant = self.compute_normalization_factor(subset_idx)
        for transform in self.obj2obj_transforms[::-1]:
            if return_norm_constant:
                BP, norm_constant = transform.backward(BP, norm_constant=norm_constant)
            else:
                BP  = transform.backward(BP)
        # Return
        if return_norm_constant:
            return BP.to(self.output_device), norm_constant.to(self.output_device)
        else:
            return BP.to(self.output_device)