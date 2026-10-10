from __future__ import annotations
from collections.abc import Sequence
import functools
import torch
import numpy as np
import pytomography
from pytomography.utils import get_1d_gaussian_kernel
from pytomography.utils.memory import block_size, prefer_lazy

_GEOMETRY_CACHE: dict = {}

def _memoize_by_info(fn):
    """Caches the sinogram lookup tables per geometry ``info`` dictionary. They are built by pure-Python loops over all crystal pairs (seconds for a clinical scanner) and were rebuilt on every call, e.g. once per TOF bin during scatter estimation. The cached tensors are shared between callers and must not be modified in place."""
    @functools.wraps(fn)
    def wrapper(info: dict):
        try:
            key = (fn.__name__, tuple(sorted((k, v.item() if hasattr(v, 'item') else v) for k, v in info.items())))
            hash(key)
        except TypeError:
            return fn(info)
        if key not in _GEOMETRY_CACHE:
            _GEOMETRY_CACHE[key] = fn(info)
        return _GEOMETRY_CACHE[key]
    return wrapper

@_memoize_by_info
def sinogram_coordinates(info: dict) -> Sequence[torch.Tensor]:
    """Obtains two tensors: the first yields the sinogram coordinates (r/theta) given two crystal IDs (shape [N_crystals_per_ring, N_crystals_per_ring, 2]), the second yields the sinogram index given two ring IDs (shape [Nrings, Nrings])

    Args:
        info (dict): PET geometry information dictionary    

    Returns:
        Sequence[torch.Tensor]: LOR coordinates and sinogram index lookup tensors
    """
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = info['NrRings']
    nr_crystals_per_ring = info['NrCrystalsPerRing']

    min_sector_difference = info['min_rsector_difference']
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans

    radial_size = nr_crystals_per_ring - 2 * (min_crystal_difference - 1) - 1
    distance_crystal_id_0_to_first_sector_center = (nr_modules_trans * nr_crystals_trans) / 2
    lor_coordinates = np.zeros((nr_crystals_per_ring, nr_crystals_per_ring, 2))

    for full_ring_crystal_id_1 in range(nr_crystals_per_ring):
        crystal_id_1 = (full_ring_crystal_id_1 % nr_crystals_per_ring) - distance_crystal_id_0_to_first_sector_center
        if crystal_id_1 < 0:
                crystal_id_1 += nr_crystals_per_ring
        for full_ring_crystal_id_2 in range(nr_crystals_per_ring):
            crystal_id_2 = (full_ring_crystal_id_2 % nr_crystals_per_ring) - distance_crystal_id_0_to_first_sector_center
            if crystal_id_2 < 0:
                crystal_id_2 += nr_crystals_per_ring
            id_a = 0
            id_b = 0
            if crystal_id_1 < crystal_id_2:
                id_a = crystal_id_1
                id_b = crystal_id_2
            else:
                id_a = crystal_id_2
                id_b = crystal_id_1
            radial = 0
            angular = 0
            if id_b - id_a < min_crystal_difference:
                continue
            else:
                if id_a + id_b >= (3 * nr_crystals_per_ring) / 2 or id_a + id_b < nr_crystals_per_ring / 2:
                    if id_a == id_b:
                        radial = -nr_crystals_per_ring / 2
                    else:
                        radial = ((id_b - id_a - 1) / 2) - ((nr_crystals_per_ring - (id_b - id_a + 1)) / 2)
                else:
                    if id_a == id_b:
                        radial = nr_crystals_per_ring / 2
                    else:
                        radial = ((nr_crystals_per_ring - (id_b - id_a + 1)) / 2) - ((id_b - id_a - 1) / 2)

                radial = np.floor(radial)

                if id_a + id_b < nr_crystals_per_ring / 2:
                    angular = (2 * id_a + nr_crystals_per_ring + radial) / 2
                else:
                    if id_a + id_b >= (3 * nr_crystals_per_ring) / 2:
                        angular = (2 * id_a - nr_crystals_per_ring + radial) / 2
                    else:
                        angular = (2 * id_a - radial) / 2
                lor_coordinates[full_ring_crystal_id_1, full_ring_crystal_id_2, 0] = np.floor(angular)
                lor_coordinates[full_ring_crystal_id_1, full_ring_crystal_id_2, 1] = np.floor(radial + radial_size / 2)
    sinogram_index = np.zeros((nr_rings, nr_rings))
    for ring1 in range(1, nr_rings+1):
        for ring2 in range(1, nr_rings+1):
            ring_difference = abs(ring2 - ring1)
            if ring_difference == 0:
                current_sinogram_index = ring1
            else:
                current_sinogram_index = nr_rings
                if ring1 < ring2:
                    if ring_difference > 1:
                        for ring_distance in range(1, ring_difference):
                            current_sinogram_index += 2 * (nr_rings - ring_distance)
                    current_sinogram_index += ring1
                else:
                    if ring_difference > 1:
                        for ring_distance in range(1, ring_difference):
                            current_sinogram_index += 2 * (nr_rings - ring_distance)
                    current_sinogram_index += nr_rings - ring_difference + ring1 - ring_difference
            sinogram_index[ring1-1, ring2-1] = current_sinogram_index - 1
    return torch.tensor(lor_coordinates).to(torch.long), torch.tensor(sinogram_index).to(torch.long)

@_memoize_by_info
def sinogram_to_spatial(info: dict) -> Sequence[torch.Tensor]:
    """Returns two tensors: the first yields the detector coordinates (x1/y1/x2/y2) of each of the two crystals given the element of the sinogram (shape [N_crystals_per_ring, N_crystals_per_ring, 2, 2]), the second yields the ring coordinates (z1/z2) given two ring IDs (shape [Nrings*Nrings, 2])

    Args:
        info (dict): PET geometry information dictionary

    Returns:
        Sequence[torch.Tensor]: Two tensors yielding spatial coordinates
    """
    scanner_lut = get_scanner_LUT(info)
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = nr_sectors_axial * nr_modules_axial * nr_crystals_axial
    nr_crystals_per_ring = nr_sectors_trans * nr_modules_trans * nr_crystals_trans
    min_sector_difference = 0
    min_crystal_difference = min_sector_difference * nr_modules_trans * nr_crystals_trans
    radial_size = int(nr_crystals_per_ring - 2 * (min_crystal_difference - 1) - 1)
    angular_size = int(nr_crystals_per_ring / 2)
    distance_crystal_id_0_to_first_sector_center = (nr_modules_trans * nr_crystals_trans) / 2
    detector_coordinates = np.zeros((angular_size, radial_size, 2, 2), dtype=np.float32)
    # Generates first the coordinates on each sinogram
    for full_ring_crystal_id_1 in range(nr_crystals_per_ring):
        crystal_id_1 = full_ring_crystal_id_1 % nr_crystals_per_ring - distance_crystal_id_0_to_first_sector_center
        if crystal_id_1 < 0:
            crystal_id_1 += nr_crystals_per_ring
        for full_ring_crystal_id_2 in range(nr_crystals_per_ring):
            crystal_id_2 = full_ring_crystal_id_2 % nr_crystals_per_ring - distance_crystal_id_0_to_first_sector_center
            if crystal_id_2 < 0:
                crystal_id_2 += nr_crystals_per_ring
            id_a = 0
            id_b = 0
            if crystal_id_1 < crystal_id_2:
                id_a = crystal_id_1
                id_b = crystal_id_2
            else:
                id_a = crystal_id_2
                id_b = crystal_id_1
            radial = 0
            angular = 0
            if id_b - id_a < min_crystal_difference:
                continue
            else:
                if id_a + id_b >= (3 * nr_crystals_per_ring) / 2 or id_a + id_b < nr_crystals_per_ring / 2:
                    if id_a == id_b:
                        radial = -nr_crystals_per_ring / 2
                    else:
                        radial = ((id_b - id_a - 1) / 2) - ((nr_crystals_per_ring - (id_b - id_a + 1)) / 2)
                else:
                    if id_a == id_b:
                        radial = nr_crystals_per_ring / 2
                    else:
                        radial = ((nr_crystals_per_ring - (id_b - id_a + 1)) / 2) - ((id_b - id_a - 1) / 2)
                radial = np.floor(radial)
                if id_a + id_b < nr_crystals_per_ring / 2:
                    angular = (2 * id_a + nr_crystals_per_ring + radial) / 2
                else:
                    if id_a + id_b >= (3 * nr_crystals_per_ring) / 2:
                        angular = (2 * id_a - nr_crystals_per_ring + radial) / 2
                    else:
                        angular = (2 * id_a - radial) / 2
                if full_ring_crystal_id_1 >= full_ring_crystal_id_2:
                    detector_coordinates[int(np.floor(angular)), int(np.floor(radial + radial_size / 2)), 0, :] = scanner_lut[full_ring_crystal_id_1, 0:2]
                    detector_coordinates[int(np.floor(angular)), int(np.floor(radial + radial_size / 2)), 1, :] = scanner_lut[full_ring_crystal_id_2, 0:2]
    ring_coordinates = np.zeros((nr_rings * nr_rings, 2), dtype=np.float32)
    for ring1 in range(1, nr_rings+1):
        for ring2 in range(1, nr_rings+1):
            ring_difference = abs(ring2 - ring1)
            if ring_difference == 0:
                current_sinogram_index = ring1
            else:
                current_sinogram_index = nr_rings
                if ring1 < ring2:
                    if ring_difference > 1:
                        for ring_distance in range(1, ring_difference):
                            current_sinogram_index += 2 * (nr_rings - ring_distance)
                    current_sinogram_index += ring1
                else:
                    if ring_difference > 1:
                        for ring_distance in range(1, ring_difference):
                            current_sinogram_index += 2 * (nr_rings - ring_distance)
                    current_sinogram_index += nr_rings - ring_difference + ring1 - ring_difference
            ring_coordinates[current_sinogram_index-1, 0] = scanner_lut[info['NrCrystalsPerRing']*(ring1-1), 2]
            ring_coordinates[current_sinogram_index-1, 1] = scanner_lut[info['NrCrystalsPerRing']*(ring2-1), 2]
    return torch.tensor(detector_coordinates).to(torch.float32), torch.tensor(ring_coordinates).to(torch.float32)

def _is_array_index(index) -> bool:
    """Whether ``index`` indexes a tensor with an array of positions (a list, an array or a tensor that is not 0-d)."""
    if isinstance(index, (list, tuple, np.ndarray)):
        return True
    return isinstance(index, torch.Tensor) and (index.ndim > 0 or index.dtype == torch.bool)

class LazySinogram:
    r"""A sinogram that is computed one group of angles at a time, when those angles are asked for, instead of being held in memory.

    A sinogram with 21 TOF bins of the Siemens Biograph mMR (224 angles, 449 radial bins, 4096 ring pairs) takes 34.6 GB, so a TOF reconstruction that holds its data, its additive term and the scatter estimate as dense sinograms needs over 100 GB of memory. Reconstruction algorithms only ever use one subset of angles at a time (:meth:`PETSinogramSystemMatrix.get_projection_subset` indexes the first dimension of the projections), and so does the scaling of the scatter estimate, so these sinograms never need to exist whole. A ``LazySinogram`` can be passed wherever a likelihood takes projections or an additive term:

    * ``sinogram[angles]`` (an int, a slice, or a list or tensor of angle indices, in any order) computes those angles. Further indices are applied to each group of angles as it is computed, so ``sinogram[:, :, :64, 10].sum(dim=0)`` never holds the whole sinogram.
    * Arithmetic with numbers, with tensors whose first dimension is the same angles (such as a dense randoms or sensitivity sinogram; use ``unsqueeze`` to add trailing dimensions), and with other lazy sinograms gives a lazy sinogram, e.g. ``(randoms.unsqueeze(-1) + scatter) / sensitivity``.
    * ``to_dense()`` computes the whole sinogram as one tensor.

    The angles are computed in groups of at most :attr:`chunk_bytes` bytes (and a sixteenth of the memory budget, if one is set with :func:`pytomography.set_memory_budget`) whenever more than one group is asked for at once. Values are float32 on the CPU; :meth:`compute_at` gives a group of angles on another device, computed there when the sinogram can be.

    Args:
        compute (Callable[[torch.Tensor], torch.Tensor]): Computes the sinogram at the given angles: it gets a 1D long tensor of angle indices on the CPU and returns a new tensor of shape ``[len(angles), *shape[1:]]``.
        shape (Sequence[int]): Shape of the whole sinogram; the first dimension is the angles.
        description (str, optional): What the sinogram holds, for its ``repr``. Defaults to ''.
        compute_on (Callable[[torch.Tensor, torch.device], torch.Tensor] | None, optional): Like ``compute``, but returns the angles as a new tensor on the device it is given, computed there. Defaults to None: :meth:`compute_at` computes on the CPU and copies.
        memory_bytes (float, optional): Host memory the sinogram keeps to compute itself (e.g. its events), in bytes, for memory estimates (:attr:`memory_bytes`). Defaults to 0.
    """
    #: Largest group of angles computed at once when more than one group is asked for, in bytes.
    chunk_bytes = 1e9
    dtype = torch.float32
    device = torch.device('cpu')

    def __init__(self, compute, shape: Sequence[int], description: str = '', compute_on=None, memory_bytes: float = 0) -> None:
        self._compute = compute
        self._compute_on = compute_on
        self.shape = torch.Size(shape)
        self.description = description
        #: Host memory (bytes) the sinogram keeps to compute itself: its events, its interpolation samples, the tensors
        #: its arithmetic holds. Memory estimates count this instead of the size of the whole sinogram.
        self.memory_bytes = float(memory_bytes)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"LazySinogram(shape={tuple(self.shape)}" + (f", {self.description}" if self.description else "") + ")"

    def _angles_per_chunk(self) -> int:
        """Number of angles in one group of at most :attr:`chunk_bytes` bytes (and a sixteenth of the memory budget)."""
        limit = self.chunk_bytes if pytomography.memory_budget is None else min(self.chunk_bytes, pytomography.memory_budget / 16)
        return max(1, int(limit // (4 * int(np.prod(self.shape[1:])))))

    def _angle_indices(self, index) -> tuple[torch.Tensor, bool]:
        """The angles an index of the first dimension selects, as a 1D long tensor, and whether that dimension is dropped (an integer index)."""
        n = self.shape[0]
        if isinstance(index, (int, np.integer)) or (isinstance(index, torch.Tensor) and index.ndim == 0 and index.dtype != torch.bool):
            i = int(index)
            if not -n <= i < n:
                raise IndexError(f"angle {i} is out of range for a sinogram with {n} angles")
            return torch.tensor([i % n]), True
        if isinstance(index, slice):
            return torch.arange(n)[index], False
        if not _is_array_index(index):
            raise IndexError(f"index a LazySinogram along its first dimension (the angles) first, with an int, a slice, a list or a tensor; got {index!r}")
        angles = torch.as_tensor(index).cpu()
        if angles.dtype == torch.bool:
            if angles.shape != (n,):
                raise IndexError(f"a boolean index of the angles must have shape ({n},), got {tuple(angles.shape)}")
            angles = angles.nonzero().flatten()
        if angles.ndim != 1 or angles.is_floating_point():
            raise IndexError(f"the angles must be a 1D integer index, got shape {tuple(angles.shape)} and dtype {angles.dtype}")
        angles = angles.to(torch.long)
        if angles.numel() and (angles.min() < -n or angles.max() >= n):
            raise IndexError(f"angles must lie in [-{n}, {n}), got {int(angles.min())} to {int(angles.max())}")
        return angles % n, False

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 0:
            raise IndexError("index a LazySinogram along its first dimension (the angles) first")
        angles, drop = self._angle_indices(key[0])
        rest = key[1:]
        # Further indices are applied to each group of angles; that equals indexing the whole sinogram as long as the
        # first dimension stays first, which holds for one array index in ``rest`` next to slices (and for basic indices)
        if any(_is_array_index(k) for k in rest):
            if not isinstance(key[0], slice) or sum(_is_array_index(k) for k in rest) > 1 or any(isinstance(k, (int, np.integer)) for k in rest):
                raise IndexError("a LazySinogram supports one array index after the angles, with slices for the angles and no integer indices")
        index = (slice(None),) + rest
        per_chunk = self._angles_per_chunk()
        if len(angles) <= per_chunk:
            values = self._compute(angles)
            if rest:
                values = values[index].clone()   # a copy, so the computed angles are freed
            return values[0] if drop else values
        out = None
        for start in range(0, len(angles), per_chunk):
            part = self._compute(angles[start:start + per_chunk])
            if rest:
                part = part[index]
            if out is None:
                out = torch.empty((len(angles), *part.shape[1:]), dtype=part.dtype, device=part.device)
            out[start:start + part.shape[0]] = part
            del part
        return out

    def compute_at(self, angles: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        """The sinogram at ``angles`` as a new tensor on ``device``, all the angles at once. A sinogram that can be computed on a device (binned list mode events, the interpolated scatter estimate, and arithmetic of those with tensors and numbers) is computed there, without passing through host memory; any other is computed on the CPU and copied.

        Args:
            angles (torch.Tensor): 1D long tensor of angle indices.
            device (str | torch.device): Device of the result.

        Returns:
            torch.Tensor: Sinogram at those angles, of shape ``[len(angles), *shape[1:]]``.
        """
        angles = torch.as_tensor(angles).cpu().to(torch.long)
        device = torch.device(device)
        if self._compute_on is not None:
            return self._compute_on(angles, device)
        return self._compute(angles).to(device)

    def to_dense(self) -> torch.Tensor:
        """The whole sinogram as one tensor, computed a group of angles at a time.

        Returns:
            torch.Tensor: Sinogram of shape :attr:`shape`.
        """
        return self[:]

    def cpu(self) -> LazySinogram:
        return self

    def to(self, *args, **kwargs) -> LazySinogram:
        """Accepts only the device and dtype the sinogram already has (the CPU and float32), so that code moving its projections there works; a lazy sinogram cannot move anywhere else."""
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.dtype):
                if value != self.dtype:
                    raise TypeError(f"a LazySinogram is {self.dtype}; it cannot be converted to {value}")
            elif isinstance(value, (str, torch.device)):
                if torch.device(value).type != self.device.type:
                    raise TypeError(f"a LazySinogram is computed on the CPU; it cannot be moved to {value}")
        return self

    def _combine(self, other, op, reflected: bool = False):
        """Lazy ``op(self, other)`` (``op(other, self)`` if ``reflected``), computed at the angles asked for."""
        if isinstance(other, LazySinogram):
            if other.shape[0] != self.shape[0]:
                raise ValueError(f"cannot combine lazy sinograms with {self.shape[0]} and {other.shape[0]} angles")
            other_at = lambda angles: other[angles]
            other_on = lambda angles, device: other.compute_at(angles, device)
            shape = torch.broadcast_shapes(self.shape, other.shape)
            other_bytes = other.memory_bytes
        elif isinstance(other, torch.Tensor) and other.ndim > 0:
            if other.ndim != self.ndim or other.shape[0] != self.shape[0]:
                raise ValueError(f"cannot combine a LazySinogram of shape {tuple(self.shape)} with a tensor of shape {tuple(other.shape)}: "
                                 "the tensor's first dimension must be the same angles, with as many dimensions (use unsqueeze to add trailing ones)")
            other_at = lambda angles: other[angles.to(other.device)].to(self.device)
            other_on = lambda angles, device: other[angles.to(other.device)].to(device)
            shape = torch.broadcast_shapes(self.shape, other.shape)
            other_bytes = other.untyped_storage().nbytes() if other.device.type == 'cpu' else 0
        elif isinstance(other, (int, float, np.number)) or (isinstance(other, torch.Tensor) and other.ndim == 0):
            other_at = lambda angles: other
            other_on = lambda angles, device: other
            shape = self.shape
            other_bytes = 0
        else:
            return NotImplemented
        if reflected:
            compute = lambda angles: op(other_at(angles), self[angles])
            compute_on = lambda angles, device: op(other_on(angles, device), self.compute_at(angles, device))
        else:
            compute = lambda angles: op(self[angles], other_at(angles))
            compute_on = lambda angles, device: op(self.compute_at(angles, device), other_on(angles, device))
        return LazySinogram(compute, shape, description=self.description, compute_on=compute_on, memory_bytes=self.memory_bytes + other_bytes)

    def __add__(self, other):
        return self._combine(other, torch.add)

    def __radd__(self, other):
        return self._combine(other, torch.add, reflected=True)

    def __sub__(self, other):
        return self._combine(other, torch.sub)

    def __rsub__(self, other):
        return self._combine(other, torch.sub, reflected=True)

    def __mul__(self, other):
        return self._combine(other, torch.mul)

    def __rmul__(self, other):
        return self._combine(other, torch.mul, reflected=True)

    def __truediv__(self, other):
        return self._combine(other, torch.div)

    def __rtruediv__(self, other):
        return self._combine(other, torch.div, reflected=True)

def _event_bins(detector_ids: torch.Tensor, info: dict, num_tof_bins: int | None = None, events_per_chunk: int | None = None) -> tuple:
    """The sinogram bin of each list mode event, as ``listmode_to_sinogram`` and ``sinogram_to_listmode`` find it: the flat (angle, radial bin, plane) index of ``_bin_keys``, whether it lies inside the sinogram, and, for TOF, the TOF bin as the sinogram stores it (mirrored when the event's two crystals were swapped to look the bin up). The events are processed a chunk at a time on their own device, so the temporaries stay small (all 50 million events of the GATE mMR scan at once took about 6 GB); the results are returned on the CPU.

    Args:
        detector_ids (torch.Tensor): [N, 2] or [N, 3] detector IDs of the events (with the TOF bin as the third column).
        info (dict): PET geometry information dictionary.
        num_tof_bins (int | None, optional): Number of TOF bins; None for non-TOF. Defaults to None.
        events_per_chunk (int | None, optional): Events processed at once. Defaults to None: as many as fit in an eighth of the memory budget (about 130 bytes each), or 2**22 without a budget.

    Returns:
        tuple: key ([N] int64), inside ([N] bool) and TOF bin ([N] int64, or None for non-TOF).
    """
    device = detector_ids.device
    lor_coordinates, sinogram_index = (table.to(device) for table in sinogram_coordinates(info))
    shape = _sinogram_shape(info)
    if events_per_chunk is None:
        events_per_chunk = block_size(130, default=2**22)
    # the results are filled a chunk at a time, rather than concatenated from lists of chunks at the end
    n_events = detector_ids.shape[0]
    keys = torch.empty(n_events, dtype=torch.long)
    insides = torch.empty(n_events, dtype=torch.bool)
    tof_bins = torch.empty(n_events, dtype=torch.long) if num_tof_bins is not None else None
    for start in range(0, n_events, events_per_chunk):
        ids = detector_ids[start:start + events_per_chunk]
        end = start + ids.shape[0]
        within_ring_id = (ids[:,:2] % info['NrCrystalsPerRing']).to(torch.long)
        ring_ids = (ids[:,:2] // info['NrCrystalsPerRing']).to(torch.long)
        within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)
        ring_ids = ring_ids.gather(index=idx, dim=1)
        key, inside = _bin_keys(lor_coordinates[within_ring_id[:,0], within_ring_id[:,1]], sinogram_index[ring_ids[:,0], ring_ids[:,1]], shape)
        keys[start:end] = key
        insides[start:end] = inside
        if num_tof_bins is not None:
            tof_bin = ids[:,2].to(torch.long)
            tof_bins[start:end] = torch.where(idx[:,0] == 1, num_tof_bins - 1 - tof_bin, tof_bin)
    return keys, insides, tof_bins

def _events_by_angle(detector_ids: torch.Tensor, info: dict, num_tof_bins: int | None = None, drop_outside: bool = True, weights: torch.Tensor | None = None, event_index: bool = False, events_per_chunk: int | None = None) -> tuple:
    """The events grouped by the angle of their sinogram bin (``_event_bins``), each angle's events in their order in the list: a counting sort done a chunk of events at a time, in two passes (count the events of each angle, then place them), so that only the grouped arrays are as long as the list. Binning all 107 million events of the GATE mMR brain scan at once and sorting them took about 55 bytes per event (5.9 GB).

    Args:
        detector_ids (torch.Tensor): [N, 2] or [N, 3] detector IDs of the events (with the TOF bin as the third column).
        info (dict): PET geometry information dictionary.
        num_tof_bins (int | None, optional): Number of TOF bins; None for non-TOF. Defaults to None.
        drop_outside (bool, optional): Leave out the events outside the sinogram or its TOF bins, as ``listmode_to_sinogram`` does. If False, every event is kept, and one outside the sinogram raises an ``IndexError``. Defaults to True.
        weights (torch.Tensor | None, optional): [N] weights of the events, to group with them. Defaults to None.
        event_index (bool, optional): Also give each grouped event's index in ``detector_ids``. Defaults to False.
        events_per_chunk (int | None, optional): Events binned at once. Defaults to None: as ``_event_bins``.

    Returns:
        tuple: ``offsets`` ([angles + 1] int64: the events of angle ``a`` are entries ``offsets[a]:offsets[a + 1]`` of the arrays that follow), and for each grouped event its position within its angle ([M] int32: radial bin times planes plus plane), its TOF bin as the sinogram stores it ([M] int16, or None for non-TOF), its weight ([M] float32, or None) and its index in ``detector_ids`` ([M] int64, or None).
    """
    shape = _sinogram_shape(info)
    per_angle = shape[1] * shape[2]
    n_events = detector_ids.shape[0]
    if events_per_chunk is None:
        events_per_chunk = block_size(130, default=2**22)

    def chunk_bins(start: int) -> tuple:
        """Bins, TOF bins and indices of one chunk's events, without the events left out."""
        key, inside, tof_bin = _event_bins(detector_ids[start:start + events_per_chunk], info, num_tof_bins, events_per_chunk)
        index = torch.arange(start, start + key.shape[0])
        if not drop_outside:
            if not bool(inside.all()):
                raise IndexError("some events lie outside the sinogram")
            return key, tof_bin, index
        if tof_bin is not None:
            inside &= (tof_bin >= 0) & (tof_bin < num_tof_bins)   # listmode_to_sinogram bins only events in one of the TOF bins
            tof_bin = tof_bin[inside]
        return key[inside], tof_bin, index[inside]

    counts = torch.zeros(shape[0], dtype=torch.long)
    for start in range(0, n_events, events_per_chunk):
        counts += torch.bincount(chunk_bins(start)[0] // per_angle, minlength=shape[0])
    offsets = torch.zeros(shape[0] + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, 0)
    n = int(offsets[-1])
    within_angle = torch.empty(n, dtype=torch.int32)
    tof_bins = torch.empty(n, dtype=torch.int16) if num_tof_bins is not None else None
    grouped_weights = torch.empty(n, dtype=torch.float32) if weights is not None else None
    indices = torch.empty(n, dtype=torch.long) if event_index else None
    # a chunk's events of an angle go after those of the earlier chunks, in their order (a stable sort within the chunk)
    cursor = offsets[:-1].clone()
    for start in range(0, n_events, events_per_chunk):
        key, tof_bin, index = chunk_bins(start)
        angle = key // per_angle
        order = torch.argsort(angle, stable=True)
        angle = angle[order]
        chunk_counts = torch.bincount(angle, minlength=shape[0])
        position = cursor[angle] + torch.arange(angle.shape[0]) - (torch.cumsum(chunk_counts, 0) - chunk_counts)[angle]
        within_angle[position] = (key[order] % per_angle).to(torch.int32)
        if tof_bins is not None:
            tof_bins[position] = tof_bin[order].to(torch.int16)
        if grouped_weights is not None:
            grouped_weights[position] = weights[start:start + events_per_chunk].to(device='cpu', dtype=torch.float32)[index[order] - start]
        if indices is not None:
            indices[position] = index[order]
        cursor += chunk_counts
    return offsets, within_angle, tof_bins, grouped_weights, indices

def _listmode_to_lazy_sinogram(detector_ids: torch.Tensor, info: dict, tof_meta: PETTOFMeta | None = None, weights: torch.Tensor | None = None) -> LazySinogram:
    """``listmode_to_sinogram`` as a :class:`LazySinogram`: the events are kept, grouped by angle (``_events_by_angle``), as their position within the angle (4 bytes each), their TOF bin (2 bytes) and their weight if any, and the angles asked for are binned when they are asked for. The bins hold exactly what ``listmode_to_sinogram`` gives: counts are exact, and weights are summed in the same order (the events of each angle keep their order). Binned on a GPU (:meth:`LazySinogram.compute_at`), counts are still exact, but weights are added up in an order that can differ (atomic adds)."""
    shape = _sinogram_shape(info)
    num_tof_bins = None if tof_meta is None else int(tof_meta.num_bins)
    offsets, within_angle, tof_bin, weights, _ = _events_by_angle(detector_ids, info, num_tof_bins, weights=weights)
    per_angle = shape[1] * shape[2]
    bins_per_angle = per_angle * (1 if num_tof_bins is None else num_tof_bins)
    out_shape = shape[1:] if num_tof_bins is None else (*shape[1:], num_tof_bins)

    def compute(angles: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        device = torch.device('cpu') if device is None else device
        starts, counts = offsets[angles], offsets[angles + 1] - offsets[angles]
        n = int(counts.sum())
        group = torch.repeat_interleave(torch.arange(len(angles)), counts)                 # which of the angles each event is in
        event = torch.arange(n) + torch.repeat_interleave(starts - (torch.cumsum(counts, 0) - counts), counts)
        # only the events' positions within their angle and TOF bins (6 bytes each) go to the device
        local = group.to(device) * per_angle + within_angle[event].to(device, torch.long)
        if num_tof_bins is not None:
            local = local * num_tof_bins + tof_bin[event].to(device, torch.long)
        values = torch.ones(n, dtype=torch.float32, device=device) if weights is None else weights[event].to(device)
        sinogram = torch.zeros(len(angles) * bins_per_angle, dtype=torch.float32, device=device)
        sinogram.index_add_(0, local, values)   # on the CPU, adds the events in order, like bincount
        return sinogram.reshape(len(angles), *out_shape)
    kept = within_angle.nbytes + offsets.nbytes + (0 if tof_bin is None else tof_bin.nbytes) + (0 if weights is None else weights.nbytes)
    return LazySinogram(compute, (shape[0], *out_shape), description=f"binned from {int(offsets[-1]):,} list mode events", compute_on=compute, memory_bytes=kept)

def listmode_to_sinogram(
    detector_ids: torch.Tensor,
    info: dict,
    weights: torch.Tensor = None,
    normalization: bool = False,
    tof_meta: PETTOFMeta = None,
    lazy: bool | None = None
    ) -> torch.Tensor | LazySinogram:
    """Converts PET listmode data to sinogram

    Args:
        detector_ids (torch.Tensor): Listmode detector ID data
        info (dict): PET geometry information dictionary
        weights (torch.Tensor, optional): Binning weights for each listmode event. Defaults to None.
        normalization (bool, optional): Whether or not this is a normalization sinogram (need to do some extra steps). Defaults to False.
        tof_meta (PETTOFMeta, optional): PET TOF metadata. Defaults to None.
        lazy (bool | None, optional): Return a :class:`LazySinogram`, which bins the events of the angles it is asked for when it is asked for them, instead of the whole sinogram. A TOF sinogram is large (34.6 GB with 21 TOF bins for the Siemens Biograph mMR), while a reconstruction reads one subset of angles at a time. Not available with ``normalization``. Defaults to None: lazy when the whole sinogram would take more than a quarter of the memory budget (:func:`pytomography.set_memory_budget`), or more than 8 GB without a budget.

    Returns:
        torch.Tensor | LazySinogram: PET sinogram
    """
    if lazy is None:
        lazy = not normalization and prefer_lazy(4 * np.prod(_sinogram_shape(info)) * (1 if tof_meta is None else tof_meta.num_bins))
    if lazy:
        if normalization:
            raise NotImplementedError("a normalization sinogram cannot be lazy")
        return _listmode_to_lazy_sinogram(detector_ids, info, tof_meta=tof_meta, weights=weights)
    if tof_meta is not None: # if tof_meta is provided
        return _listmodeTOF_to_sinogramTOF(detector_ids, info, tof_meta, weights=weights)
    # The events are binned on the device they are on (a list mode system matrix keeps them on the GPU)
    device = detector_ids.device
    lor_coordinates, sinogram_index = (table.to(device) for table in sinogram_coordinates(info))
    detector_ids = detector_ids[:,:2]
    within_ring_id = (detector_ids % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (detector_ids // info['NrCrystalsPerRing']).to(torch.long)
    # Need to bin by largest "within_ring_id" first (for use with the "ring_coordinates" function yielding spatial coordinates for each ID-pair at each sinogram coordinate)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)   # stable: a pair with equal within-ring IDs keeps its order on any device
    ring_ids = ring_ids.gather(index=idx, dim=1)
    # Bin sinogram
    shape = _sinogram_shape(info)
    sinogram = _bin_events(*_bin_keys(lor_coordinates[within_ring_id[:,0], within_ring_id[:,1]], sinogram_index[ring_ids[:,0], ring_ids[:,1]], shape), shape, weights)
    # Opposite binning for normalization sinogram, which always considers "ring_id"s in order (this only works because of +/- z symmetry of normalization factors)
    if normalization:
        sinogram += _bin_events(*_bin_keys(lor_coordinates[within_ring_id[:,1], within_ring_id[:,0]], sinogram_index[ring_ids[:,1], ring_ids[:,0]], shape), shape, weights)
        sinogram /= 2
    return sinogram

def crystal_pair_blocks(n_crystals: int, pairs_per_block: int):
    """Every pair of crystals ``(i, j)``, ``i < j``, in the order of ``torch.combinations(torch.arange(n_crystals), 2)``, a block of about ``pairs_per_block`` pairs at a time (whole rows of ``i``), so that the 411 million pairs of a clinical scanner never have to be held at once (3.3 GB as int32, and several times that in the temporaries of using them).

    Args:
        n_crystals (int): Number of crystals.
        pairs_per_block (int): Pairs per block (at least one row of ``i``, ``n_crystals - 1 - i`` pairs, is always taken).

    Yields:
        tuple[int, torch.Tensor]: Index of the block's first pair in ``torch.combinations`` order, and the block's pairs as an [N, 2] long tensor.
    """
    first, offset = 0, 0
    while first < n_crystals - 1:
        last = first + 1
        n_pairs = n_crystals - 1 - first
        while last < n_crystals - 1 and n_pairs + (n_crystals - 1 - last) <= pairs_per_block:
            n_pairs += n_crystals - 1 - last
            last += 1
        i = torch.arange(first, last)
        counts = n_crystals - 1 - i
        crystal_1 = torch.repeat_interleave(i, counts)
        crystal_2 = torch.arange(n_pairs) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts) + crystal_1 + 1
        yield offset, torch.stack([crystal_1, crystal_2], dim=1)
        offset += n_pairs
        first = last

def crystal_pair_index(crystal_1: torch.Tensor, crystal_2: torch.Tensor, n_crystals: int) -> torch.Tensor:
    """Index of each pair of crystals ``(crystal_1, crystal_2)``, ``crystal_1 < crystal_2``, in the order of ``torch.combinations(torch.arange(n_crystals), 2)`` (the order of :func:`crystal_pair_blocks` and of ``weights_sensitivity``). Computed with integers: in float32 the products round, and for the mMR (28,672 crystals) about 9 in 10 pairs were given a neighbouring pair's index.

    Args:
        crystal_1 (torch.Tensor): Lower crystal of each pair.
        crystal_2 (torch.Tensor): Higher crystal of each pair.
        n_crystals (int): Number of crystals.

    Returns:
        torch.Tensor: Index of each pair (int64).
    """
    crystal_1, crystal_2 = crystal_1.to(torch.long), crystal_2.to(torch.long)
    return crystal_1 * (2 * n_crystals - crystal_1 - 1) // 2 + crystal_2 - crystal_1 - 1

def all_pairs_to_sinogram(weights: torch.Tensor, info: dict, normalization: bool = False, pairs_per_chunk: int | None = None) -> torch.Tensor:
    """``listmode_to_sinogram`` of every pair of crystals of the scanner, with a weight for each pair (such as normalization weights), binned a block of pairs at a time.

    The pairs are those of ``torch.combinations(torch.arange(N_crystals), 2)``, in that order, which is the order of ``weights``. For the 411 million pairs of the Siemens Biograph mMR, the pair list alone takes 3.3 GB and binning all pairs at once took tens of GB of temporaries. Here each block of pairs is added into one sinogram (two for a normalization sinogram, one per order of the crystals) in the order ``listmode_to_sinogram`` adds them, so the result is the same.

    Args:
        weights (torch.Tensor): Weight of each crystal pair, in ``torch.combinations`` order.
        info (dict): PET geometry information dictionary.
        normalization (bool, optional): Bin as a normalization sinogram (see ``listmode_to_sinogram``). Defaults to False.
        pairs_per_chunk (int | None, optional): Number of pairs binned at once. Defaults to None: as many as fit in an eighth of the memory budget (about 130 bytes each; :func:`pytomography.set_memory_budget`), or 2**22 without a budget.

    Returns:
        torch.Tensor: PET sinogram.
    """
    n_crystals = int(info['NrCrystalsPerRing'] * info['NrRings'])
    if weights.shape[0] != n_crystals * (n_crystals - 1) // 2:
        raise ValueError(f"expected one weight per crystal pair ({n_crystals * (n_crystals - 1) // 2:,}), got {weights.shape[0]:,}")
    if pairs_per_chunk is None:
        pairs_per_chunk = block_size(130, default=2**22)
    lor_coordinates, sinogram_index = sinogram_coordinates(info)
    shape = _sinogram_shape(info)
    # one sinogram per order of the two crystals that is binned: the normalization sinogram bins each pair both ways
    sinograms = [torch.zeros(int(np.prod(shape)), dtype=torch.float32) for _ in range(2 if normalization else 1)]
    for offset, ids in crystal_pair_blocks(n_crystals, pairs_per_chunk):
        within_ring_id, ring_ids = ids % info['NrCrystalsPerRing'], ids // info['NrCrystalsPerRing']
        within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)   # as in listmode_to_sinogram
        ring_ids = ring_ids.gather(index=idx, dim=1)
        block_weights = weights[offset:offset + ids.shape[0]].to(device='cpu', dtype=torch.float32)
        for (a, b), sinogram in zip(((0, 1), (1, 0)), sinograms):
            key, inside = _bin_keys(lor_coordinates[within_ring_id[:,a], within_ring_id[:,b]], sinogram_index[ring_ids[:,a], ring_ids[:,b]], shape)
            sinogram.index_add_(0, key[inside], block_weights[inside])   # adds in order, like bincount
    sinogram = sinograms[0]
    if normalization:
        sinogram += sinograms[1]
        sinogram /= 2
    return sinogram.reshape(shape)

def _sinogram_shape(info: dict) -> tuple:
    """Shape (angles, radial bins, planes) of the sinogram of a scanner."""
    return (int(info['NrCrystalsPerRing']/2), int(info['NrCrystalsPerRing'])+1, int((info['moduleAxialNr']*info['crystalAxialNr'])**2))

def _bin_keys(angular_radial: torch.Tensor, plane: torch.Tensor, shape: tuple) -> tuple:
    """Flat sinogram index of each event, and which events lie inside the sinogram (``torch.histogramdd``, used before, dropped the others; ``_bin_events`` does too)."""
    inside = (angular_radial[:,0] >= 0) & (angular_radial[:,0] < shape[0]) & (angular_radial[:,1] >= 0) & (angular_radial[:,1] < shape[1]) & (plane >= 0) & (plane < shape[2])
    return (angular_radial[:,0] * shape[1] + angular_radial[:,1]) * shape[2] + plane, inside

def _bin_events(key: torch.Tensor, inside: torch.Tensor, shape: tuple, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Sinogram of events with flat sinogram indices ``key``: the sum of their weights (or their number) in each bin, as a float32 tensor on the CPU.

    The events are counted with ``torch.bincount`` on their own device, which builds one sinogram. ``torch.histogramdd``, used before, ran on the CPU and allocated one sinogram per thread (44 GB with 24 threads for the 1.65 GB sinogram of the mMR). Counts are exact; weighted sums agree to float32 rounding.

    Args:
        key (torch.Tensor): [N] flat sinogram index of each event (see ``_bin_keys``).
        inside (torch.Tensor): [N] whether each event lies inside the sinogram; the others are dropped.
        shape (tuple): sinogram shape (angles, radial bins, planes).
        weights (torch.Tensor | None, optional): [N] weight of each event. Defaults to None (count the events).

    Returns:
        torch.Tensor: sinogram of the given shape.
    """
    device = key.device
    weights = torch.ones(key.shape[0], dtype=torch.float32, device=device) if weights is None else weights.to(device=device, dtype=torch.float32)
    counts = torch.bincount(key[inside], weights=weights[inside], minlength=int(np.prod(shape)))
    return counts.reshape(shape).cpu()

def _listmodeTOF_to_sinogramTOF(
    detector_ids: torch.Tensor,
    info: dict,
    tof_meta: PETTOFMeta,
    weights: torch.Tensor | None = None
    ) -> torch.Tensor:
    """Helper function to ``listmode_to_sinogram`` for TOF data

    Args:
        detector_ids (torch.Tensor): Listmode detector ID data
        info (dict): PET geometry information dictionary
        weights (torch.Tensor, optional): Binning weights for each listmode event. Defaults to None.
        tof_meta (PETTOFMeta, optional): PET TOF metadata. Defaults to None.

    Returns:
        torch.Tensor: PET TOF sinogram
    """
    # The events are binned on the device they are on (a list mode system matrix keeps them on the GPU)
    device = detector_ids.device
    lor_coordinates, sinogram_index = (table.to(device) for table in sinogram_coordinates(info))
    if weights is not None:
        weights = weights.to(device=device, dtype=torch.float32)
    # Sort by decreasing detector ids
    # Only consider events within TOF range
    TOF_bins = detector_ids[:,2].clone()
    detector_ids = detector_ids[:,:2]
    within_ring_id = (detector_ids % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (detector_ids // info['NrCrystalsPerRing']).to(torch.long)
    # Sort by greatest value within ring (required for using various lookup tables)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)   # stable: a pair with equal within-ring IDs keeps its order on any device
    # Opposite detector order
    TOF_bins[idx[:,0]==1] = tof_meta.num_bins - 1 - TOF_bins[idx[:,0]==1]
    ring_ids = ring_ids.gather(index=idx, dim=1)
    # Bin sinogram, one TOF bin at a time (which bounds the counts held on the events' device to one sinogram)
    shape = _sinogram_shape(info)
    key, inside = _bin_keys(lor_coordinates[within_ring_id[:,0], within_ring_id[:,1]], sinogram_index[ring_ids[:,0], ring_ids[:,1]], shape)
    sinogram = torch.empty((*shape, tof_meta.num_bins), dtype=torch.float32)
    for bin in range(tof_meta.num_bins):
        in_bin = TOF_bins == bin
        sinogram[...,bin] = _bin_events(key[in_bin], inside[in_bin], shape, None if weights is None else weights[in_bin])
    return sinogram

def get_detector_ids_from_trans_axial_ids(
    ids_trans_crystal: torch.Tensor,
    ids_trans_submodule: torch.Tensor,
    ids_trans_module: torch.Tensor,
    ids_trans_rsector: torch.Tensor,
    ids_axial_crystal: torch.Tensor,
    ids_axial_submodule: torch.Tensor,
    ids_axial_module: torch.Tensor,
    ids_axial_rsector: torch.Tensor,
    info: dict
    ) -> torch.Tensor:
    """Obtain detector IDs from individual part IDs

    Args:
        ids_trans_crystal (torch.Tensor): Transaxial crystal IDs
        ids_trans_submodule (torch.Tensor): Transaxial submodule IDs
        ids_trans_module (torch.Tensor): Transaxial module IDs
        ids_trans_rsector (torch.Tensor): Transaxial rsector IDs
        ids_axial_crystal (torch.Tensor): Axial crystal IDs
        ids_axial_submodule (torch.Tensor): Axial submodule IDs 
        ids_axial_module (torch.Tensor): Axial module IDs
        ids_axial_rsector (torch.Tensor): Axial rsector IDs 
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: Tensor containing (spatial) detector IDs
    """
    ids_ring = ids_axial_crystal +\
        ids_axial_submodule * info['crystalAxialNr'] +\
        ids_axial_module * info['crystalAxialNr'] * info['submoduleAxialNr'] +\
        ids_axial_rsector * info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr']
    ids_within_ring = ids_trans_crystal +\
        ids_trans_submodule * info['crystalTransNr'] +\
        ids_trans_module * info['crystalTransNr'] * info['submoduleTransNr'] +\
        ids_trans_rsector * info['crystalTransNr'] * info['submoduleTransNr'] * info['moduleTransNr']
    nb_crystal_per_ring = info['crystalTransNr'] * info['moduleTransNr'] * info['submoduleTransNr'] * info['rsectorTransNr']
    ids_detector = ids_ring * nb_crystal_per_ring + ids_within_ring
    return ids_detector

def get_axial_trans_ids_from_info(
    info: dict,
    return_combinations: bool = False,
    sort_by_detector_ids: bool = False
    ):
    """Get axial and transaxial IDs corresponding to each crystal in the scanner

    Args:
        info (dict): PET geometry information dictionary
        return_combinations (bool, optional): Whether or not to return all possible combinations (crystal pairs). Defaults to False.
        sort_by_detector_ids (bool, optional): Whether or not to sort by increasing detector IDs. Defaults to False.

    Returns:
        Sequence[torch.Tensor]: IDs corresponding to axial/transaxial components of each part
    """
    ids_trans_crystal = torch.arange(0, info['crystalTransNr'])
    ids_axial_crystal = torch.arange(0, info['crystalAxialNr'])
    ids_trans_submodule = torch.arange(0, info['submoduleTransNr'])
    ids_axial_submodule = torch.arange(0, info['submoduleAxialNr'])
    ids_trans_module = torch.arange(0, info['moduleTransNr'])
    ids_axial_module = torch.arange(0, info['moduleAxialNr'])
    ids_trans_rsector = torch.arange(0, info['rsectorTransNr'])
    ids_axial_rsector = torch.arange(0, info['rsectorAxialNr'])
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = torch.cartesian_prod(ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector).T
    if sort_by_detector_ids:
        ids_detector = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
        idx_sort = torch.argsort(ids_detector)
        ids_trans_crystal = ids_trans_crystal[idx_sort]
        ids_axial_crystal = ids_axial_crystal[idx_sort]
        ids_trans_submodule = ids_trans_submodule[idx_sort]
        ids_axial_submodule = ids_axial_submodule[idx_sort]
        ids_trans_module = ids_trans_module[idx_sort]
        ids_axial_module = ids_axial_module[idx_sort]
        ids_trans_rsector = ids_trans_rsector[idx_sort]
        ids_axial_rsector = ids_axial_rsector[idx_sort]
    if return_combinations:
        ids_trans_crystal = torch.combinations(ids_trans_crystal, 2)
        ids_axial_crystal = torch.combinations(ids_axial_crystal, 2)
        ids_trans_submodule = torch.combinations(ids_trans_submodule, 2)
        ids_axial_submodule = torch.combinations(ids_axial_submodule, 2)
        ids_trans_module = torch.combinations(ids_trans_module, 2)
        ids_axial_module = torch.combinations(ids_axial_module, 2)
        ids_trans_rsector = torch.combinations(ids_trans_rsector, 2)
        ids_axial_rsector = torch.combinations(ids_axial_rsector, 2)
    return ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector

def get_scanner_LUT(info: dict):
    """Obtains scanner lookup table (gives x/y/z coordinates for each detector ID)

    Args:
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor[N_detectors, 3]: Lookup table
    """
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_info(info)
    ids_detector = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
    Z_modules = ids_axial_module * info['moduleAxialSpacing'] - (info['moduleAxialNr']-1) * info['moduleAxialSpacing'] / 2
    Z_submodules = Z_modules + ids_axial_submodule * info['submoduleAxialSpacing'] - (info['submoduleAxialNr']-1) * info['submoduleAxialSpacing'] / 2
    Z_crystals = Z_submodules + ids_axial_crystal * info['crystalAxialSpacing'] - (info['crystalAxialNr']-1) * info['crystalAxialSpacing'] / 2
    # Get X/Y position of crystals
    # Start by getting X/Y position inside a submodule aligned with the center of the scanner
    Y_modules = ids_trans_module * info['moduleTransSpacing'] - (info['moduleTransNr']-1) * info['moduleTransSpacing'] / 2
    Y_submodules = Y_modules + ids_trans_submodule * info['submoduleTransSpacing'] - (info['submoduleTransNr']-1) * info['submoduleTransSpacing'] / 2
    Y_crystals = Y_submodules +ids_trans_crystal * info['crystalTransSpacing'] - (info['crystalTransNr']-1) * info['crystalTransSpacing'] / 2
    X_crystals = info['radius'] * torch.ones(len(Y_crystals))
    # Now apply rotation based on angle of the rsector
    angle_rsector = ids_trans_rsector / info['rsectorTransNr'] * 2 * np.pi
    rotation_matrices = torch.stack([
        torch.cos(angle_rsector),
        -torch.sin(angle_rsector),
        torch.sin(angle_rsector),
        torch.cos(angle_rsector)], dim=1).view(-1, 2, 2)
    XY_crystals = torch.vstack([X_crystals, Y_crystals])
    XY_crystals = torch.einsum('ijk,ki->ij', rotation_matrices, XY_crystals)
    if info['firstCrystalAxis'] == 0: # crystal with index 0 along x axis
        XY_crystals = XY_crystals.flip(dims=(1,))
    # Stack all together (for some reason Z needs to be negative?)
    XYZ_crystals = torch.vstack([XY_crystals.T, -Z_crystals.unsqueeze(0)]).T
    # Now sort scanner LUT by ids_detector order
    XYZ_crystals = XYZ_crystals[torch.argsort(ids_detector)]
    
    return XYZ_crystals

def _lazy_sinogram_to_listmode(detector_ids: torch.Tensor, sinogram: LazySinogram, info: dict) -> torch.Tensor:
    """``sinogram_to_listmode`` of a :class:`LazySinogram`: the events are grouped by angle (``_events_by_angle``), and the sinogram is computed a group of angles at a time, so it is never held whole."""
    shape = _sinogram_shape(info)
    num_tof_bins = sinogram.shape[-1] if len(sinogram.shape) > 3 else None
    offsets, within_angle, tof_bin, _, event = _events_by_angle(detector_ids, info, num_tof_bins, drop_outside=False, event_index=True)
    values = torch.empty(detector_ids.shape[0], dtype=torch.float32)
    per_chunk = sinogram._angles_per_chunk()
    for first in range(0, shape[0], per_chunk):
        last = min(first + per_chunk, shape[0])
        lo, hi = int(offsets[first]), int(offsets[last])
        if lo == hi:
            continue
        part = sinogram[torch.arange(first, last)]
        k = within_angle[lo:hi].to(torch.long)
        angle = torch.repeat_interleave(torch.arange(last - first), offsets[first + 1:last + 1] - offsets[first:last])
        index = (angle, k // shape[2], k % shape[2])
        values[event[lo:hi]] = part[index + ((tof_bin[lo:hi].to(torch.long),) if num_tof_bins is not None else ())]
        del part
    return values

def sinogram_to_listmode(detector_ids: torch.Tensor, sinogram: torch.Tensor | LazySinogram, info: dict) -> torch.Tensor:
    """Obtains listmode data from a sinogram at the given detector IDs

    Args:
        detector_ids (torch.Tensor): Detector IDs at which to obtain listmode data
        sinogram (torch.Tensor | LazySinogram): PET sinogram. A :class:`LazySinogram` is computed a group of angles at a time, and the values are returned on the CPU.
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: Listmode data
    """
    if isinstance(sinogram, LazySinogram):
        return _lazy_sinogram_to_listmode(detector_ids, sinogram, info)
    return _dense_sinogram_to_listmode(detector_ids, sinogram, info)

def _dense_sinogram_to_listmode(detector_ids: torch.Tensor, sinogram: torch.Tensor, info: dict, events_per_chunk: int | None = None) -> torch.Tensor:
    """``sinogram_to_listmode`` of a sinogram held whole, a chunk of events at a time into one output: all 107 million events of the GATE mMR brain scan at once took about 75 bytes each in temporaries (8 GB, which Windows kept committed afterwards). Each event is looked up on its own, so the values do not depend on the chunks.

    Args:
        detector_ids (torch.Tensor): Detector IDs at which to obtain listmode data
        sinogram (torch.Tensor): PET sinogram
        info (dict): PET geometry information dictionary
        events_per_chunk (int | None, optional): Events looked up at once. Defaults to None: as many as fit in an eighth of the memory budget (about 80 bytes each), or 2**22 without a budget.

    Returns:
        torch.Tensor: Listmode data, on the device of ``sinogram``
    """
    # TODO: multiple IDs map to same sinogram bin -> need to divide by number of LORs mapping to each sinogram bin
    # Look the events up where the sinogram is (a list mode system matrix keeps its events' detector IDs on its lor_device)
    device = sinogram.device
    lor_coordinates, sinogram_index = (table.to(device) for table in sinogram_coordinates(info))
    if events_per_chunk is None:
        events_per_chunk = block_size(80, default=2**22)
    n_events = detector_ids.shape[0]
    lm_return = torch.empty(n_events, dtype=sinogram.dtype, device=device)
    for start in range(0, n_events, events_per_chunk):
        ids = detector_ids[start:start + events_per_chunk]
        end = start + ids.shape[0]
        detector_ids_spatial = ids[:,:2].to(device)
        within_ring_id = (detector_ids_spatial % info['NrCrystalsPerRing']).to(torch.long)
        ring_ids = (detector_ids_spatial // info['NrCrystalsPerRing']).to(torch.long)
        # Same bin as listmode_to_sinogram: crystals ordered by descending within-ring index, ring IDs reordered with them
        within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)   # stable: a pair with equal within-ring IDs keeps its order on any device
        ring_ids = ring_ids.gather(index=idx, dim=1)
        idx0, idx1 = lor_coordinates[within_ring_id[:,0], within_ring_id[:,1]].T
        idx2 = sinogram_index[ring_ids[:,0], ring_ids[:,1]]
        if len(sinogram.shape)>3: # If TOF
            idxTOF = ids[:,2].to(device)
            # the TOF bin of an event whose crystals were swapped is mirrored, as in listmode_to_sinogram
            idxTOF = torch.where(idx[:,0] == 1, sinogram.shape[-1] - 1 - idxTOF, idxTOF)
            lm_return[start:end] = sinogram[idx0, idx1, idx2, idxTOF]
        else:
            lm_return[start:end] = sinogram[idx0, idx1, idx2]
    return lm_return

def _convolve_last_axis(x: torch.Tensor, kernel: torch.nn.Conv1d) -> torch.Tensor:
    """``x`` convolved with a 1D ``kernel`` along its last axis, a block of rows at a time into one output. The convolution
    of all rows at once took about 13 times the input in temporaries on the CPU (21 GB for the randoms sinogram of the
    Siemens Biograph mMR); the blocks hold an eighth of the memory budget (:func:`pytomography.set_memory_budget`), or
    about 0.5 GB without one. Each row is convolved on its own, so the result does not depend on the blocks."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    bytes_per_row = 64 * x.shape[-1]   # 16 float32 temporaries per sample
    rows_per_slice = max(1, int(np.prod(x.shape[1:-1])))
    slices = max(1, block_size(bytes_per_row, default=max(1, int(5e8 // bytes_per_row))) // rows_per_slice)
    for start in range(0, x.shape[0], slices):
        block = x[start:start + slices]
        out[start:start + slices] = kernel(block.reshape(-1, 1, x.shape[-1])).reshape(block.shape)
    return out

@torch.no_grad()
def smooth_randoms_sinogram(
    sinogram_random: torch.Tensor,
    info: dict,
    sigma_r: float = 4,
    sigma_theta: float = 4,
    sigma_z: float = 4,
    kernel_size_r: int = 21,
    kernel_size_theta: int = 21,
    kernel_size_z: int = 21
    ) -> torch.Tensor:
    """Smooths a PET randoms sinogram using a Gaussian filter in the r, theta, and z direction. Rebins the sinogram into (r,theta,z1,z2) before blurring (same blurring applied to z1 and z2)

    Args:
        sinogram_random (torch.Tensor): PET sinogram of randoms
        info (dict): PET geometry information dictionary
        sigma_r (float, optional): Blurring (in pixel size) in r direction. Defaults to 4.
        sigma_theta (float, optional): Blurring (in pixel size) in r direction. Defaults to 4.
        sigma_z (float, optional): Blurring (in pixel size) in z direction. Defaults to 4.
        kernel_size_r (int, optional): Kernel size in r direction. Defaults to 21.
        kernel_size_theta (int, optional): Kernel size in theta direction. Defaults to 21.
        kernel_size_z (int, optional): Kernel size in z1/z2 diretions. Defaults to 21.

    Returns:
        torch.Tensor: Smoothed randoms sinogram
    """
    _, sinogram_index = sinogram_coordinates(info)
    sino = sinogram_random[:,:,sinogram_index]
    ktheta = get_1d_gaussian_kernel(sigma_theta, kernel_size_theta, 'circular')
    kr = get_1d_gaussian_kernel(sigma_r, kernel_size_r, 'replicate')
    kz = get_1d_gaussian_kernel(sigma_z, kernel_size_z, 'replicate')
    for i, k in enumerate([ktheta,kr,kz,kz]):
        sino = _convolve_last_axis(sino.swapaxes(i,3), k).swapaxes(i,3)
    ii = torch.argsort(sinogram_index.ravel())
    ix, iy = ii // sino.shape[-2], ii % sino.shape[-1]
    sinogram_random_interp = sino[:,:,ix,iy]
    return sinogram_random_interp

def randoms_sinogram_to_sinogramTOF(
    sinogram_random: torch.Tenor,
    tof_meta: PETTOFMeta,
    coincidence_timing_width: float,
) -> torch.Tensor:
    """Converts a non-TOF randoms sinogram to a TOF randoms sinogram.

    Args:
        sinogram_random (torch.Tenor): Randoms sinogram (non-TOF)
        tof_meta (PETTOFMeta): PET TOF metadata
        coincidence_timing_width (float): Coincidence timing width used for the acceptance of coincidence events

    Returns:
        torch.Tensor: Randoms sinogram (TOF)
    """
    sinogram_random *= tof_meta.bin_width / (2 * coincidence_timing_width * 0.3 / 2) # 
    return sinogram_random
    

