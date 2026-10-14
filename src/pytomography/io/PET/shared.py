"""PET sinograms: binning list mode events into sinograms, and looking events up in them, for any scanner's ``info``.

A sinogram of a scanner with ``N`` crystals per ring and ``R`` rings (``info['NrCrystalsPerRing']`` and
``info['NrRings']``) has shape ``[N / 2, N + 1, R**2]``: angles, radial bins, and one plane per ordered pair of rings. A
TOF sinogram has the TOF bins as a fourth dimension. An event is binned by the within-ring ids of its two crystals, the
crystal with the larger id first, and the plane of their two rings in the same order; the TOF bin of an event whose
crystals were swapped is mirrored. Every function here bins events as :func:`listmode_to_sinogram` does, so sinograms
and list mode values correspond bin for bin.

* :func:`listmode_to_sinogram` and :func:`sinogram_to_listmode`: events to a sinogram, and a sinogram's value at each
  event;
* :class:`LazySinogram`: a sinogram computed a group of angles at a time, for sinograms too large to hold (TOF);
* :func:`normalization_sinogram` and :func:`all_pairs_to_sinogram`: a weight for every crystal pair, binned;
* :func:`smooth_randoms_sinogram` and :func:`randoms_sinogram_to_sinogramTOF`: randoms estimates;
* :func:`remove_events_outside`: the events whose LOR crosses the reconstruction grid;
* :func:`sinogram_coordinates` and :func:`sinogram_to_spatial`: the lookup tables behind the binning.
"""
from __future__ import annotations
from collections.abc import Sequence
import functools
import torch
import numpy as np
import pytomography
from pytomography.utils import get_1d_gaussian_kernel
from pytomography.utils.memory import BLOCK_CAP_BYTES, block_size, prefer_lazy, tensor_nbytes
from .scanner import scanner_lut as scanner_lut_of

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
    """Returns the lookup tables that bin events: the bin of each pair of crystals, and the plane of each pair of rings.

    The tables are built by Python loops over all crystal pairs (seconds for a clinical scanner) and cached per
    ``info``: the tensors are shared between callers and must not be changed in place.

    Args:
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).

    Returns:
        Sequence[torch.Tensor]: int64 ``[N, N, 2]``, the (angle, radial bin) of each pair of within-ring crystal ids
        (``N`` crystals per ring), and int64 ``[R, R]``, the plane of each pair of ring ids (``R`` rings).
    """
    nr_sectors_trans, nr_sectors_axial, nr_modules_axial, nr_modules_trans, nr_crystals_trans, nr_crystals_axial = info['rsectorTransNr'], info['rsectorAxialNr'], info['moduleAxialNr'], info['moduleTransNr'], info['crystalTransNr'], info['crystalAxialNr']
    nr_rings = info['NrRings']
    nr_crystals_per_ring = info['NrCrystalsPerRing']

    # The sinogram holds every crystal pair, whatever info['min_rsector_difference'] is: that is an event filter, which
    # the readers apply (scanner.keep_rsector_difference). Leaving pairs out here, as before PyTomography 4, changed
    # the radial bins without changing sinogram_to_spatial's geometry, so no pair landed in the bin that holds it.
    min_sector_difference = 0
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
    """Returns the positions of the two crystals of every sinogram bin, in the object frame, in mm.

    The first crystal of a bin is the one :func:`listmode_to_sinogram` puts first (the larger within-ring id), and the
    first ring of a plane is that crystal's ring. Bins that no crystal pair falls in hold zeros. The tables are cached
    per ``info``, like those of :func:`sinogram_coordinates`: they must not be changed in place.

    Args:
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).

    Returns:
        Sequence[torch.Tensor]: float32 ``[N / 2, N + 1, 2, 2]``, the (x, y) of the two crystals of each (angle,
        radial bin), and float32 ``[R**2, 2]``, the z of the two rings of each plane (``N`` crystals per ring, ``R``
        rings).
    """
    # a NumPy array (the same float32 values): copying rows of a tensor into the NumPy arrays below made NumPy 2 warn
    # that Tensor.__array__ takes no copy argument, once per crystal pair
    scanner_lut = scanner_lut_of(info).numpy()
    nr_modules_trans, nr_crystals_trans = info['moduleTransNr'], info['crystalTransNr']
    nr_rings = info['NrRings']
    nr_crystals_per_ring = info['NrCrystalsPerRing']
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
    r"""A sinogram that is computed a group of angles at a time, when those angles are asked for, instead of being held.

    A sinogram with 21 TOF bins of the Siemens Biograph mMR as GATE models it (224 angles, 449 radial bins, 4096 ring
    pairs) takes 34.6 GB, so a TOF reconstruction that holds its data, its additive term and the scatter estimate as
    dense sinograms needs over 100 GB of memory. Reconstruction algorithms only ever use one subset of angles at a time
    (:meth:`PETSinogramSystemMatrix.get_projection_subset
    <pytomography.projectors.PET.pet_sinogram_system_matrix.PETSinogramSystemMatrix.get_projection_subset>` indexes
    the first dimension of the projections), and so does the scaling of the scatter estimate, so these sinograms never
    need to exist whole. A ``LazySinogram`` can be passed wherever a likelihood takes projections or an additive term:

    * ``sinogram[angles]`` (an int, a slice, a boolean mask, or a list or tensor of angle indices, in any order)
      computes those angles. Further indices are applied to each group of angles as it is computed, so
      ``sinogram[:, :, :64, 10].sum(dim=0)`` never holds the whole sinogram. After the angles, at most one array index
      is allowed, with a slice for the angles and no integer indices; other indices raise an ``IndexError``.
    * Arithmetic (``+``, ``-``, ``*``, ``/``) with numbers, with tensors whose first dimension is the same angles (such
      as a dense randoms or sensitivity sinogram, with as many dimensions: use ``unsqueeze`` to add trailing ones),
      and with other lazy sinograms gives a lazy sinogram, e.g. ``(randoms.unsqueeze(-1) + scatter) / sensitivity``.
    * :meth:`to_dense` computes the whole sinogram as one tensor.

    The angles are computed in groups of at most :attr:`chunk_bytes` bytes (and a sixteenth of the memory budget, if
    one is set with :func:`pytomography.set_memory_budget`) whenever more than one group is asked for at once. Values
    are float32 on the CPU; :meth:`compute_at` gives a group of angles on another device, computed there when the
    sinogram can be.

    Args:
        compute (Callable[[torch.Tensor], torch.Tensor]): Computes the sinogram at the given angles: it gets a 1D long
            tensor of angle indices on the CPU and returns a new tensor of shape ``[len(angles), *shape[1:]]``.
        shape (Sequence[int]): Shape of the whole sinogram; the first dimension is the angles.
        description (str, optional): What the sinogram holds, for its ``repr``. Defaults to ''.
        compute_on (Callable[[torch.Tensor, torch.device], torch.Tensor] | None, optional): Like ``compute``, but
            returns the angles as a new tensor on the device it is given, computed there. Defaults to None:
            :meth:`compute_at` computes on the CPU and copies.
        memory_bytes (float, optional): Host memory the sinogram keeps to compute itself (e.g. its events), in bytes,
            for memory estimates. Defaults to 0.

    Attributes:
        shape (torch.Size): Shape of the whole sinogram; the first dimension is the angles.
        description (str): What the sinogram holds, for its ``repr``.
        memory_bytes (float): Host memory the sinogram keeps to compute itself (its events, its interpolation samples,
            the tensors its arithmetic holds), in bytes. Memory estimates count this instead of the size of the whole
            sinogram.
        chunk_bytes (float): Largest group of angles computed at once when more than one group is asked for, in bytes:
            :data:`pytomography.utils.memory.BLOCK_CAP_BYTES` (0.5 GB). A class attribute.
        dtype (torch.dtype): ``torch.float32``, the type of the values.
        device (torch.device): The CPU, where the values are computed.
    """
    #: Largest group of angles computed at once when more than one group is asked for, in bytes: the cap of a block of
    #: temporaries (:data:`pytomography.utils.memory.BLOCK_CAP_BYTES`, 0.5 GB). Looking the 107 million events of the
    #: GATE mMR scan up in its lazy scatter estimate (21 TOF bins) in groups of 1 GB left 1.0 GB more committed on Windows.
    chunk_bytes = BLOCK_CAP_BYTES
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
        """Number of dimensions of the sinogram."""
        return len(self.shape)

    def dim(self) -> int:
        """Returns the number of dimensions of the sinogram, as ``torch.Tensor.dim`` does.

        Returns:
            int: Number of dimensions.
        """
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
        """Computes the sinogram at ``angles`` as a new tensor on ``device``, all the angles at once.

        A sinogram that can be computed on a device (binned list mode events, the interpolated scatter estimate, and
        arithmetic of those with tensors and numbers) is computed there, without passing through host memory; any
        other is computed on the CPU and copied.

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
        """Computes the whole sinogram as one tensor, a group of angles at a time.

        Returns:
            torch.Tensor: Sinogram of shape :attr:`shape`.
        """
        return self[:]

    def cpu(self) -> LazySinogram:
        """Returns the sinogram itself, which is already computed on the CPU.

        Returns:
            LazySinogram: This sinogram.
        """
        return self

    def to(self, *args, **kwargs) -> LazySinogram:
        """Returns the sinogram itself, if it is asked for on the device and type it already has (the CPU, float32).

        This lets code that moves its projections to the CPU or to float32 work with a lazy sinogram, which cannot be
        moved or converted to anything else.

        Args:
            *args: Arguments of ``torch.Tensor.to``: devices and dtypes are checked, other values are ignored.
            **kwargs: Keyword arguments of ``torch.Tensor.to``, checked in the same way.

        Returns:
            LazySinogram: This sinogram.

        Raises:
            TypeError: If a dtype other than ``torch.float32``, or a device other than the CPU, is asked for.
        """
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
            other_bytes = tensor_nbytes(other) if other.device.type == 'cpu' else 0
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
    """Bins list mode events into a sinogram, counting them or summing their weights.

    Each event is binned as described at the top of this module (:mod:`pytomography.io.PET.shared`); with TOF, events
    outside the TOF bins are dropped. The events are binned a chunk at a time into one float32 sinogram on the CPU: as
    many events at once as fit in an eighth of the memory budget (:func:`pytomography.set_memory_budget`), at most
    about 3.8 million (0.5 GB of temporaries), or about 4.2 million without a budget. The sinogram is the one binning
    all the events at once gives, bit for bit.

    Args:
        detector_ids (torch.Tensor): Events, ``[N, 2]`` detector ids, or ``[N, 3]`` with the TOF bin (with
            ``tof_meta``), on any device.
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).
        weights (torch.Tensor, optional): ``[N]`` weight of each event, summed in its bin. Defaults to None: the events
            are counted.
        normalization (bool, optional): Bin a normalisation sinogram: each event is binned with its two crystals in
            both orders, and the two sinograms are averaged. A normalisation sinogram has no TOF bins (the
            normalisation of a line of response is the same in all of them), so it is not available with
            ``tof_meta``. Defaults to False.
        tof_meta (PETTOFMeta, optional): The TOF bins of the events (the third column of ``detector_ids``); the
            sinogram then has a fourth dimension, the TOF bins. Defaults to None (no TOF).
        lazy (bool | None, optional): Return a :class:`LazySinogram`, which bins the events of the angles it is asked
            for when it is asked for them, instead of the whole sinogram. A TOF sinogram is large (34.6 GB with 21 TOF
            bins for the Siemens Biograph mMR), while a reconstruction reads one subset of angles at a time. Not
            available with ``normalization``. Defaults to None: lazy when the whole sinogram would take more than a
            quarter of the memory budget (:func:`pytomography.set_memory_budget`), or more than 8 GB without a budget.

    Returns:
        torch.Tensor | LazySinogram: float32 sinogram on the CPU, ``[N / 2, N + 1, R**2]`` (angles, radial bins,
        planes; ``N`` crystals per ring, ``R`` rings), or ``[N / 2, N + 1, R**2, T]`` with ``T`` TOF bins.

    Raises:
        ValueError: If ``normalization`` is given with ``tof_meta``.
        NotImplementedError: If ``lazy`` is True with ``normalization``.
    """
    if normalization and tof_meta is not None:
        raise ValueError("a normalization sinogram has no TOF bins (it applies to every one): bin it without tof_meta")
    if lazy is None:
        lazy = not normalization and prefer_lazy(4 * np.prod(_sinogram_shape(info)) * (1 if tof_meta is None else tof_meta.num_bins))
    if lazy:
        if normalization:
            raise NotImplementedError("a normalization sinogram cannot be lazy")
        return _listmode_to_lazy_sinogram(detector_ids, info, tof_meta=tof_meta, weights=weights)
    if tof_meta is not None: # if tof_meta is provided
        return _listmodeTOF_to_sinogramTOF(detector_ids, info, tof_meta, weights=weights)
    return _listmode_to_dense_sinogram(detector_ids, info, weights=weights, normalization=normalization)

BINNING_BYTES_PER_EVENT = 130
"""Host memory that binning takes per list mode event (or crystal pair), temporaries included, in bytes. It sizes the
chunks that :func:`listmode_to_sinogram` and :func:`all_pairs_to_sinogram` bin at once."""

def _events_per_binning_chunk() -> int:
    """Number of list mode events :func:`listmode_to_sinogram` bins at once: as many as fit in an eighth of the memory budget (:func:`pytomography.set_memory_budget`), at most 0.5 GB of temporaries (:data:`pytomography.utils.memory.BLOCK_CAP_BYTES`), at :data:`BINNING_BYTES_PER_EVENT` bytes each; 2**22 (about 0.5 GB) without a budget."""
    return block_size(BINNING_BYTES_PER_EVENT, default=2**22)

def _add_events(sinograms: Sequence[torch.Tensor], ids: torch.Tensor, weights: torch.Tensor | None, tables: tuple, info: dict, shape: tuple, num_tof_bins: int | None = None) -> None:
    """Adds one chunk of events into flat float32 sinograms on the CPU: the first sinogram with each event's two crystals in descending order of their within-ring IDs (as ``listmode_to_sinogram`` bins them), the second, if there is one, in the opposite order (a normalization sinogram bins both).

    The bins are found on the device of ``ids`` (with ``tables``, the lookup tables of ``sinogram_coordinates`` on that device) and added on the CPU with ``index_add_``, which adds the events of each bin in their order in the chunk, as ``torch.bincount`` does: chunks added one after the other give, bit for bit, the sinogram of all the events binned at once, counts and weighted sums alike, on any device. Events outside the sinogram (or, with ``num_tof_bins``, outside its TOF bins) are dropped.

    Args:
        sinograms (Sequence[torch.Tensor]): One or two flat float32 sinograms on the CPU, added to in place.
        ids (torch.Tensor): [N, 2] detector IDs of the events, or [N, 3] with the TOF bin as the third column.
        weights (torch.Tensor | None): [N] weight of each event, on any device; None counts the events.
        tables (tuple): ``sinogram_coordinates(info)`` on the device of ``ids``.
        info (dict): PET geometry information dictionary.
        shape (tuple): Sinogram shape (angles, radial bins, planes).
        num_tof_bins (int | None, optional): Number of TOF bins; the flat sinograms then have the TOF bin as their last (fastest) index. Defaults to None (non-TOF).
    """
    lor_coordinates, sinogram_index = tables
    within_ring_id = (ids[:,:2] % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (ids[:,:2] // info['NrCrystalsPerRing']).to(torch.long)
    # Need to bin by largest "within_ring_id" first (for use with the "ring_coordinates" function yielding spatial coordinates for each ID-pair at each sinogram coordinate)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)   # stable: a pair with equal within-ring IDs keeps its order on any device
    ring_ids = ring_ids.gather(index=idx, dim=1)
    if num_tof_bins is not None:
        # the TOF bin of an event whose crystals were swapped is mirrored
        tof_bin = ids[:,2].to(torch.long)
        tof_bin = torch.where(idx[:,0] == 1, num_tof_bins - 1 - tof_bin, tof_bin)
    del idx
    if weights is not None:
        weights = weights.to(device='cpu', dtype=torch.float32)
    for (a, b), sinogram in zip(((0, 1), (1, 0)), sinograms):
        key, inside = _bin_keys(lor_coordinates[within_ring_id[:,a], within_ring_id[:,b]], sinogram_index[ring_ids[:,a], ring_ids[:,b]], shape)
        if num_tof_bins is not None:
            inside &= (tof_bin >= 0) & (tof_bin < num_tof_bins)
            key = key * num_tof_bins + tof_bin
        key = key[inside].cpu()
        values = torch.ones(key.shape[0], dtype=torch.float32) if weights is None else weights[inside.cpu()]
        sinogram.index_add_(0, key, values)   # adds the events of each bin in their order, like bincount
        del key, inside, values

def _listmode_to_dense_sinogram(detector_ids: torch.Tensor, info: dict, weights: torch.Tensor | None = None, normalization: bool = False, num_tof_bins: int | None = None, events_per_chunk: int | None = None) -> torch.Tensor:
    """``listmode_to_sinogram`` as one sinogram tensor, binned a chunk of events at a time.

    Each chunk is added into one float32 sinogram on the CPU (two for a normalization sinogram), in the order of the events (``_add_events``), so the sinogram is the one binning all the events at once gives: counts are exact, and weighted sums are the same bit for bit, whatever the chunks. Binning all 117 million events of the GATE mMR brain scan at once took 9.6 GB of temporaries on top of the 1.65 GB sinogram on the CPU (10.6 GB with 3 TOF bins), and on Windows the process kept part of it committed after it was freed (2.2 GB, 4.8 GB with 3 TOF bins, and 12 GB after an earlier binning in the same process); a chunk at a time with a 16 GB budget, 0.3 GB. The bins of each chunk are found on the device the events are on (a list mode system matrix keeps them on the GPU), so the device holds one chunk's temporaries, not the sinogram.

    Args:
        detector_ids (torch.Tensor): [N, 2] detector IDs of the events, or [N, 3] with the TOF bin as the third column.
        info (dict): PET geometry information dictionary.
        weights (torch.Tensor | None, optional): [N] weight of each event. Defaults to None (count the events).
        normalization (bool, optional): Bin as a normalization sinogram (see ``listmode_to_sinogram``). Defaults to False.
        num_tof_bins (int | None, optional): Number of TOF bins; the sinogram then has a last TOF dimension. Defaults to None (non-TOF).
        events_per_chunk (int | None, optional): Events binned at once. Defaults to None: :func:`_events_per_binning_chunk` (from the memory budget).

    Returns:
        torch.Tensor: Sinogram on the CPU, [angles, radial bins, planes] or [angles, radial bins, planes, TOF bins].
    """
    tables = tuple(table.to(detector_ids.device) for table in sinogram_coordinates(info))
    shape = _sinogram_shape(info)
    if events_per_chunk is None:
        events_per_chunk = _events_per_binning_chunk()
    T = 1 if num_tof_bins is None else num_tof_bins
    # one sinogram per order of the two crystals that is binned: the normalization sinogram bins each event both ways
    sinograms = [torch.zeros(int(np.prod(shape)) * T, dtype=torch.float32) for _ in range(2 if normalization else 1)]
    for start in range(0, detector_ids.shape[0], events_per_chunk):
        ids = detector_ids[start:start + events_per_chunk]
        _add_events(sinograms, ids, None if weights is None else weights[start:start + ids.shape[0]], tables, info, shape, num_tof_bins)
        del ids
    sinogram = sinograms[0]
    # Opposite binning for normalization sinogram, which always considers "ring_id"s in order (this only works because of +/- z symmetry of normalization factors)
    if normalization:
        sinogram += sinograms[1]
        sinogram /= 2
    return sinogram.reshape(shape if num_tof_bins is None else (*shape, num_tof_bins))

def crystal_pair_blocks(n_crystals: int, pairs_per_block: int):
    """Yields every pair of crystals ``(i, j)``, ``i < j``, in ``torch.combinations`` order, a block of pairs at a time.

    The order is that of ``torch.combinations(torch.arange(n_crystals), 2)``. Each block holds whole rows of ``i``, at
    most about ``pairs_per_block`` pairs, so that the 411 million pairs of a clinical scanner never have to be held at
    once (3.3 GB as int32, and several times that in the temporaries of using them).

    Args:
        n_crystals (int): Number of crystals.
        pairs_per_block (int): Pairs per block (at least one row of ``i``, ``n_crystals - 1 - i`` pairs, is always
            taken).

    Yields:
        tuple[int, torch.Tensor]: Index of the block's first pair in ``torch.combinations`` order, and the block's
        pairs as an int64 ``[M, 2]`` tensor.
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

def all_pairs_to_sinogram(weights: torch.Tensor, info: dict, normalization: bool = False, pairs_per_chunk: int | None = None) -> torch.Tensor:
    """Bins a weight for every crystal pair of the scanner into a sinogram, a block of pairs at a time.

    This is :func:`listmode_to_sinogram` of every pair of crystals, with the pair's weight (such as a normalisation
    weight). The pairs are those of ``torch.combinations(torch.arange(N_crystals), 2)``, in that order, which is the
    order of ``weights``. For the 411 million pairs of the Siemens Biograph mMR, the pair list alone takes 3.3 GB and
    binning all pairs at once took tens of GB of temporaries. Here each block of pairs is added into one sinogram (two
    for a normalisation sinogram, one per order of the crystals) in the order :func:`listmode_to_sinogram` adds them,
    so the result is the same.

    Args:
        weights (torch.Tensor): Weight of each crystal pair, ``[N_crystals * (N_crystals - 1) / 2]``, in
            ``torch.combinations`` order.
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).
        normalization (bool, optional): Bin as a normalisation sinogram (see :func:`listmode_to_sinogram`). Defaults
            to False.
        pairs_per_chunk (int | None, optional): Number of pairs binned at once. Defaults to None: as many as fit in an
            eighth of the memory budget (:data:`BINNING_BYTES_PER_EVENT` bytes each, at most 0.5 GB;
            :func:`pytomography.set_memory_budget`), or 2**22 without a budget.

    Returns:
        torch.Tensor: float32 sinogram on the CPU, ``[N / 2, N + 1, R**2]`` (angles, radial bins, planes).

    Raises:
        ValueError: If ``weights`` does not have one weight per crystal pair.
    """
    n_crystals = int(info['NrCrystalsPerRing'] * info['NrRings'])
    if weights.shape[0] != n_crystals * (n_crystals - 1) // 2:
        raise ValueError(f"expected one weight per crystal pair ({n_crystals * (n_crystals - 1) // 2:,}), got {weights.shape[0]:,}")
    if pairs_per_chunk is None:
        pairs_per_chunk = block_size(BINNING_BYTES_PER_EVENT, default=2**22)
    tables = sinogram_coordinates(info)
    shape = _sinogram_shape(info)
    # one sinogram per order of the two crystals that is binned: the normalization sinogram bins each pair both ways
    sinograms = [torch.zeros(int(np.prod(shape)), dtype=torch.float32) for _ in range(2 if normalization else 1)]
    for offset, ids in crystal_pair_blocks(n_crystals, pairs_per_chunk):
        _add_events(sinograms, ids, weights[offset:offset + ids.shape[0]], tables, info, shape)   # as listmode_to_sinogram bins them
    sinogram = sinograms[0]
    if normalization:
        sinogram += sinograms[1]
        sinogram /= 2
    return sinogram.reshape(shape)

def _sinogram_shape(info: dict) -> tuple:
    """Shape (angles, radial bins, planes) of the sinogram of a scanner: one plane per ordered pair of rings, whatever
    parts (crystals, submodules, modules, rsectors) the rings are made of."""
    return (int(info['NrCrystalsPerRing']/2), int(info['NrCrystalsPerRing'])+1, int(info['NrRings']**2))

def _bin_keys(angular_radial: torch.Tensor, plane: torch.Tensor, shape: tuple) -> tuple:
    """Flat sinogram index of each event, and which events lie inside the sinogram (``torch.histogramdd``, used before, dropped the others; ``_add_events`` does too)."""
    inside = (angular_radial[:,0] >= 0) & (angular_radial[:,0] < shape[0]) & (angular_radial[:,1] >= 0) & (angular_radial[:,1] < shape[1]) & (plane >= 0) & (plane < shape[2])
    return (angular_radial[:,0] * shape[1] + angular_radial[:,1]) * shape[2] + plane, inside

def _listmodeTOF_to_sinogramTOF(
    detector_ids: torch.Tensor,
    info: dict,
    tof_meta: PETTOFMeta,
    weights: torch.Tensor | None = None
    ) -> torch.Tensor:
    """Helper function to ``listmode_to_sinogram`` for TOF data: the events are binned a chunk at a time, as without TOF (``_listmode_to_dense_sinogram``). Events outside the TOF bins are dropped.

    Args:
        detector_ids (torch.Tensor): Listmode detector ID data
        info (dict): PET geometry information dictionary
        weights (torch.Tensor, optional): Binning weights for each listmode event. Defaults to None.
        tof_meta (PETTOFMeta, optional): PET TOF metadata. Defaults to None.

    Returns:
        torch.Tensor: PET TOF sinogram
    """
    return _listmode_to_dense_sinogram(detector_ids, info, weights=weights, num_tof_bins=int(tof_meta.num_bins))

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
    """Looks up the value of a sinogram at each list mode event, in the bin :func:`listmode_to_sinogram` bins it in.

    Every event must lie inside the sinogram (and, for a TOF sinogram, inside its TOF bins): none is dropped.

    Args:
        detector_ids (torch.Tensor): Events, ``[N, 2]`` detector ids, or ``[N, 3]`` with the TOF bin for a TOF
            sinogram.
        sinogram (torch.Tensor | LazySinogram): Sinogram of the scanner, ``[angles, radial bins, planes]``, or with a
            fourth dimension of TOF bins. A :class:`LazySinogram` is computed a group of angles at a time.
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).

    Returns:
        torch.Tensor: ``[N]`` value of the sinogram at each event: on the device of ``sinogram`` for a tensor, and
        float32 on the CPU for a :class:`LazySinogram`.
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
    about 0.5 GB without one. Each row is convolved on its own, so the result does not depend on the blocks. The
    kernel is applied on the device and in the dtype of ``x``."""
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    bytes_per_row = 16 * x.element_size() * x.shape[-1]   # 16 temporaries per sample
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
    """Smooths a randoms sinogram with Gaussian filters along the angle, the radial bin and the two rings of each plane.

    The planes are first arranged as ``[R, R]`` (ring 1, ring 2) with :func:`sinogram_coordinates`; the sinogram is
    then convolved with a 1D Gaussian along the angle (wrapping around), the radial bin and each of the two rings (the
    edges extended), a block of rows at a time, and its planes are put back in order.

    Args:
        sinogram_random (torch.Tensor): Randoms sinogram (non-TOF), ``[angles, radial bins, planes]``, on any device
            and in any floating point dtype (the result is on the same device, in the same dtype).
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).
        sigma_r (float, optional): Standard deviation of the Gaussian along the radial bins, in bins. Defaults to 4.
        sigma_theta (float, optional): Standard deviation of the Gaussian along the angles, in angles. Defaults to 4.
        sigma_z (float, optional): Standard deviation of the Gaussian along each ring of a plane, in rings. Defaults to
            4.
        kernel_size_r (int, optional): Length of the radial kernel, in bins. Defaults to 21.
        kernel_size_theta (int, optional): Length of the angular kernel, in angles. Defaults to 21.
        kernel_size_z (int, optional): Length of the kernel along each ring, in rings. Defaults to 21.

    Returns:
        torch.Tensor: The smoothed randoms sinogram, of the shape of ``sinogram_random``.
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

SPEED_OF_LIGHT_MM_PER_PS = 0.3
"""The speed of light that the PET readers and corrections use, in mm/ps (c is 0.2998 mm/ps). The TOF position of an
event along its LOR, from the LOR's midpoint, is (t1 - t2) c / 2."""

def randoms_sinogram_to_sinogramTOF(
    sinogram_random: torch.Tensor,
    tof_meta: PETTOFMeta,
    coincidence_timing_width: float,
) -> torch.Tensor:
    """Returns the randoms in each TOF bin, from a non-TOF randoms sinogram.

    Randoms are spread evenly over the coincidence window: their TOF positions (t1 - t2) c / 2 fill an interval
    ``2 * coincidence_timing_width * c / 2`` long, so each TOF bin holds the fraction
    ``tof_meta.bin_width / (2 * coincidence_timing_width * c / 2)`` of them, with c =
    :data:`SPEED_OF_LIGHT_MM_PER_PS`.

    Args:
        sinogram_random (torch.Tensor): Randoms sinogram (non-TOF); it is not changed.
        tof_meta (PETTOFMeta): The TOF bins (their width ``tof_meta.bin_width`` is in mm).
        coincidence_timing_width (float): Coincidence timing width used to accept coincidence events, in ps.

    Returns:
        torch.Tensor: A new tensor of the shape of ``sinogram_random``: the randoms in one TOF bin, which is the same
        for every bin (add a last dimension with ``unsqueeze(-1)`` to broadcast it over the TOF bins). Before
        PyTomography 4 the input itself was scaled and returned: use the return value.
    """
    return sinogram_random * (tof_meta.bin_width / (2 * coincidence_timing_width * SPEED_OF_LIGHT_MM_PER_PS / 2))

def normalization_sinogram(weights: torch.Tensor, info: dict) -> torch.Tensor:
    """Bins the normalisation weight of every crystal pair into the scanner's normalisation sinogram.

    The weights are binned as :func:`listmode_to_sinogram` bins events, with both orders of each pair's crystals
    averaged (``normalization=True``), a block of pairs at a time (:func:`all_pairs_to_sinogram`).

    Args:
        weights (torch.Tensor): Weight of each crystal pair, ``[N_crystals * (N_crystals - 1) / 2]``, in
            ``torch.combinations`` order: the ``weights_sensitivity`` a reader returns, or
            :func:`~pytomography.io.PET.gate_normalization`.
        info (dict): The scanner's ``info`` (:func:`~pytomography.io.PET.scanner_info`).

    Returns:
        torch.Tensor: float32 normalisation sinogram on the CPU, ``[N / 2, N + 1, R**2]`` (angles, radial bins,
        planes).

    Raises:
        ValueError: If ``weights`` does not have one weight per crystal pair.
    """
    return all_pairs_to_sinogram(weights, info, normalization=True)

def remove_events_outside(events: torch.Tensor, scanner_lut, object_meta) -> torch.Tensor:
    """Returns the events whose LOR crosses the reconstruction grid of ``object_meta``.

    The grid is the box of its voxels, centred on the scanner's centre (the origin of the scanner lookup table, where
    every reader centres its grid). Events whose LOR misses the box project to zero, so leaving them out changes no
    reconstruction; it saves their memory and time. The test is the slab test of a line segment (from column 0's
    crystal to column 1's) against an axis-aligned box: the segment crosses the box where its intervals inside the x,
    y and z slabs overlap. Before PyTomography 4, only some of the slabs were compared, and LORs that miss the box
    were kept.

    Args:
        events (torch.Tensor): Events, ``[N, 2]`` detector ids, or ``[N, 3]`` with the TOF bin.
        scanner_lut (torch.Tensor | dict): The scanner lookup table, ``[N_crystals, 3]`` in mm
            (:func:`~pytomography.io.PET.scanner_lut`), or the scanner's ``info``.
        object_meta (ObjectMeta): The reconstruction grid: its ``dr`` (mm) and ``shape`` are used.

    Returns:
        torch.Tensor: The events whose LOR crosses the grid, in their order.
    """
    if isinstance(scanner_lut, dict):
        scanner_lut = scanner_lut_of(scanner_lut)
    lut = scanner_lut.to(device=events.device, dtype=torch.float64)
    half = torch.tensor(np.asarray(object_meta.shape, dtype=np.float64) * np.asarray(object_meta.dr, dtype=np.float64) / 2,
                        dtype=torch.float64, device=events.device)
    keep = torch.empty(events.shape[0], dtype=torch.bool, device=events.device)
    for start in range(0, events.shape[0], 2**22):
        ids = events[start:start + 2**22, :2].to(torch.long)
        origin = lut[ids[:, 0]]
        direction = lut[ids[:, 1]] - origin
        t1 = (-half - origin) / direction
        t2 = (half - origin) / direction
        enter = torch.minimum(t1, t2)
        leave = torch.maximum(t1, t2)
        # a segment parallel to a slab is inside it everywhere or nowhere
        parallel = direction == 0
        inside_slab = origin.abs() < half
        enter = torch.where(parallel, torch.where(inside_slab, -torch.inf, torch.inf), enter)
        leave = torch.where(parallel, torch.where(inside_slab, torch.inf, -torch.inf), leave)
        t_in = torch.clamp(enter.max(dim=1).values, min=0)
        t_out = torch.clamp(leave.min(dim=1).values, max=1)
        keep[start:start + ids.shape[0]] = t_in < t_out
    return events[keep]
