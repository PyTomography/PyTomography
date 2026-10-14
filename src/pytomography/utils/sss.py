from __future__ import annotations
from typing import Sequence
import math
import torch
import pytomography
from pytomography.io.PET import shared
from pytomography.projectors.PET import PETLMSystemMatrix
import numpy as np
import parallelproj_core
from torchrbf import RBFInterpolator
from torch.nn.functional import grid_sample
from pytomography.io.PET.shared import sinogram_coordinates, sinogram_to_spatial, listmode_to_sinogram, LazySinogram
from pytomography.utils.memory import gpu_budget, prefer_lazy, subsets_for_budget, MemoryEstimate, MemoryPart, MemoryPlan, memory_estimate, nbytes, plan_settings
from pytomography.utils import _sss_cuda
from pytomography.projectors.PET import create_sinogramSM_from_LMSM
from pytomography.projectors.PET.petlm_system_matrix import _float32
from pytomography.metadata.PET import PETTOFMeta
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix

#: Scatter points whose emission and transmission integrals the non-TOF scatter loop computes in one projector call.
#: parallelproj 2 synchronises the device at the end of every call, which stalls the CPU queueing the loop's kernels:
#: with a call per scatter point the GATE mMR estimate (7233 scatter points) took 19.7 s on an RTX 5090, against
#: 10.3 s with 32 per call. The result does not depend on it.
_POINTS_PER_PROJECTION = 32

def total_compton_cross_section(energy: torch.Tensor) -> torch.Tensor:
    r"""Computes the total compton cross section of interaction :math:`\sigma` at the given photon energies

    Args:
        energy (torch.Tensor): Energies of photons considered

    Returns:
        torch.Tensor: Cross section at each corresponding energy
    """
    a = energy / 511
    l = torch.log(1+2*a)
    sigma0 = 6.65e-25
    return 0.75 * sigma0 * ((1+a)/a**2 * (2*(1+a) / (1+2*a) - l/a) + l/(2*a) - (1+3*a) / (1+2*a) / (1+2*a))

def photon_energy_after_compton_scatter_511kev(cos_theta: torch.Tensor) -> torch.Tensor:
    """Computes the corresponding photon energy after a 511keV photon scatters 

    Args:
        cos_theta (torch.Tensor): Angle of scatter

    Returns:
        torch.Tensor: Photon energy after scattering.
    """
    return 511 / (2 - cos_theta)

def diff_compton_cross_section(cos_theta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    r"""Computes the differential cross section :math:`d\sigma/d\omega` at given photon energies and scattering angles

    Args:
        cos_theta (torch.Tensor): Cosine of the scattering angle
        energy (torch.Tensor): Energy of the incident photon before scattering

    Returns:
        torch.Tensor: Differential compton cross section
    """
    Re = 2.818e-13
    sin_theta_2 = 1- cos_theta**2
    P = 1 / (1+energy/511 * (1-cos_theta))
    return Re**2 / 2 * P * (1-P * sin_theta_2 + P**2)

#: The detector's energy window and resolution when the scanner's ``info`` does not give them (see
#: :func:`detector_energy_window`): a lower threshold of 430 keV, no upper limit, and a resolution of 15% FWHM at
#: 511 keV. These are the values the scatter simulation always used, so data whose ``info`` has none of the keys (GATE,
#: the GE scanners) get the same estimate as before, bit for bit.
DEFAULT_ENERGY_WINDOW = {'energy_window_low_keV': 430.0, 'energy_window_high_keV': None, 'energy_resolution': 0.15}

def detector_energy_window(info: dict) -> tuple[float, float | None, float]:
    """The detector's energy window and energy resolution, as the scatter simulation reads them from the scanner's
    ``info``:

    * ``info['energy_window_low_keV']``: the lower limit of the energy window, in keV;
    * ``info['energy_window_high_keV']``: its upper limit, in keV;
    * ``info['energy_resolution']``: the FWHM of the photopeak at 511 keV, as a fraction of 511 keV (PETSIRD's
      ``energy_resolution_at_511``).

    The PETSIRD and GE readers give them from the file; the GATE reader from its arguments. The
    GATE and GE readers give none of them; set them in ``info`` to use a scanner's own values. Each key that is absent
    (or None) takes its default on its own (:data:`DEFAULT_ENERGY_WINDOW`): 430 keV, no upper limit, 0.15.

    Args:
        info (dict): PET geometry information dictionary.

    Raises:
        ValueError: if the resolution is not between 0 and 1, or the upper limit is not above the lower one.

    Returns:
        tuple[float, float | None, float]: lower limit (keV), upper limit (keV, or None for no upper limit), resolution.
    """
    def value(key):
        v = info.get(key)
        v = DEFAULT_ENERGY_WINDOW[key] if v is None else v
        return None if v is None else float(v)
    low, high, resolution = value('energy_window_low_keV'), value('energy_window_high_keV'), value('energy_resolution')
    if not 0 < resolution < 1:
        raise ValueError(f"info['energy_resolution'] is the FWHM at 511 keV as a fraction of 511 keV (0.15 for 15%), "
                         f"between 0 and 1; it is {resolution}. Set it to the scanner's value, or remove it to use 0.15")
    if high is not None and not high > low:
        raise ValueError(f"the energy window's upper limit (info['energy_window_high_keV'] = {high} keV) must be above its "
                         f"lower limit ({low} keV)")
    return low, high, resolution

def detector_efficiency(
    scatter_energy: torch.Tensor,
    energy_resolution: float = 0.15,
    energy_threshhold: float = 430,
    energy_window_high: float | None = None
    ) -> torch.Tensor:
    r"""Computes the probability a photon of given energy is detected within the energy window of the detector.

    The measured energy of a photon of energy :math:`E` is Gaussian about :math:`E`, with a FWHM of
    ``energy_resolution`` :math:`\times` 511 keV at every energy (the resolution is not scaled with the energy), so
    :math:`\sigma = 511\,\mathrm{keV} \times` ``energy_resolution`` :math:`/ (2\sqrt{2\ln 2})`. The probability that it
    falls in the window :math:`(E_\mathrm{low}, E_\mathrm{high})` is

    .. math::

        P = \frac{1}{2}\left[\mathrm{erf}\left(\frac{E_\mathrm{high} - E}{\sqrt{2}\sigma}\right)
            - \mathrm{erf}\left(\frac{E_\mathrm{low} - E}{\sqrt{2}\sigma}\right)\right],

    where the first term is 1 without an upper limit. The scatter simulation takes the three values from the scanner's
    ``info`` (:func:`detector_energy_window`).

    Args:
        scatter_energy (torch.Tensor): Energy of the photon impinging the detector, in keV
        energy_resolution (float, optional): Energy resolution of the crystals: the FWHM at 511 keV, as a fraction of 511 keV. This is the uncertainty of energy measurements. Defaults to 0.15.
        energy_threshhold (float, optional): Lower limit of the energy window, in keV: the lowest energy registered as an event. Defaults to 430.
        energy_window_high (float | None, optional): Upper limit of the energy window, in keV. Defaults to None: no upper limit.

    Returns:
        torch.Tensor: Probability that the photon gets detected
    """
    sigma = 511 * energy_resolution / (2*np.sqrt(2*np.log(2)))
    erf_low = torch.erf((energy_threshhold-scatter_energy) / (np.sqrt(2) * sigma))
    if energy_window_high is None or energy_window_high == math.inf:
        return 0.5 * (1 - erf_low)
    return 0.5 * (torch.erf((energy_window_high-scatter_energy) / (np.sqrt(2) * sigma)) - erf_low)

def _tof_efficiency(offset: torch.Tensor, tof_bins_dense_centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """``tof_efficiency`` with the TOF resolution given as a Python float (so no device-to-host transfer per call)."""
    prob =  torch.exp(-(offset.unsqueeze(-1)-tof_bins_dense_centers.unsqueeze(0))**2 / (2*sigma**2))
    prob = prob / prob.sum(dim=0).unsqueeze(0)
    return prob

def _tof_weighted_emission(offset: torch.Tensor, centers: torch.Tensor, emission: torch.Tensor, sigma: float, tof_splits: list) -> torch.Tensor:
    r"""Emission along each sampled LOR as seen by each TOF bin: :math:`\sum_k w_{tlk} e_{lk}`, where :math:`e_{lk}` is the emission integral of dense piece :math:`k` of LOR :math:`l`, and :math:`w_{tlk}` the Gaussian TOF kernel between that piece and TOF bin :math:`t`, normalised over all the TOF bins.

    The kernel is a [LORs, TOF bins, pieces] tensor; ``tof_splits`` bounds its size by computing it for a group of TOF bins at a time. Its normalisation runs over every TOF bin, so with more than one group the kernel is computed twice: once to sum it, once to apply it. The normalisation is applied to the emission (one value per piece) instead of to the kernel, and the sum over pieces is a batched matrix product, which reads the kernel once.

    Args:
        offset (torch.Tensor): [TOF bins, LORs] TOF offsets of the LORs.
        centers (torch.Tensor): [LORs, pieces] distance of each piece's centre along its LOR.
        emission (torch.Tensor): [LORs, pieces] emission integrals of the pieces.
        sigma (float): TOF resolution, as a standard deviation in spatial units.
        tof_splits (list): (start, end) ranges of the TOF bins computed together.

    Returns:
        torch.Tensor: [TOF bins, LORs] emission seen by each TOF bin.
    """
    def kernel(start, end):
        k = offset[start:end].T.unsqueeze(-1) - centers.unsqueeze(1)     # [LORs, bins, pieces]
        return k.square_().mul_(-0.5 / sigma**2).exp_()
    if len(tof_splits) == 1:
        k = kernel(*tof_splits[0])
        norm = k.sum(dim=1)
    else:
        k = None
        norm = sum(kernel(start, end).sum(dim=1) for start, end in tof_splits)
    # a piece far from every TOF bin has a kernel of zero in all of them: it contributes nothing (rather than 0/0)
    weighted = torch.where(norm > 0, emission / norm, torch.zeros_like(emission)).unsqueeze(-1)      # [LORs, pieces, 1]
    if k is not None:
        return torch.bmm(k, weighted).squeeze(-1).T
    return torch.cat([torch.bmm(kernel(start, end), weighted).squeeze(-1) for start, end in tof_splits], dim=1).T

def _tof_weighted_emission_of_crystals(offset: torch.Tensor, crystal: torch.Tensor, centers: torch.Tensor, emission: torch.Tensor, sigma: float, tof_splits: list) -> torch.Tensor:
    """:func:`_tof_weighted_emission` of LORs whose pieces are those of the line from the scatter point to one of their
    crystals: ``crystal`` ([LORs]) is that crystal's row in the per-crystal tables ``centers`` and ``emission``
    ([crystals, pieces]). On a CUDA device with CuPy installed it runs as one fused kernel (``_sss_cuda``), which does not
    build the [LORs, TOF bins, pieces] kernel tensor; otherwise with PyTorch, ``tof_splits`` bounding that tensor."""
    if _sss_cuda.available(offset.device, offset.shape[0]):
        return _sss_cuda.tof_weighted_emission(offset, crystal, centers, emission, sigma)
    return _tof_weighted_emission(offset, centers[crystal], emission[crystal], sigma, tof_splits)

def tof_efficiency(
    offset: torch.Tensor,
    tof_bins_dense_centers: torch.Tensor,
    tof_meta: PETTOFMeta
    ) -> torch.Tensor:
    """Computes the probability that a coincidence event with timing difference offset is detected in each of the TOF bins specified by ``tof_bins_dense_centers``.

    Args:
        offset (torch.Tensor): Timing offset (in spatial units) between a coincidence event. When this function is used in SSS, ``offset`` has shape :math:`(N_{TOF}, N_{coinc})` where :math:`N_{coinc}` is the number of coincidence events considered, and :math:`N_{TOF}` is the number of time of flight bins in the sinogram.
        tof_bins_dense_centers (torch.Tensor): The centers of each of the dense TOF bins. These are seperate from the TOF bins of the sinogram: these TOF bins correspond to the partioning of the integrals in Watson(2007) Equation 2. When used in SSS, this tensor has shape :math:`(N_{coinc}, N_{denseTOF})` where :math:`N_{denseTOF}` are the number of dense TOF bins considered.
        tof_meta (PETTOFMeta): TOF metadata for the sinogram

    Returns:
        torch.Tensor: Relative probability of detecting the event at offset ``offset`` in each of the ``tof_bins_dense_centers`` locations.
    """
    return _tof_efficiency(offset, tof_bins_dense_centers, tof_meta.sigma.item())

class SparseSinogram:
    """Scatter estimate at the sampled LORs only, stored as a flat table of sinogram bins and weights instead of a dense sinogram. A dense sinogram of a clinical scanner is large (1.65 GB for 224x449x4096 bins, 35 GB with 21 TOF bins) and binning it with ``torch.histogramdd`` allocates one copy per CPU thread, while the interpolation only ever reads the sampled bins. The bin of each LOR is the one ``listmode_to_sinogram`` would put it in.

    Args:
        detector_ids (torch.Tensor): [N, 2] detector ID pairs of the sampled LORs.
        weights (torch.Tensor): [N] (non-TOF) or [N_TOF, N] (TOF) scatter probabilities.
        info (dict): PET geometry information dictionary.
        tof_meta (PETTOFMeta | None, optional): TOF metadata when ``weights`` has a TOF dimension. Defaults to None.
    """
    def __init__(self, detector_ids: torch.Tensor, weights: torch.Tensor, info: dict, tof_meta: PETTOFMeta | None = None):
        self.info = info
        self.tof_meta = tof_meta
        self.shape = (int(info['NrCrystalsPerRing']/2), int(info['NrCrystalsPerRing'])+1, int(info['NrRings']**2))
        lor_coordinates, sinogram_index = sinogram_coordinates(info)
        ids = detector_ids[:, :2].to(torch.long).cpu()
        within_ring_id = ids % info['NrCrystalsPerRing']
        ring_ids = ids // info['NrCrystalsPerRing']
        within_ring_id, order = within_ring_id.sort(dim=1, descending=True)
        ring_ids = ring_ids.gather(index=order, dim=1)
        theta_r = lor_coordinates[within_ring_id[:, 0], within_ring_id[:, 1]]
        plane = sinogram_index[ring_ids[:, 0], ring_ids[:, 1]]
        keys = (theta_r[:, 0] * self.shape[1] + theta_r[:, 1]) * self.shape[2] + plane
        keys, sort = keys.sort()
        device = weights.device
        self.keys = keys.to(device)
        self.weights = weights[..., sort.to(device)]
        self.flipped = (order[:, 0] == 1)[sort].to(device)     # detector order was swapped: TOF bins are mirrored, as in listmode_to_sinogram

    def gather(self, theta: torch.Tensor, r: torch.Tensor, plane: torch.Tensor, tof_bin: int | None = None) -> torch.Tensor:
        """Weights at sinogram bins ``(theta, r, plane)`` (broadcastable index tensors); bins that were not sampled give 0.

        Args:
            theta (torch.Tensor): Angular bin indices.
            r (torch.Tensor): Radial bin indices.
            plane (torch.Tensor): Sinogram plane (ring pair) indices.
            tof_bin (int | None, optional): TOF bin of the sinogram (required for a TOF estimate). Defaults to None.

        Returns:
            torch.Tensor: Weights, broadcast shape of the index tensors.
        """
        key = (theta.to(torch.long) * self.shape[1] + r.to(torch.long)) * self.shape[2] + plane.to(torch.long)
        shape = key.shape
        key = key.flatten().to(self.keys.device)
        pos = torch.searchsorted(self.keys, key).clamp_(max=self.keys.shape[0]-1)
        found = self.keys[pos] == key
        if tof_bin is None:
            values = self.weights[pos]
        else:
            row = torch.where(self.flipped[pos], self.tof_meta.num_bins - 1 - tof_bin, tof_bin)
            values = self.weights[row, pos]
        return torch.where(found, values, torch.zeros((), dtype=values.dtype, device=values.device)).reshape(shape)

    def to_dense(self) -> torch.Tensor:
        """Dense sinogram [theta, r, plane(, TOF)] on the CPU (a single allocation), for code that expects a sinogram tensor.

        Returns:
            torch.Tensor: Dense sinogram.
        """
        T = 1 if self.tof_meta is None else self.tof_meta.num_bins
        sinogram = torch.zeros((*self.shape, T), dtype=torch.float32)
        keys = self.keys.cpu()
        theta, r, plane = keys // (self.shape[1]*self.shape[2]), (keys // self.shape[2]) % self.shape[1], keys % self.shape[2]
        for t in range(T):
            values = self.weights if self.tof_meta is None else torch.where(self.flipped, self.weights[self.tof_meta.num_bins - 1 - t], self.weights[t])
            sinogram[theta, r, plane, t] = values.cpu().to(torch.float32)
        return sinogram if self.tof_meta is not None else sinogram[..., 0]

def get_sample_scatter_points(
    attenuation_map: torch.Tensor,
    stepsize: float = 4,
    attenuation_cutoff: float = 0.004
    ) -> torch.Tensor:
    """Selects a subset of points in the attenuation map used as scatter points. 

    Args:
        attenuation_map (torch.Tensor): Attenuation map
        stepsize (float, optional): Stepsize in x/y/z between sampled points. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider points above this threshhold. Defaults to 0.004.

    Returns:
        torch.Tensor: Tensor of coordinates
    """
    mgrid = torch.meshgrid(*[torch.arange(0,s,stepsize) for s in attenuation_map.shape])
    coords = torch.vstack([m.flatten() for m in mgrid])
    idx_above_cutoff = (attenuation_map[::stepsize,::stepsize,::stepsize].permute((2,1,0)).cpu().numpy().T>attenuation_cutoff).flatten()
    coords = coords[:,idx_above_cutoff]
    return coords.to(pytomography.device)

def get_sample_detector_ids(
    proj_meta: ProjMeta,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4
    ) -> Sequence[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selects a subset of detector IDs in the PET scanner used for obtaining scatter estimates in the sparse sinogram

    Args:
        proj_meta (ProjMeta): PET projection metadata (sinogram/listmode)
        sinogram_interring_stepsize (int, optional): Axial stepsize between rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize of crystals within a given ring. Defaults to 4.

    Returns:
        Sequence[torch.Tensor, torch.Tensor, torch.Tensor]: Crystal index within ring, ring index, and detector ID pairs corresponding to all sampled LORs.
    """
    idx_intraring = torch.arange(0, proj_meta.info['NrCrystalsPerRing'], sinogram_intraring_stepsize)
    idx_ring = torch.arange(0, proj_meta.info['NrRings'], sinogram_interring_stepsize)
    # Include the top ring for interpolation to not have to extrapolate
    if not(proj_meta.info['NrRings']-1 in idx_ring):
        idx_ring = torch.cat((idx_ring, torch.tensor([proj_meta.info['NrRings']-1])))
    idx = torch.cartesian_prod(idx_ring, idx_intraring).T
    idx = idx[1] + idx[0]*proj_meta.info['NrCrystalsPerRing']
    return idx_intraring, idx_ring, torch.combinations(idx.cpu(), 2)
    
def _line_integrals(lor_start: torch.Tensor, lor_end: torch.Tensor, image: torch.Tensor, object_origin: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
    """Integrals of ``image`` along the segments from ``lor_start`` to ``lor_end`` ([N, 3] each). parallelproj 2 takes contiguous float32 arrays on the image's device and writes the integrals into a buffer the caller supplies."""
    device = image.device
    integrals = torch.zeros(lor_start.shape[0], dtype=torch.float32, device=device)
    parallelproj_core.joseph3d_fwd(_float32(lor_start, device), _float32(lor_end, device), image, object_origin, voxel_size, integrals)
    return integrals

def _integrals_from_points(points: torch.Tensor, detectors: torch.Tensor, image: torch.Tensor, object_origin: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
    """Integrals of ``image`` from each of ``points`` ([K, 3]) to each of ``detectors`` ([N, 3]), as a [K, N] tensor, in one projector call."""
    K, N = points.shape[0], detectors.shape[0]
    lor_start = points.unsqueeze(1).expand(K, N, 3).reshape(-1, 3)
    lor_end = detectors.unsqueeze(0).expand(K, N, 3).reshape(-1, 3)
    return _line_integrals(lor_start, lor_end, image, object_origin, voxel_size).reshape(K, N)

def _scatter_setup(object_meta, proj_meta, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize):
    """Everything the scatter-point loops need, computed once on the device: the loops themselves must not synchronise with the host (indexing a device tensor with a 0-d device tensor, ``.tolist()``/``.item()`` of one, or copying a host tensor to the device all stall the CPU until the GPU queue has drained, which made the loops CPU-bound)."""
    device = pytomography.device
    dr = torch.tensor(object_meta.dr)
    shape = torch.tensor(object_meta.shape)
    coords = get_sample_scatter_points(attenuation_image, stepsize=image_stepsize, attenuation_cutoff=attenuation_cutoff)
    coords_position = (coords - shape.unsqueeze(1).to(device)/2 + 0.5) * dr.unsqueeze(1).to(device)
    N_points = coords.shape[1]
    # random offset of each scatter point within its voxel (drawn on the host RNG as before, moved to the device once)
    positions = coords_position + ((torch.rand(N_points, 3) - 0.5) * dr).to(device).T
    # attenuation coefficient at each scatter point
    mu_values = attenuation_image.to(device)[coords[0], coords[1], coords[2]]
    _, _, detector_ids_scatter = get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    # The integrals from a scatter point are needed only to the crystals of the sampled LORs (900 of the mMR's 28,672
    # with every sixth ring and crystal), so only those crystals are kept, and the crystals of each LOR become indices
    # into them. (The integrals to every crystal of the scanner were computed before: 32 times the projector work.)
    sampled_crystals, local_ids = torch.unique(detector_ids_scatter, return_inverse=True)
    scanner_LUT = proj_meta.scanner_lut.to(device)[sampled_crystals.to(device)]
    idxA, idxB = local_ids.to(device).T
    rA = scanner_LUT[idxA]
    rB = scanner_LUT[idxB]
    return positions, mu_values, detector_ids_scatter, scanner_LUT, idxA, idxB, rA, rB

def compute_sss_sparse_sinogram(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    pet_image: torch.Tensor,
    attenuation_image: torch.Tensor,
    image_stepsize: int = 4,
    attenuation_cutoff: float = 0.004,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4
    ) -> SparseSinogram:
    """Generates a sparse single scatter simulation sinogram for non-TOF PET data.

    The detector's energy window and resolution, which give the probability that a scattered photon is detected, come
    from ``proj_meta.info`` (``energy_window_low_keV`` and ``energy_window_high_keV``, ``energy_resolution`` as the FWHM
    at 511 keV over 511 keV, as the readers give them; see :func:`detector_energy_window`). Each one that is
    absent takes its default: 430 keV, no upper limit, 0.15.

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to reconstructed PET image used in the simulation
        proj_meta (ProjMeta): Projection metadata specifying the details of the PET scanner, including its energy window (see above)
        pet_image (torch.Tensor): PET image used to estimate the scatter
        attenuation_image (torch.Tensor): Attenuation map used in scatter simulation
        image_stepsize (int, optional): Stepsize in x/y/z between sampled scatter points. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider points above this threshhold. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Axial stepsize between rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize of crystals within a given ring. Defaults to 4.

    Returns:
        SparseSinogram: Estimated single scatter simulation at the sampled LORs (``.to_dense()`` gives the sinogram the function used to return).
    """
    # Important quantities
    E_PET = torch.tensor(511).to(pytomography.device)
    # the geometry and images in the form every parallelproj kernel expects (float32, on the projection device)
    object_origin = _float32((- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr)), pytomography.device)
    voxel_size = _float32(np.array(object_meta.dr), pytomography.device)
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    voxel_volume = np.prod(object_meta.dr)
    energy_low, energy_high, energy_resolution = detector_energy_window(proj_meta.info)
    positions, mu_values, detector_ids_scatter, scanner_LUT, idxA, idxB, rA, rB = _scatter_setup(object_meta, proj_meta, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    attenuation_image = _float32(attenuation_image, pytomography.device)
    pet_image = _float32(pet_image, pytomography.device)
    rA_xy_norm = torch.norm(rA[:,:2], dim=1)
    rB_xy_norm = torch.norm(rB[:,:2], dim=1)
    N_points = positions.shape[1]
    # Loop over scatter points
    probability = 0
    for scatter_point in range(N_points):
        # Compute emission/transmission integrals, for a block of scatter points at once (see _POINTS_PER_PROJECTION)
        j = scatter_point % _POINTS_PER_PROJECTION
        if j == 0:
            block = positions[:, scatter_point:scatter_point+_POINTS_PER_PROJECTION].T
            emission_block = _integrals_from_points(block, scanner_LUT, pet_image, object_origin, voxel_size)
            transmission_exp_block = torch.exp(-_integrals_from_points(block, scanner_LUT, attenuation_image, object_origin, voxel_size))
        scatter_point_position = positions[:,scatter_point]
        mu_value = mu_values[scatter_point]
        emission_integrals = emission_block[j]
        transmission_integrals_exp = transmission_exp_block[j]
        # Compute scatter contribution
        rSA = rA - scatter_point_position
        rSB = rB - scatter_point_position
        rSA_norm = torch.norm(rSA, dim=1) # distance between S and A
        rSB_norm = torch.norm(rSB, dim=1) # distance between S and B
        # Compute cos(scattering_angle) = cos(pi-angle_between_vectors) = -cos(angle_between_vectors)
        cos_theta = - (rSA*rSB).sum(axis=1) / rSA_norm / rSB_norm
        E_new = photon_energy_after_compton_scatter_511kev(cos_theta)
        energy_efficiency = detector_efficiency(E_new, energy_resolution, energy_low, energy_high)
        # Angle of impingement upon detectors (assumes circle, maybe fix later)
        cos_thetaA_incidence = (rSA[:,:2]*rA[:,:2]).sum(axis=1) / rSA_norm / rA_xy_norm
        cos_thetaB_incidence = (rSB[:,:2]*rB[:,:2]).sum(axis=1) / rSB_norm / rB_xy_norm
        compton_cross_section_ratio = total_compton_cross_section(E_new) / total_compton_cross_section_511keV
        # Compute probability without considering TOF information
        probability_without_tof = 1/(rSB_norm**2 * rSA_norm**2) *\
        (emission_integrals[idxA] * transmission_integrals_exp[idxB] ** (compton_cross_section_ratio - 1) + emission_integrals[idxB] * transmission_integrals_exp[idxA] ** (compton_cross_section_ratio - 1)) *\
        transmission_integrals_exp[idxB] * transmission_integrals_exp[idxA] * mu_value * energy_efficiency * cos_thetaA_incidence * cos_thetaB_incidence * diff_compton_cross_section(cos_theta, E_PET) / total_compton_cross_section_511keV * voxel_volume
        probability += probability_without_tof
    return SparseSinogram(detector_ids_scatter, probability/N_points, proj_meta.info)

def compute_sss_sparse_sinogram_TOF(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    pet_image: torch.Tensor,
    attenuation_image: torch.Tensor,
    tof_meta: PETTOFMeta,
    image_stepsize: int = 4,
    attenuation_cutoff: float = 0.004,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4,
    num_dense_tof_bins: int = 25,
    N_splits: int = 1
    )->SparseSinogram:
    """Generates a sparse single scatter simulation sinogram for TOF PET data.

    The detector's energy window and resolution come from ``proj_meta.info``, as in
    :func:`compute_sss_sparse_sinogram` (defaults: 430 keV, no upper limit, 0.15).

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to reconstructed PET image used in the simulation
        proj_meta (ProjMeta): Projection metadata specifying the details of the PET scanner, including its energy window (see :func:`detector_energy_window`)
        pet_image (torch.Tensor): PET image used to estimate the scatter
        attenuation_image (torch.Tensor): Attenuation map used in scatter simulation
        tof_meta (PETTOFMeta): PET TOF Metadata corresponding to the sinogram estimate
        image_stepsize (int, optional): Stepsize in x/y/z between sampled scatter points. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider points above this threshhold. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Axial stepsize between rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize of crystals within a given ring. Defaults to 4.
        num_dense_tof_bins (int, optional): Number of dense TOF bins used when partioning the emission integrals (these integrals must be partioned for TOF-based estimation). Defaults to 25.
        N_splits (int, optional): Splits the TOF bins into subsets and loops over them sequentially (as opposed to parallel) to bound device memory. With more than one subset the TOF kernel is computed twice per scatter point, since its normalisation runs over all the TOF bins. Defaults to 1.

    Returns:
        SparseSinogram: Estimated single scatter simulation at the sampled LORs and TOF bins (``.to_dense()`` gives the sinogram the function used to return).
    """
    # Important quantities
    E_PET = torch.tensor(511).to(pytomography.device)
    # the geometry and images in the form every parallelproj kernel expects (float32, on the projection device)
    object_origin = _float32((- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr)), pytomography.device)
    voxel_size = _float32(np.array(object_meta.dr), pytomography.device)
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    voxel_volume = np.prod(object_meta.dr)
    energy_low, energy_high, energy_resolution = detector_energy_window(proj_meta.info)
    positions, mu_values, detector_ids_scatter, scanner_LUT, idxA, idxB, rA, rB = _scatter_setup(object_meta, proj_meta, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    attenuation_image = _float32(attenuation_image, pytomography.device)
    pet_image = _float32(pet_image, pytomography.device)
    rA_xy_norm = torch.norm(rA[:,:2], dim=1)
    rB_xy_norm = torch.norm(rB[:,:2], dim=1)
    N_points = positions.shape[1]
    N_detectors = scanner_LUT.shape[0]
    probability = torch.zeros([tof_meta.num_bins, detector_ids_scatter.shape[0]]).to(pytomography.device)
    tof_bin_positions = tof_meta.bin_positions.to(pytomography.device)
    sigma = tof_meta.sigma.item()
    bin_edges_scaling = torch.linspace(0,1,num_dense_tof_bins+1).to(pytomography.device)
    base, extra = divmod(tof_meta.num_bins, N_splits)
    tof_splits, start = [], 0
    for i in range(N_splits):
        tof_splits.append((start, start + base + (1 if i < extra else 0)))
        start = tof_splits[-1][1]
    # Loop over scatter points
    for scatter_point in range(N_points):
        scatter_point_position = positions[:,scatter_point]
        mu_value = mu_values[scatter_point]
        # Compute emission/transmission integrals for that scatter point
        rSD = scanner_LUT - scatter_point_position
        rSD_norm = torch.norm(rSD, dim=1)
        bin_edges_distance_along_LOR = bin_edges_scaling.reshape((1,-1)) * rSD_norm.reshape((-1,1))
        bin_centers_distance_along_LOR = (bin_edges_distance_along_LOR[:,1:] + bin_edges_distance_along_LOR[:,:-1]) / 2
        bin_edges = scatter_point_position.reshape((1,1,-1)) + bin_edges_distance_along_LOR.unsqueeze(-1) * (rSD/rSD_norm.unsqueeze(-1)).unsqueeze(1)
        # Evaluate emission integral in many distinct line segments between scatter point and detectors (used for TOF)
        emission_integrals = _line_integrals(
            bin_edges[:,:-1].flatten(end_dim=-2),
            bin_edges[:,1:].flatten(end_dim=-2),
            pet_image,
            object_origin,
            voxel_size,
        ).reshape((N_detectors,num_dense_tof_bins))
        transmission_integrals = _line_integrals(
            scatter_point_position.unsqueeze(0).expand(N_detectors, -1),
            scanner_LUT,
            attenuation_image,
            object_origin,
            voxel_size,
        )
        transmission_integrals_exp = torch.exp(-transmission_integrals)
        rSA = rA - scatter_point_position
        rSB = rB - scatter_point_position
        rSA_norm = torch.norm(rSA, dim=1) # distance between S and A
        rSB_norm = torch.norm(rSB, dim=1) # distance between S and B
        offset_SA = - ((rSB_norm-rSA_norm).unsqueeze(0)/2 + tof_bin_positions.unsqueeze(1)) # first dim TOFbin
        offset_SB = -offset_SA
        # Quantities that do not depend on the TOF bin
        cos_theta = - (rSA*rSB).sum(axis=1) / rSA_norm / rSB_norm
        E_new = photon_energy_after_compton_scatter_511kev(cos_theta)
        energy_efficiency = detector_efficiency(E_new, energy_resolution, energy_low, energy_high)
        # Angle of impingement upon detectors (assumes circle, maybe fix later)
        cos_thetaA_incidence = (rSA[:,:2]*rA[:,:2]).sum(axis=1) / rSA_norm / rA_xy_norm
        cos_thetaB_incidence = (rSB[:,:2]*rB[:,:2]).sum(axis=1) / rSB_norm / rB_xy_norm
        compton_cross_section_ratio = total_compton_cross_section(E_new) / total_compton_cross_section_511keV
        transmission_powB = transmission_integrals_exp[idxB] ** (compton_cross_section_ratio - 1)
        transmission_powA = transmission_integrals_exp[idxA] ** (compton_cross_section_ratio - 1)
        # Emission integrals seen by each TOF bin (first dim TOF bin)
        emission_integralsA = _tof_weighted_emission_of_crystals(offset_SA, idxA, bin_centers_distance_along_LOR, emission_integrals, sigma, tof_splits)
        emission_integralsB = _tof_weighted_emission_of_crystals(offset_SB, idxB, bin_centers_distance_along_LOR, emission_integrals, sigma, tof_splits)
        probability += 1/(rSB_norm**2 * rSA_norm**2) *\
        (emission_integralsA * transmission_powB + emission_integralsB * transmission_powA) *\
        transmission_integrals_exp[idxB] * transmission_integrals_exp[idxA] * mu_value * energy_efficiency * cos_thetaA_incidence * cos_thetaB_incidence * diff_compton_cross_section(cos_theta, E_PET) / total_compton_cross_section_511keV * voxel_volume
    return SparseSinogram(detector_ids_scatter, probability/N_points, proj_meta.info, tof_meta=tof_meta)

def interpolate_sparse_sinogram(
    scatter_sinogram_sparse: SparseSinogram | torch.Tensor,
    proj_meta: ProjMeta,
    idx_intraring: torch.Tensor,
    idx_ring: torch.Tensor,
    tof_bins: Sequence[int] | None = None,
    eval_chunk_size: int = 8192,
    lazy: bool | None = None
    ) -> torch.Tensor | LazySinogram:
    """Interpolates a sparse SSS sinogram estimate using linear interpolation on all oblique planes.

    Args:
        scatter_sinogram_sparse (SparseSinogram | torch.Tensor): Estimated sparse SSS sinogram from the ``compute_sss_sparse_sinogram`` or ``compute_sss_sparse_sinogram_TOF`` functions (a dense sinogram tensor is also accepted)
        proj_meta (ProjMeta): PET projection metadata corresponding to the sinogram
        idx_intraring (torch.Tensor): Intraring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
        idx_ring (torch.Tensor): Interring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
        tof_bins (Sequence[int] | None, optional): TOF bins to interpolate. The interpolator is fit once for all of them and the returned sinogram gets a trailing TOF dimension. Defaults to None (non-TOF).
        eval_chunk_size (int, optional): Number of sinogram (r, theta) positions evaluated at once; bounds device memory (the kernel matrix is ``eval_chunk_size`` x number of sampled positions). Defaults to 8192.
        lazy (bool, optional): Return a :class:`~pytomography.io.PET.shared.LazySinogram`, which keeps the interpolation over angle and radius of the sampled ring pairs (1 GB for the Siemens Biograph mMR with 21 TOF bins and every sixth ring) and interpolates over the ring pairs when a group of angles is asked for, instead of the whole sinogram (34.6 GB with 21 TOF bins). The values are the same. Defaults to None: lazy when the whole sinogram would take more than a quarter of the memory budget (:func:`pytomography.set_memory_budget`), or more than 8 GB without a budget.

    Returns:
        torch.Tensor | LazySinogram: Interpolated SSS sinogram [theta, r, plane] (or [theta, r, plane, TOF]) on the CPU
    """
    device = pytomography.device
    lor_coordinates, sinogram_index = sinogram_coordinates(proj_meta.info)
    _, ring_coordinates = sinogram_to_spatial(proj_meta.info)
    # First interpolate r/theta in all seperate oblique planes
    intra_crystal_index_pairs_sparse = torch.combinations(torch.arange(proj_meta.info['NrCrystalsPerRing']),2).T
    intra_crystal_index_pairs = torch.combinations(idx_intraring,2).T
    inter_crystal_index_pairs = torch.cartesian_prod(idx_ring, idx_ring).T
    angular_radial_idx = lor_coordinates[intra_crystal_index_pairs_sparse[0], intra_crystal_index_pairs_sparse[1]]
    angular_radial_idx_sparse = lor_coordinates[intra_crystal_index_pairs[0], intra_crystal_index_pairs[1]]
    sinogram_plane_idx_sparse = sinogram_index[inter_crystal_index_pairs[0], inter_crystal_index_pairs[1]]
    bins = [None] if tof_bins is None else list(tof_bins)
    # Values at the sampled (r, theta, plane) positions: [N_sparse, N_planes_sparse * N_bins]
    if isinstance(scatter_sinogram_sparse, SparseSinogram):
        theta, r = angular_radial_idx_sparse[:, 0:1], angular_radial_idx_sparse[:, 1:2]
        values = torch.cat([scatter_sinogram_sparse.gather(theta, r, sinogram_plane_idx_sparse.unsqueeze(0), tof_bin=b).to(device) for b in bins], dim=1)
    else:
        dense = scatter_sinogram_sparse
        values = torch.cat([(dense if b is None else dense[..., b])[angular_radial_idx_sparse.T[0], angular_radial_idx_sparse.T[1]][:,sinogram_plane_idx_sparse].to(device) for b in bins], dim=1)
    interpolator = RBFInterpolator(
        angular_radial_idx_sparse.to(torch.float32).to(device),
        values,
        kernel='linear',
        device=device
    )
    x = angular_radial_idx.to(torch.float32).to(device)
    N_theta, N_r = int(proj_meta.info['NrCrystalsPerRing']/2), int(proj_meta.info['NrCrystalsPerRing'])+1
    # The interpolation over angle and radius of every sampled ring pair, kept on the host: [theta, r, sampled ring pair, bin].
    # Each chunk of positions goes to the host as it is evaluated (all of them at once took 1 GB on the device for the
    # mMR with 21 TOF bins, which Windows also counts as host memory)
    rtheta = torch.zeros((N_theta, N_r, sinogram_plane_idx_sparse.shape[0], len(bins)), dtype=torch.float32)
    for i in range(0, x.shape[0], eval_chunk_size):
        rows = angular_radial_idx[i:i+eval_chunk_size]
        values_chunk = interpolator(x[i:i+eval_chunk_size]).reshape(rows.shape[0], len(bins), sinogram_plane_idx_sparse.shape[0])
        rtheta[rows[:, 0], rows[:, 1]] = values_chunk.permute(0, 2, 1).cpu()
    del values_chunk, interpolator
    # Now interpolate Z using grid_sample
    z1_sparse = z2_sparse = ring_coordinates[idx_ring][:,0].cpu().numpy().astype(np.float32)
    z1 = z2 = ring_coordinates[np.arange(proj_meta.info['NrRings'])][:,0].cpu().numpy().astype(np.float32)
    # the rings' z runs up or down with their number (info['axial_direction']): search along increasing z
    direction = 1.0 if z1[-1] >= z1[0] else -1.0
    idx = torch.searchsorted(torch.tensor(direction * z1_sparse), torch.tensor(direction * z1[1:-1]), side='right') - 1
    idx += -(z1_sparse[idx] - z1[1:-1]) / (z1_sparse[idx+1] - z1_sparse[idx])
    idx = torch.concatenate([torch.tensor([0]), idx, torch.tensor([z1_sparse.shape[0]-1])])
    idx = 2/idx.max() * idx  - 1
    interp_mesh = np.stack(np.meshgrid(idx,idx, indexing='ij'), axis=-1)
    interp_mesh = torch.tensor(interp_mesh).to(torch.float32).to(device)
    idx_ring1 = torch.argsort(sinogram_index.ravel()) % sinogram_index.shape[-1]
    idx_ring2 = torch.argsort(sinogram_index.ravel()) // sinogram_index.shape[-1]
    # position of each sinogram plane in the interpolated (z1, z2) grid, flattened
    plane_order = (idx_ring1 * len(z2) + idx_ring2).to(device)
    n_planes = len(z1) * len(z2)
    # angles interpolated at once: the device holds the interpolated planes, the planes in sinogram order, and those
    # in (angle, r, plane, bin) order for the copy to the host
    angles_per_call = max(1, int(gpu_budget(device=device) // (3 * 4 * N_r * n_planes * len(bins))))

    def interpolate_z(angles: torch.Tensor, on: torch.device | None = None) -> torch.Tensor:
        """The interpolated sinogram at the given angles: [angles, r, plane, bin], on the host, or on device ``on``. All
        the bins of a group of angles are interpolated in one call and copied out at once (one call and one copy per bin
        took 90 ms per angle of the mMR with 21 TOF bins)."""
        out = torch.empty((len(angles), N_r, n_planes, len(bins)), dtype=torch.float32, device=on)
        for start in range(0, len(angles), angles_per_call):
            group = angles[start:start + angles_per_call]
            # (angle, r, bin) becomes the channel of grid_sample, which interpolates each channel on its own. On CUDA,
            # grid_sample hands calls with at most 1024 channels to cuDNN, which rounds differently from PyTorch's own
            # kernel; without cuDNN the values do not depend on how many angles or bins are computed at once.
            sampled = rtheta[group].to(device).permute(0, 1, 3, 2).reshape(1, len(group) * N_r * len(bins), len(idx_ring), len(idx_ring))
            with torch.backends.cudnn.flags(enabled=False):
                planes = grid_sample(sampled, interp_mesh.unsqueeze(0), align_corners=True)
            del sampled
            planes = planes.reshape(len(group), N_r, len(bins), n_planes)[..., plane_order]
            out[start:start + len(group)] = planes.permute(0, 1, 3, 2)
            del planes
        return out if tof_bins is not None else out[..., 0]
    shape = (N_theta, N_r, len(z1)*len(z2)) + ((len(bins),) if tof_bins is not None else ())
    if lazy is None:
        lazy = prefer_lazy(4 * np.prod(shape))
    if not lazy:
        # every angle straight into one sinogram (the values do not depend on how many angles are interpolated at
        # once): LazySinogram.to_dense computed groups of up to 1 GB and copied them in, 1.7 GB of temporaries for
        # the 1.65 GB sinogram of the mMR
        return interpolate_z(torch.arange(N_theta))
    return LazySinogram(interpolate_z, shape, description="interpolated single scatter estimate", compute_on=interpolate_z, memory_bytes=rtheta.nbytes)

def _masked(subset: torch.Tensor, projections, mask: torch.Tensor) -> torch.Tensor:
    """``subset * mask``, computed in the memory of ``subset`` when it is a tensor of its own (a copy that indexing
    ``projections`` made, or a subset computed on its own from a LazySinogram), so that one subset-sized array is held
    instead of two."""
    own = (isinstance(subset, torch.Tensor) and subset._base is None and subset.is_floating_point()
           and subset.device == mask.device and torch.broadcast_shapes(subset.shape, mask.shape) == subset.shape
           and not (isinstance(projections, torch.Tensor) and subset.untyped_storage().data_ptr() == projections.untyped_storage().data_ptr()))
    return subset.mul_(mask) if own else subset * mask

def _subset_on(projections, angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The projections at ``angles`` as a new tensor on ``device``: a LazySinogram computes them there; a tensor is
    indexed (a copy) and moved."""
    if isinstance(projections, LazySinogram):
        return projections.compute_at(angles, device)
    return projections[angles.to(projections.device)].to(device)

def scale_estimated_scatter(
    proj_scatter: torch.Tensor,
    system_matrix: SystemMatrix,
    proj_data: torch.Tensor,
    attenuation_image: torch.Tensor,
    attenuation_image_cutoff: float = 0.004,
    sinogram_random: torch.Tensor | None = None,
    in_place: bool = False
    ) -> torch.Tensor:
    """Given an interpolated (but unscaled) SSS sinogram/listmode, scales the scatter estimate by considering back projection of masked data. The mask corresponds to all locations below a certain attenuation value, where it is likely that all detected events are purely due to scatter.

    The sinograms are read one subset of angles at a time, so either may be a :class:`~pytomography.io.PET.shared.LazySinogram`; the scaled estimate is then lazy too.

    Args:
        proj_scatter (torch.Tensor | LazySinogram): Estimated (but unscaled) SSS data.
        system_matrix (SystemMatrix): PET system matrix
        proj_data (torch.Tensor | LazySinogram): PET projection data corresponding to all detected events
        attenuation_image (torch.Tensor): Attenuation map
        attenuation_image_cutoff (float, optional): Mask considers regions below this value (forward projected). In particular, the attenuation map is masked above this value, then forward projected. Regions equal to zero in the forward projection are considered for the mask. This allows for hollow regions within the attenuation map to still be considered. Defaults to 0.004.
        sinogram_random (torch.Tensor | None, optional): Projection data of estimated random events. Defaults to None.
        in_place (bool, optional): Scale ``proj_scatter`` in its own memory when it is a tensor, and return it, instead of a scaled copy (a second 1.65 GB sinogram for the mMR without TOF). The values are the same. Defaults to False.

    Returns:
        torch.Tensor: Scaled SSS projection data (sinogram/listmode).
    """
    # Everything is computed one subset of angles at a time, since these sinograms are large (34.6 GB each for the mMR
    # with 21 TOF bins, 1.65 GB without TOF): the sensitivity image, the mask of the bins whose LOR misses the
    # attenuating object and the masked randoms (without TOF), and the masked scatter estimate and data. Each subset is
    # computed, masked and back projected on pytomography.device, so on a GPU the sinograms never pass through host
    # memory (lazy sinograms are computed there); a subset then takes at most a third of the GPU budget, and is
    # projected in as few chunks as that budget allows. On the CPU, one masked subset at a time, plus the temporaries of
    # computing it; more subsets if the memory budget needs them. The back projections are added up over the subsets,
    # and divided by the sensitivity image at the end.
    TOF = len(proj_data.shape) > 3 # TOF dimension added
    device = torch.device(pytomography.device)
    sinogram_bytes = 4 * float(np.prod(proj_data.shape))
    N_SUBSETS = subsets_for_budget(sinogram_bytes, arrays=2, minimum=20)
    if device.type == 'cuda':
        N_SUBSETS = max(N_SUBSETS, math.ceil(sinogram_bytes / (gpu_budget(device=device) / 3)))
    N_SUBSETS = min(N_SUBSETS, proj_data.shape[0])
    system_matrix.set_n_subsets(N_SUBSETS)
    mask_image = (attenuation_image > attenuation_image_cutoff).to(torch.float32)
    N_splits = getattr(system_matrix, 'N_splits', None)
    if N_splits is not None:
        system_matrix.N_splits = 1
    norm_BP = BP_random = BP_scatter = BP_total = 0
    try:
        for subset_idx in range(N_SUBSETS):
            angles = system_matrix.subset_indices_array[subset_idx].cpu()
            system_matrix.TOF = False
            norm_BP = norm_BP + system_matrix.compute_normalization_factor(subset_idx)
            mask_subset = ~(system_matrix.forward(mask_image, subset_idx=subset_idx) > 0)
            mask_subset = mask_subset.reshape(len(angles), *proj_data.shape[1:3]).to(device)
            if sinogram_random is not None:
                BP_random = BP_random + system_matrix.backward(_masked(_subset_on(sinogram_random, angles, device), None, mask_subset), subset_idx=subset_idx)
            system_matrix.TOF = TOF
            if TOF:
                mask_subset = mask_subset.unsqueeze(-1)
            proj_scatter_masked = _masked(_subset_on(proj_scatter, angles, device), None, mask_subset)
            BP_scatter = BP_scatter + system_matrix.backward(proj_scatter_masked, subset_idx=subset_idx)
            del proj_scatter_masked
            proj_total_masked = _masked(_subset_on(proj_data, angles, device), None, mask_subset)
            BP_total = BP_total + system_matrix.backward(proj_total_masked, subset_idx=subset_idx)
            del proj_total_masked
    finally:
        if N_splits is not None:
            system_matrix.N_splits = N_splits
    def per_sensitivity(BP):
        # voxels no line of response reaches (outside the scanner's field of view) have no sensitivity: they get no
        # weight, instead of 0/0 = NaN, which would make the whole estimate NaN
        return torch.where(norm_BP > 0, BP / norm_BP, torch.zeros_like(BP))
    BP_random_mask = per_sensitivity(BP_random) if sinogram_random is not None else 0
    BP_scatter_mask = per_sensitivity(BP_scatter)
    BP_total_mask = per_sensitivity(BP_total)
    BP_scatter_estimated_mask = BP_total_mask - BP_random_mask
    BP_scatter_estimated_mask[BP_scatter_estimated_mask<0] = 0
    scale_factor = ((BP_scatter_mask*BP_scatter_estimated_mask).sum() / (BP_scatter_mask**2).sum()).item()
    if in_place and isinstance(proj_scatter, torch.Tensor):
        return proj_scatter.mul_(scale_factor)
    return scale_factor * proj_scatter

def get_sss_scatter_estimate(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    pet_image: torch.Tensor,
    attenuation_image: torch.Tensor,
    system_matrix: SystemMatrix,
    proj_data: torch.Tensor | None = None,
    image_stepsize: int = 4,
    attenuation_cutoff: float = 0.004,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4,
    sinogram_random: torch.Tensor | None = None,
    tof_meta: PETTOFMeta = None,
    num_dense_tof_bins: int = 25,
    N_splits: int = 1,
    lazy: bool | None = None
) -> torch.Tensor | LazySinogram:
    """Main function used to get SSS scatter estimation during PET reconstruction

    The detector's energy window and resolution come from ``proj_meta.info``: ``energy_window_low_keV`` and
    ``energy_window_high_keV``, and ``energy_resolution``, the FWHM at 511 keV as a fraction of 511 keV. The readers
    give them where their source does; each one that is None takes its default, 430 keV, no upper limit and 0.15 (see
    :func:`detector_energy_window`).

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to ``pet_image``.
        proj_meta (ProjMeta): Projection metadata corresponding to ``proj_data``, whose ``info`` gives the scanner's geometry and energy window.
        pet_image (torch.Tensor): Reconstructed PET image used to get SSS estimate
        attenuation_image (torch.Tensor): Attenuation map corresponding to PET image
        system_matrix (SystemMatrix): PET system matrix
        proj_data (torch.Tensor | LazySinogram | None): All measured coincident events (sinogram/listmode). If None, then assumes listmode (coincidence events stored in ``proj_meta``). A :class:`~pytomography.io.PET.shared.LazySinogram` is read one subset of angles at a time.
        image_stepsize (int, optional): Spacing between points in object space used to obtain initial sparse sinogram estimate. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider point located at attenuation values above this value as scatter points. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Sinogram interring spacing for initial sparse sinogram estimate. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Sinogram intraring spacing for initial sparse sinogram estimate. Defaults to 4.
        sinogram_random (torch.Tensor | None, optional): Estimated randoms. Defaults to None.
        tof_meta (PETTOFMeta, optional): TOFMetadata corresponding to ``proj_data`` (if TOF is considered). Defaults to None.
        num_dense_tof_bins (int, optional): Number of dense TOF bins to use for partioning emission integrals when performing a TOF estimate. This is seperate from TOF bins used in the PET data. Defaults to 25.
        N_splits (int, optional): Splits the TOF bins into subsets and loops over them sequentially (as opposed to parallel) for scatter estimation. Defaults to 1.
        lazy (bool, optional): Return the estimate as a :class:`~pytomography.io.PET.shared.LazySinogram`, computed a group of angles at a time when it is read, instead of a dense sinogram (34.6 GB for the Siemens Biograph mMR with 21 TOF bins; scaling it made a second one). Use it with ``sinogram_to_listmode`` for list mode data, or as part of the additive term of a sinogram reconstruction. The values are the same. Defaults to None: lazy when the dense estimate would take more than a quarter of the memory budget (:func:`pytomography.set_memory_budget`), or more than 8 GB without a budget.

    Returns:
        torch.Tensor | LazySinogram: Estimated SSS projection data (sinogram/listmode)
    """
    if type(system_matrix) is PETLMSystemMatrix:
        listmode = True
    else:
        listmode = False
    idx_intraring, idx_ring, _ = get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    if tof_meta is None:
        # Get sparse sinogram
        scatter_sinogram_sparse_unscaled = compute_sss_sparse_sinogram(object_meta, proj_meta, pet_image, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
        # Interpolate sparse sinogram
        scatter_sinogram_unscaled  = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled, proj_meta, idx_intraring, idx_ring, lazy=lazy)
    else:
        # Get sparse sinogram
        scatter_sinogram_sparse_unscaled = compute_sss_sparse_sinogram_TOF(object_meta, proj_meta, pet_image, attenuation_image, tof_meta, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize, num_dense_tof_bins, N_splits)
        # Interpolate sparse sinogram (all TOF bins with one interpolator fit)
        scatter_sinogram_unscaled = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled, proj_meta, idx_intraring, idx_ring, tof_bins=range(tof_meta.num_bins), lazy=lazy)
    del(scatter_sinogram_sparse_unscaled) # save memory for next step
    # Need to create a sinogram system matrix for scaling
    if listmode:
        system_matrix = create_sinogramSM_from_LMSM(system_matrix)
        # The events as a lazy sinogram: the scaling reads one subset of angles at a time, and a dense TOF sinogram of
        # the mMR (21 TOF bins) is 34.6 GB. Their bins are found where the system matrix keeps them (its lor_device).
        proj_data = listmode_to_sinogram(proj_meta.detector_ids, proj_meta.info, tof_meta=tof_meta, lazy=True)
    # Scale sinogram (in place: the unscaled estimate is this function's own, and is not needed after)
    proj_scatter = scale_estimated_scatter(scatter_sinogram_unscaled, system_matrix, proj_data, attenuation_image, attenuation_cutoff, sinogram_random = sinogram_random, in_place=True)
    del scatter_sinogram_unscaled
    # Release the GPU memory PyTorch keeps cached from the estimate (3.5 GB for the mMR with 21 TOF bins): on Windows it
    # also counts as the process's host memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return proj_scatter


# ------------------------------------------------------------------------------------------------------------- memory
def _sss_sizes(proj_meta: ProjMeta, tof_meta: PETTOFMeta | None, sinogram_interring_stepsize: int, sinogram_intraring_stepsize: int) -> dict:
    """The sizes the scatter estimate's arrays are made of, from the scanner (``proj_meta.info``) and the sampling."""
    info = proj_meta.info
    crystals, rings = int(info['NrCrystalsPerRing']), int(info['NrRings'])
    sampled_rings = len(range(0, rings, sinogram_interring_stepsize)) + (0 if (rings - 1) % sinogram_interring_stepsize == 0 else 1)
    sampled_crystals_per_ring = len(range(0, crystals, sinogram_intraring_stepsize))
    sampled = sampled_rings * sampled_crystals_per_ring
    return dict(T=1 if tof_meta is None else int(tof_meta.num_bins), N_theta=crystals // 2, N_r=crystals + 1, planes=rings ** 2,
                lors=sampled * (sampled - 1) // 2,                                         # sampled lines of response
                rtheta=sampled_crystals_per_ring * (sampled_crystals_per_ring - 1) // 2,   # sampled (r, theta) positions
                sampled_planes=sampled_rings ** 2, crystal_pairs=crystals * (crystals - 1) // 2)


def estimate_sss_memory(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    system_matrix: SystemMatrix | None = None,
    proj_data=None,
    sinogram_random=None,
    tof_meta: PETTOFMeta | None = None,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4,
    num_dense_tof_bins: int = 25,
    N_splits: int = 1,
    lazy: bool | None = None,
) -> MemoryEstimate:
    r"""The predicted peak RAM and GPU memory of :func:`get_sss_scatter_estimate` with the same arguments, without
    running it (the images themselves are not needed, only their grid, ``object_meta``). ``print`` it, or ``to_dict()``
    for JSON; :func:`plan_sss_memory` chooses ``N_splits`` and ``lazy`` for the memory there is.

    Held for the whole estimate: the PET image and the attenuation map on ``pytomography.device``, the data and the
    randoms (their size where they are; for list mode, the events binned lazily, about 8 bytes an event), the system
    matrix's sensitivity sinogram (for list mode, that of the sinogram system matrix made from it), and the
    interpolated estimate: lazy, the interpolation over angle and radius of the sampled ring pairs on the host; dense,
    the whole sinogram on the host, twice (scaling it makes a copy). Then the steps' temporaries: on the host the
    largest step's, and on the GPU every step's together, since PyTorch's cache keeps each step's blocks until the
    estimate ends (without a GPU, the largest step's, on the host). The steps:

    * the sparse estimate: per sampled line of response, its geometry, the probability of every TOF bin and the
      temporaries of a scatter point; and, for TOF without the fused CUDA kernel, the TOF kernel of
      ceil(T / ``N_splits``) bins by ``num_dense_tof_bins`` pieces, which is what ``N_splits`` bounds;
    * the interpolation over angle and radius: the sampled values, the radial basis function system (3.5 matrices of
      the sampled positions squared: the system, the distances, the kernel and the solver's copy; 3.3 measured) and its
      evaluation in chunks of 8192 positions;
    * for a dense estimate, the interpolation over the ring pairs, a GPU budget at a time;
    * the scaling: a subset of angles of the data and the estimate (TOF), of the mask, randoms and sensitivity (not
      TOF), and the projector's chunk, in as many subsets as :func:`scale_estimated_scatter` uses.

    Args:
        object_meta (ObjectMeta): Object metadata of the PET image.
        proj_meta (ProjMeta): Projection metadata of the data (its ``info`` gives the scanner).
        system_matrix (SystemMatrix | None, optional): The PET system matrix (list mode or sinogram). Defaults to None.
        proj_data (optional): The data, as passed to :func:`get_sss_scatter_estimate` (None for list mode). Defaults to None.
        sinogram_random (optional): Estimated randoms. Defaults to None.
        tof_meta (PETTOFMeta | None, optional): TOF metadata (None: non-TOF). Defaults to None.
        sinogram_interring_stepsize (int, optional): Axial stepsize between sampled rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize between sampled crystals of a ring. Defaults to 4.
        num_dense_tof_bins (int, optional): Pieces of each line for the TOF emission integrals. Defaults to 25.
        N_splits (int, optional): Groups the TOF bins are computed in. Defaults to 1.
        lazy (bool | None, optional): As for :func:`get_sss_scatter_estimate`. Defaults to None: what it would choose.

    Returns:
        MemoryEstimate: The estimate.
    """
    device = torch.device(pytomography.device)
    cuda = device.type == 'cuda'
    on = (lambda b: {'gpu_bytes': float(b)}) if cuda else (lambda b: {'ram_bytes': float(b)})
    s = _sss_sizes(proj_meta, tof_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    T, lors, D = s['T'], s['lors'], int(num_dense_tof_bins)
    N_splits = max(1, min(int(N_splits), T))
    dense_bytes = 4.0 * s['N_theta'] * s['N_r'] * s['planes'] * T
    if lazy is None:
        lazy = bool(prefer_lazy(dense_bytes))
    image = 4.0 * math.prod(object_meta.shape)
    listmode = type(system_matrix) is PETLMSystemMatrix or proj_data is None
    # held for the whole estimate
    parts = [MemoryPart('PET image and attenuation map', **on(2 * image))]
    if proj_data is not None:
        parts.append(MemoryPart('data', *nbytes(proj_data)))
    if sinogram_random is not None:
        parts.append(MemoryPart('randoms', *nbytes(sinogram_random)))
    n_events = 0
    if listmode:
        ids = getattr(proj_meta, 'detector_ids', None)
        n_events = 0 if ids is None else int(ids.shape[0])
        parts.append(MemoryPart('events binned lazily', ram_bytes=8.0 * n_events + 8.0 * (s['N_theta'] + 1)))
        if getattr(proj_meta, 'weights_sensitivity', None) is not None:
            parts.append(MemoryPart('sensitivity sinogram of the sinogram system matrix', ram_bytes=4.0 * s['N_theta'] * s['N_r'] * s['planes']))
    elif system_matrix is not None and getattr(system_matrix, 'sinogram_sensitivity', None) is not None:
        parts.append(MemoryPart("system matrix's sensitivity sinogram", *nbytes(system_matrix.sinogram_sensitivity)))
    rtheta = 4.0 * s['N_theta'] * s['N_r'] * s['sampled_planes'] * T
    if lazy:
        parts.append(MemoryPart('interpolated estimate (lazy: angle and radius of the sampled ring pairs, host)', ram_bytes=rtheta))
    else:
        parts.append(MemoryPart('interpolated estimate (dense) and its scaled copy (host)', ram_bytes=2 * dense_bytes))
    # the steps' temporaries, as (name, host bytes, device bytes)
    fused = tof_meta is not None and _sss_cuda.available(device, T)
    per_lor = 64 + 4 * T + 9 + 4 * T + 4 * 14              # geometry, probability, the sparse result, a point's temporaries
    if tof_meta is not None:
        per_lor += 4 * 7 * T                                # offsets, emission per TOF bin, the update
        if not fused:
            per_lor += 4 * math.ceil(T / N_splits) * D + 4 * 4 * D      # the TOF kernel, gathered pieces, norm, weights
    steps = [('sparse estimate', 112.0 * lors, float(per_lor) * lors)]
    c = min(8192, s['crystal_pairs'])
    values = 4.0 * s['rtheta'] * s['sampled_planes'] * T
    system = 4.0 * (s['rtheta'] + 3) ** 2
    steps.append(('interpolation over angle and radius', rtheta + 32.0 * s['crystal_pairs'],
                  3 * values + max(3.5 * system, 3 * 4.0 * c * (s['rtheta'] + 3) + 2 * 4.0 * c * s['sampled_planes'] * T)))
    if not lazy:                                            # as many angles at once as the GPU budget holds (interpolate_z)
        per_angle = 3 * 4.0 * s['N_r'] * s['planes'] * T
        angles = min(s['N_theta'], max(1, int(gpu_budget(device=device) // per_angle)))
        steps.append(('interpolation over the ring pairs', rtheta, angles * per_angle))
    sinogram_bytes = dense_bytes if proj_data is None else 4.0 * math.prod(proj_data.shape)
    n_subsets = subsets_for_budget(sinogram_bytes, arrays=2, minimum=20)
    if cuda:
        n_subsets = max(n_subsets, math.ceil(sinogram_bytes / (gpu_budget(device=device) / 3)))
    n_subsets = min(n_subsets, s['N_theta'])
    subset_lors = math.ceil(s['N_theta'] / n_subsets) * s['N_r'] * s['planes']
    chunks = max(1, math.ceil(subset_lors * (88 + 8 * T) / gpu_budget(device=device)))      # as _chunk_bounds splits a subset
    chunk = math.ceil(subset_lors / chunks) * (88 + 8 * T)
    steps.append((f'scaling ({n_subsets} subsets)', 32.0 * n_events, 3 * 4.0 * subset_lors * T + 6 * 4.0 * subset_lors + chunk + 6 * image))
    if not cuda:                                            # without a GPU, the device's arrays are on the host
        steps = [(name, host + on_device, 0.0) for name, host, on_device in steps]
    host_step = max(steps, key=lambda step: step[1])
    if host_step[1] > 0:
        parts.append(MemoryPart(f'largest step on the host: {host_step[0]}', ram_bytes=host_step[1], scope='chunk'))
    # On the GPU the steps are added up: PyTorch's cache keeps the blocks of each step, of other sizes than the next
    # step's, until the estimate ends. Measured in tests/calibrate_memory.py, the memory reserved was 1.6 times the
    # peak allocated with the fused TOF kernel and 2.3 times with PyTorch's in three groups of TOF bins (the largest
    # step alone would have been 1.2 times too little); after the GATE mMR estimate with 21 TOF bins it held 3.5 GB
    for name, _, on_gpu in steps:
        if on_gpu > 0:
            parts.append(MemoryPart(f'{name} (GPU)', gpu_bytes=on_gpu, scope='chunk'))
    title = (f"Single scatter simulation ({'TOF, ' if tof_meta is not None else ''}{'lazy' if lazy else 'dense'}"
             + (f", N_splits {N_splits}" if tof_meta is not None else "") + ")")
    settings = {'lazy': lazy, 'N_splits': N_splits} if tof_meta is not None else {'lazy': lazy}
    # the allowance of the PET system matrices, whose tutorials measured what PyTorch's allocators keep on Windows
    return memory_estimate(title, parts, allocator_fraction=PETLMSystemMatrix.memory_allocator_fraction, settings=settings,
                           cuda_projector=True)


def plan_sss_memory(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    system_matrix: SystemMatrix | None = None,
    proj_data=None,
    sinogram_random=None,
    tof_meta: PETTOFMeta | None = None,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4,
    num_dense_tof_bins: int = 25,
    N_splits: int | None = None,
    lazy: bool | None = None,
    ram_gb: float | None = None,
    gpu_gb: float | None = None,
) -> MemoryPlan:
    """The fastest settings of :func:`get_sss_scatter_estimate` with the same arguments that fit in ``ram_gb`` of RAM and
    ``gpu_gb`` of GPU memory, as a :class:`~pytomography.utils.memory.MemoryPlan`, from :func:`estimate_sss_memory`.
    It chooses ``lazy`` (a dense estimate first: it is read faster afterwards) and, with TOF, ``N_splits`` (the fewest
    groups of TOF bins first: with more, the TOF kernel is computed twice; with the fused CUDA kernel, used on a GPU
    when CuPy is installed, ``N_splits`` changes nothing). A setting given (not None) is kept. Without
    budgets, the plan is for what is free on this machine now; on Windows GPU memory counts as RAM too. If nothing
    fits, the plan has ``fits`` False and the settings that come closest; it does not raise.

    Args:
        object_meta (ObjectMeta): As for :func:`get_sss_scatter_estimate`.
        proj_meta (ProjMeta): As for :func:`get_sss_scatter_estimate`.
        system_matrix (SystemMatrix | None, optional): As for :func:`get_sss_scatter_estimate`. Defaults to None.
        proj_data (optional): As for :func:`get_sss_scatter_estimate`. Defaults to None.
        sinogram_random (optional): As for :func:`get_sss_scatter_estimate`. Defaults to None.
        tof_meta (PETTOFMeta | None, optional): As for :func:`get_sss_scatter_estimate`. Defaults to None.
        sinogram_interring_stepsize (int, optional): As for :func:`get_sss_scatter_estimate`. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): As for :func:`get_sss_scatter_estimate`. Defaults to 4.
        num_dense_tof_bins (int, optional): As for :func:`get_sss_scatter_estimate`. Defaults to 25.
        N_splits (int | None, optional): Groups of TOF bins, to keep. Defaults to None: chosen.
        lazy (bool | None, optional): Whether the estimate is lazy, to keep. Defaults to None: chosen.
        ram_gb (float | None, optional): RAM to fit in, in GB. Defaults to None: what is available now.
        gpu_gb (float | None, optional): GPU memory to fit in, in GB. Defaults to None: what is available now.

    Returns:
        MemoryPlan: The plan; pass ``plan.settings`` on to :func:`get_sss_scatter_estimate`.

    Example:
        >>> plan = plan_sss_memory(object_meta, proj_meta, system_matrix, tof_meta=tof_meta, ram_gb=25)
        >>> scatter = get_sss_scatter_estimate(object_meta, proj_meta, pet_image, attenuation_image, system_matrix,
        ...                                    tof_meta=tof_meta, **plan.settings)
    """
    arguments = dict(object_meta=object_meta, proj_meta=proj_meta, system_matrix=system_matrix, proj_data=proj_data,
                     sinogram_random=sinogram_random, tof_meta=tof_meta, sinogram_interring_stepsize=sinogram_interring_stepsize,
                     sinogram_intraring_stepsize=sinogram_intraring_stepsize, num_dense_tof_bins=num_dense_tof_bins)
    settings = {'lazy': (lazy, [False, True])}
    if tof_meta is not None:
        settings['N_splits'] = (N_splits, list(range(1, int(tof_meta.num_bins) + 1)))
    fixed = {'lazy': lazy, 'N_splits': N_splits if tof_meta is not None else None}
    return plan_settings('Single scatter simulation', settings, lambda s, alternatives: estimate_sss_memory(**arguments, **s),
                         fixed, ram_gb, gpu_gb)
