from __future__ import annotations
from typing import Sequence
import torch
import pytomography
from pytomography.io.PET import shared
from pytomography.projectors.PET import PETLMSystemMatrix
import numpy as np
import parallelproj
from torchrbf import RBFInterpolator
from torch.nn.functional import grid_sample
from pytomography.io.PET.shared import sinogram_coordinates, sinogram_to_spatial, listmode_to_sinogram
from pytomography.projectors.PET import create_sinogramSM_from_LMSM
from pytomography.metadata.PET import PETTOFMeta
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix

def total_compton_cross_section(energy: torch.Tensor) -> torch.Tensor:
    """Computes the total compton cross section of interaction :math:`\sigma` at the given photon energies

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

def detector_efficiency(
    scatter_energy: torch.Tensor,
    energy_resolution: float = 0.15,
    energy_threshhold: float = 430
    ) -> torch.Tensor:
    """Computes the probability a photon of given energy is detected within the energy limits of the detector

    Args:
        scatter_energy (torch.Tensor): Energy of the photon impinging the detector
        energy_resolution (float, optional): Energy resolution of the crystals (represented as a fraction of 511keV). This is the uncertainty of energy measurements. Defaults to 0.15.
        energy_threshhold (float, optional): Lower limit of energies detected by the crystal which are registered as events. Defaults to 430.

    Returns:
        torch.Tensor: Probability that the photon gets detected
    """
    sigma = 511 * energy_resolution / (2*np.sqrt(2*np.log(2)))
    return 0.5 * (1 - torch.erf((energy_threshhold-scatter_energy) / (np.sqrt(2) * sigma)))

def _tof_efficiency(offset: torch.Tensor, tof_bins_dense_centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """``tof_efficiency`` with the TOF resolution given as a Python float (so no device-to-host transfer per call)."""
    prob =  torch.exp(-(offset.unsqueeze(-1)-tof_bins_dense_centers.unsqueeze(0))**2 / (2*sigma**2))
    prob = prob / prob.sum(dim=0).unsqueeze(0)
    return prob

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
        self.shape = (int(info['NrCrystalsPerRing']/2), int(info['NrCrystalsPerRing'])+1, int((info['moduleAxialNr']*info['crystalAxialNr'])**2))
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
    scanner_LUT = proj_meta.scanner_lut.to(device)
    idxA, idxB = detector_ids_scatter.to(device).T
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

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to reconstructed PET image used in the simulation
        proj_meta (ProjMeta): Projection metadata specifying the details of the PET scanner
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
    object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    voxel_volume = np.prod(object_meta.dr)
    positions, mu_values, detector_ids_scatter, scanner_LUT, idxA, idxB, rA, rB = _scatter_setup(object_meta, proj_meta, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    attenuation_image = attenuation_image.to(pytomography.dtype).to(pytomography.device)
    rA_xy_norm = torch.norm(rA[:,:2], dim=1)
    rB_xy_norm = torch.norm(rB[:,:2], dim=1)
    N_points = positions.shape[1]
    # Loop over scatter points
    probability = 0
    for scatter_point in range(N_points):
        scatter_point_position = positions[:,scatter_point]
        mu_value = mu_values[scatter_point]
        # Compute emission/transmission integrals for that scatter point
        emission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(scanner_LUT.shape[0], -1),
            scanner_LUT,
            pet_image,
            object_origin,
            object_meta.dr,
        )
        transmission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(scanner_LUT.shape[0], -1),
            scanner_LUT,
            attenuation_image,
            object_origin,
            object_meta.dr,
        )
        transmission_integrals_exp = torch.exp(-transmission_integrals)
        # Compute scatter contribution
        rSA = rA - scatter_point_position
        rSB = rB - scatter_point_position
        rSA_norm = torch.norm(rSA, dim=1) # distance between S and A
        rSB_norm = torch.norm(rSB, dim=1) # distance between S and B
        # Compute cos(scattering_angle) = cos(pi-angle_between_vectors) = -cos(angle_between_vectors)
        cos_theta = - (rSA*rSB).sum(axis=1) / rSA_norm / rSB_norm
        E_new = photon_energy_after_compton_scatter_511kev(cos_theta)
        energy_efficiency = detector_efficiency(E_new)
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

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to reconstructed PET image used in the simulation
        proj_meta (ProjMeta): Projection metadata specifying the details of the PET scanner
        pet_image (torch.Tensor): PET image used to estimate the scatter
        attenuation_image (torch.Tensor): Attenuation map used in scatter simulation
        tof_meta (PETTOFMeta): PET TOF Metadata corresponding to the sinogram estimate
        image_stepsize (int, optional): Stepsize in x/y/z between sampled scatter points. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider points above this threshhold. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Axial stepsize between rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize of crystals within a given ring. Defaults to 4.
        num_dense_tof_bins (int, optional): Number of dense TOF bins used when partioning the emission integrals (these integrals must be partioned for TOF-based estimation). Defaults to 25.
        N_splits (int, optional): Splits the TOF bins into subsets and loops over them sequentially (as opposed to parallel) to bound device memory. Defaults to 1.

    Returns:
        SparseSinogram: Estimated single scatter simulation at the sampled LORs and TOF bins (``.to_dense()`` gives the sinogram the function used to return).
    """
    # Important quantities
    E_PET = torch.tensor(511).to(pytomography.device)
    object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    voxel_volume = np.prod(object_meta.dr)
    positions, mu_values, detector_ids_scatter, scanner_LUT, idxA, idxB, rA, rB = _scatter_setup(object_meta, proj_meta, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    attenuation_image = attenuation_image.to(pytomography.dtype).to(pytomography.device)
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
        emission_integrals = parallelproj.joseph3d_fwd(
            bin_edges[:,:-1].flatten(end_dim=-2),
            bin_edges[:,1:].flatten(end_dim=-2),
            pet_image,
            object_origin,
            object_meta.dr,
        ).reshape((N_detectors,num_dense_tof_bins))
        transmission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(N_detectors, -1),
            scanner_LUT,
            attenuation_image,
            object_origin,
            object_meta.dr,
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
        energy_efficiency = detector_efficiency(E_new)
        # Angle of impingement upon detectors (assumes circle, maybe fix later)
        cos_thetaA_incidence = (rSA[:,:2]*rA[:,:2]).sum(axis=1) / rSA_norm / rA_xy_norm
        cos_thetaB_incidence = (rSB[:,:2]*rB[:,:2]).sum(axis=1) / rSB_norm / rB_xy_norm
        compton_cross_section_ratio = total_compton_cross_section(E_new) / total_compton_cross_section_511keV
        transmission_powB = transmission_integrals_exp[idxB] ** (compton_cross_section_ratio - 1)
        transmission_powA = transmission_integrals_exp[idxA] ** (compton_cross_section_ratio - 1)
        emission_integrals_A = emission_integrals[idxA].unsqueeze(0)
        emission_integrals_B = emission_integrals[idxB].unsqueeze(0)
        bin_centers_A = bin_centers_distance_along_LOR[idxA]
        bin_centers_B = bin_centers_distance_along_LOR[idxB]
        # Loop over split TOF bins
        for start, end in tof_splits:
            prob_SA = _tof_efficiency(offset_SA[start:end], bin_centers_A, sigma) # first dim TOFbin
            prob_SB = _tof_efficiency(offset_SB[start:end], bin_centers_B, sigma) # first dim TOFbin
            # Compute emission integrals
            emission_integralsA = (prob_SA*emission_integrals_A).sum(dim=-1)
            emission_integralsB = (prob_SB*emission_integrals_B).sum(dim=-1)
            probability[start:end] += 1/(rSB_norm**2 * rSA_norm**2) *\
            (emission_integralsA * transmission_powB + emission_integralsB * transmission_powA) *\
            transmission_integrals_exp[idxB] * transmission_integrals_exp[idxA] * mu_value * energy_efficiency * cos_thetaA_incidence * cos_thetaB_incidence * diff_compton_cross_section(cos_theta, E_PET) / total_compton_cross_section_511keV * voxel_volume
    return SparseSinogram(detector_ids_scatter, probability/N_points, proj_meta.info, tof_meta=tof_meta)

def interpolate_sparse_sinogram(
    scatter_sinogram_sparse: SparseSinogram | torch.Tensor,
    proj_meta: ProjMeta,
    idx_intraring: torch.Tensor,
    idx_ring: torch.Tensor,
    tof_bins: Sequence[int] | None = None,
    eval_chunk_size: int = 8192
    ) -> torch.Tensor:
    """Interpolates a sparse SSS sinogram estimate using linear interpolation on all oblique planes.

    Args:
        scatter_sinogram_sparse (SparseSinogram | torch.Tensor): Estimated sparse SSS sinogram from the ``compute_sss_sparse_sinogram`` or ``compute_sss_sparse_sinogram_TOF`` functions (a dense sinogram tensor is also accepted)
        proj_meta (ProjMeta): PET projection metadata corresponding to the sinogram
        idx_intraring (torch.Tensor): Intraring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
        idx_ring (torch.Tensor): Interring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
        tof_bins (Sequence[int] | None, optional): TOF bins to interpolate. The interpolator is fit once for all of them and the returned sinogram gets a trailing TOF dimension. Defaults to None (non-TOF).
        eval_chunk_size (int, optional): Number of sinogram (r, theta) positions evaluated at once; bounds device memory (the kernel matrix is ``eval_chunk_size`` x number of sampled positions). Defaults to 8192.

    Returns:
        torch.Tensor: Interpolated SSS sinogram [theta, r, plane] (or [theta, r, plane, TOF]) on the CPU
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
    interp_vals = torch.cat([interpolator(x[i:i+eval_chunk_size]) for i in range(0, x.shape[0], eval_chunk_size)])
    interp_vals = interp_vals.reshape(x.shape[0], len(bins), sinogram_plane_idx_sparse.shape[0])
    # Now interpolate Z using grid_sample
    z1_sparse = z2_sparse = ring_coordinates[idx_ring][:,0].cpu().numpy().astype(np.float32)
    z1 = z2 = ring_coordinates[np.arange(proj_meta.info['NrRings'])][:,0].cpu().numpy().astype(np.float32)
    idx = torch.searchsorted(torch.tensor(-z1_sparse), torch.tensor(-z1[1:-1]), side='right') - 1
    idx += -(z1_sparse[idx] - z1[1:-1]) / (z1_sparse[idx+1] - z1_sparse[idx])
    idx = torch.concatenate([torch.tensor([0]), idx, torch.tensor([z1_sparse.shape[0]-1])])
    idx = 2/idx.max() * idx  - 1
    interp_mesh = np.stack(np.meshgrid(idx,idx, indexing='ij'), axis=-1)
    interp_mesh = torch.tensor(interp_mesh).to(torch.float32).to(device)
    idx_ring1 = torch.argsort(sinogram_index.ravel()) % sinogram_index.shape[-1]
    idx_ring2 = torch.argsort(sinogram_index.ravel()) // sinogram_index.shape[-1]
    N_theta, N_r = int(proj_meta.info['NrCrystalsPerRing']/2), int(proj_meta.info['NrCrystalsPerRing'])+1
    scatter_sinogram_interp_all = torch.empty((N_theta, N_r, len(z1)*len(z2), len(bins)), dtype=torch.float32)
    for b in range(len(bins)):
        scatter_sinogram_interp_rtheta = torch.zeros(N_theta, N_r, sinogram_plane_idx_sparse.shape[0]).to(device)
        scatter_sinogram_interp_rtheta[angular_radial_idx.T[0], angular_radial_idx.T[1]] = interp_vals[:, b]
        scatter_sinogram_interp_rtheta = scatter_sinogram_interp_rtheta.reshape(N_theta, N_r, len(idx_ring), len(idx_ring))
        # r/theta becomes batch/channel in grid_sample, which is fine
        scatter_sinogram_interp_bin = grid_sample(
            scatter_sinogram_interp_rtheta.flatten(start_dim=0, end_dim=1).unsqueeze(0),
            interp_mesh.unsqueeze(0),
            align_corners=True
        ).reshape((N_theta, N_r, len(z1), len(z2))).cpu()
        scatter_sinogram_interp_all[..., b] = scatter_sinogram_interp_bin[:,:,idx_ring1,idx_ring2]
    return scatter_sinogram_interp_all if tof_bins is not None else scatter_sinogram_interp_all[..., 0]

def scale_estimated_scatter(
    proj_scatter: torch.Tensor,
    system_matrix: SystemMatrix,
    proj_data: torch.Tensor,
    attenuation_image: torch.Tensor,
    attenuation_image_cutoff: float = 0.004,
    sinogram_random: torch.Tensor | None = None
    ) -> torch.Tensor:
    """Given an interpolated (but unscaled) SSS sinogram/listmode, scales the scatter estimate by considering back projection of masked data. The mask corresponds to all locations below a certain attenuation value, where it is likely that all detected events are purely due to scatter.

    Args:
        proj_scatter (torch.Tensor): Estimated (but unscaled) SSS data.
        system_matrix (SystemMatrix): PET system matrix
        proj_data (torch.Tensor): PET projection data corresponding to all detected events
        attenuation_image (torch.Tensor): Attenuation map
        attenuation_image_cutoff (float, optional): Mask considers regions below this value (forward projected). In particular, the attenuation map is masked above this value, then forward projected. Regions equal to zero in the forward projection are considered for the mask. This allows for hollow regions within the attenuation map to still be considered. Defaults to 0.004.
        sinogram_random (torch.Tensor | None, optional): Projection data of estimated random events. Defaults to None.

    Returns:
        torch.Tensor: Scaled SSS projection data (sinogram/listmode).
    """
    system_matrix.TOF = False
    norm_BP = system_matrix.compute_normalization_factor()
    # Mask of sinogram bins whose LOR misses the attenuating object (computed once; it is the size of the sinogram)
    proj_outside_mask = ~(system_matrix.forward((attenuation_image>attenuation_image_cutoff).to(torch.float32))>0)
    # Random
    if sinogram_random is not None:
        BP_random_mask = system_matrix.backward(proj_outside_mask*sinogram_random.to(system_matrix.output_device)) / norm_BP
    else:
        BP_random_mask = 0
    if len(proj_data.shape)>3: # TOF dimension added
        system_matrix.TOF = True
        proj_outside_mask = proj_outside_mask.unsqueeze(-1)
    else:
        system_matrix.TOF = False
    # Scatter
    # Need to get back projecgion of masked scatter and masked totall;
    # we'll split into subsets to preserve memory since this requires
    # making copies of potentially very large sinogram tensors
    N_SUBSETS = 20
    system_matrix.set_n_subsets(N_SUBSETS)
    BP_scatter_mask = 0
    BP_total_mask = 0
    for subset_idx in range(N_SUBSETS):
        mask_subset = system_matrix.get_projection_subset(proj_outside_mask, subset_idx)
        proj_scatter_masked = system_matrix.get_projection_subset(proj_scatter, subset_idx) * mask_subset
        proj_total_masked = system_matrix.get_projection_subset(proj_data, subset_idx) * mask_subset
        BP_scatter_mask += system_matrix.backward(proj_scatter_masked, subset_idx = subset_idx) / norm_BP
        BP_total_mask += system_matrix.backward(proj_total_masked, subset_idx=subset_idx) / norm_BP
    BP_scatter_estimated_mask = BP_total_mask - BP_random_mask
    BP_scatter_estimated_mask[BP_scatter_estimated_mask<0] = 0
    scale_factor = ((BP_scatter_mask*BP_scatter_estimated_mask).sum() / (BP_scatter_mask**2).sum()).item()
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
    N_splits: int = 1
) -> torch.Tensor:
    """Main function used to get SSS scatter estimation during PET reconstruction

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to ``pet_image``.
        proj_meta (ProjMeta): Projection metadata corresponding to ``proj_data``.
        pet_image (torch.Tensor): Reconstructed PET image used to get SSS estimate
        attenuation_image (torch.Tensor): Attenuation map corresponding to PET image
        system_matrix (SystemMatrix): PET system matrix
        proj_data (torch.Tensor | None): All measured coincident events (sinogram/listmode). If None, then assumes listmode (coincidence events stored in ``proj_meta``).
        image_stepsize (int, optional): Spacing between points in object space used to obtain initial sparse sinogram estimate. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider point located at attenuation values above this value as scatter points. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Sinogram interring spacing for initial sparse sinogram estimate. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Sinogram intraring spacing for initial sparse sinogram estimate. Defaults to 4.
        sinogram_random (torch.Tensor | None, optional): Estimated randoms. Defaults to None.
        tof_meta (PETTOFMeta, optional): TOFMetadata corresponding to ``proj_data`` (if TOF is considered). Defaults to None.
        num_dense_tof_bins (int, optional): Number of dense TOF bins to use for partioning emission integrals when performing a TOF estimate. This is seperate from TOF bins used in the PET data. Defaults to 25.
        N_splits (int, optional): Splits the TOF bins into subsets and loops over them sequentially (as opposed to parallel) for scatter estimation. Defaults to 1.

    Returns:
        torch.Tensor: Estimated SSS projection data (sinogram/listmode)
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
        scatter_sinogram_unscaled  = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled, proj_meta, idx_intraring, idx_ring)
    else:
        # Get sparse sinogram
        scatter_sinogram_sparse_unscaled = compute_sss_sparse_sinogram_TOF(object_meta, proj_meta, pet_image, attenuation_image, tof_meta, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize, num_dense_tof_bins, N_splits)
        # Interpolate sparse sinogram (all TOF bins with one interpolator fit)
        scatter_sinogram_unscaled = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled, proj_meta, idx_intraring, idx_ring, tof_bins=range(tof_meta.num_bins))
    del(scatter_sinogram_sparse_unscaled) # save memory for next step
    # Need to create a sinogram system matrix for scaling
    if listmode:
        system_matrix = create_sinogramSM_from_LMSM(system_matrix)
        if tof_meta is None:
            proj_data = listmode_to_sinogram(proj_meta.detector_ids.cpu(), proj_meta.info)
        else:
            proj_data = listmode_to_sinogram(proj_meta.detector_ids.cpu(), proj_meta.info, tof_meta=tof_meta)
    # Scale sinogram
    proj_scatter = scale_estimated_scatter(scatter_sinogram_unscaled, system_matrix, proj_data, attenuation_image, attenuation_cutoff, sinogram_random = sinogram_random)
    return proj_scatter
