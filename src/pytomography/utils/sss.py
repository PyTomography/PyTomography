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
    prob =  torch.exp(-(offset.unsqueeze(-1)-tof_bins_dense_centers.unsqueeze(0))**2 / (2*tof_meta.sigma.item()**2))
    prob = prob / prob.sum(dim=0).unsqueeze(0)
    return prob

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
    
def compute_sss_sparse_sinogram(
    object_meta: ObjectMeta,
    proj_meta: ProjMeta,
    pet_image: torch.Tensor,
    attenuation_image: torch.Tensor,
    image_stepsize: int = 4,
    attenuation_cutoff: float = 0.004,
    sinogram_interring_stepsize: int = 4,
    sinogram_intraring_stepsize: int = 4
    ) -> torch.Tensor:
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
        torch.Tensor: Estimated sparse single scatter simulation sinogram.
    """
    # Important quantities
    E_PET = torch.tensor(511).to(pytomography.device)
    dr = torch.tensor(object_meta.dr)
    shape = torch.tensor(object_meta.shape)
    object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
    scanner_LUT = proj_meta.scanner_lut
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    # Get sample image/sinogram points
    coords = get_sample_scatter_points(attenuation_image, stepsize=image_stepsize, attenuation_cutoff=attenuation_cutoff)
    coords_position = (coords - shape.unsqueeze(1).to(pytomography.device)/2 + 0.5) * dr.unsqueeze(1).to(pytomography.device)
    _, _, detector_ids_scatter = get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    # Begin
    idxA, idxB = detector_ids_scatter.to(pytomography.device).T
    rA = scanner_LUT.to(pytomography.device)[idxA]
    rB = scanner_LUT.to(pytomography.device)[idxB]
    # Maybe now loop over scatter points
    probability = 0
    counts = 0
    for scatter_point in range(coords.shape[1]):
        # Get position and add random offset within the voxel
        scatter_point_position = coords_position[:,scatter_point] + ((torch.rand(3) - 0.5) * dr).to(pytomography.device)
        # Compute value of attenuation coefficient at scatter point
        mu_value = attenuation_image[tuple(coords[:,scatter_point].tolist())]
        # Compute emission/transmission integrals for that scatter point
        emission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(scanner_LUT.shape[0], -1),
            scanner_LUT.to(pytomography.device),
            pet_image,
            object_origin,
            object_meta.dr,
        )
        transmission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(scanner_LUT.shape[0], -1),
            scanner_LUT.to(pytomography.device),
            attenuation_image.to(pytomography.dtype).to(pytomography.device),
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
        E_new = photon_energy_after_compton_scatter_511kev(cos_theta) # 0.127 ms
        energy_efficiency = detector_efficiency(E_new)
        # Angle of impingement upon detectors (assumes circle, maybe fix later)
        cos_thetaA_incidence = (rSA[:,:2]*rA[:,:2]).sum(axis=1) / rSA_norm / torch.norm(rA[:,:2], dim=1)
        cos_thetaB_incidence = (rSB[:,:2]*rB[:,:2]).sum(axis=1) / rSB_norm / torch.norm(rB[:,:2], dim=1)
        compton_cross_section_ratio = total_compton_cross_section(E_new) / total_compton_cross_section_511keV
        # Start TOF Loop here, needs to consider many different offsets for each emission integral
        # Compute probability without considering TOF information
        probability_without_tof = 1/(rSB_norm**2 * rSA_norm**2) *\
        (emission_integrals[idxA] * transmission_integrals_exp[idxB] ** (compton_cross_section_ratio - 1) + emission_integrals[idxB] * transmission_integrals_exp[idxA] ** (compton_cross_section_ratio - 1)) *\
        transmission_integrals_exp[idxB] * transmission_integrals_exp[idxA] * mu_value * energy_efficiency * cos_thetaA_incidence * cos_thetaB_incidence * diff_compton_cross_section(cos_theta, E_PET) / total_compton_cross_section_511keV * np.prod(object_meta.dr)
        probability += probability_without_tof
        counts += 1
    scatter_sinogram_sparse = shared.listmode_to_sinogram(detector_ids_scatter, proj_meta.info, weights=(probability/counts).cpu())
    return scatter_sinogram_sparse

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
    )->torch.Tensor:
    """Generates a sparse single scatter simulation sinogram for TOF PET data. 

    Args:
        object_meta (ObjectMeta): Object metadata corresponding to reconstructed PET image used in the simulation
        proj_meta (ProjMeta): Projection metadata specifying the details of the PET scanner
        pet_image (torch.Tensor): PET image used to estimate the scatter
        attenuation_image (torch.Tensor): Attenuation map used in scatter simulation
        tof_meta (PETTOFMeta): PET TOF Metadata corresponding to the sinogram estimate
        attenuation_image (torch.Tensor): Attenuation map used in scatter simulation
        image_stepsize (int, optional): Stepsize in x/y/z between sampled scatter points. Defaults to 4.
        attenuation_cutoff (float, optional): Only consider points above this threshhold. Defaults to 0.004.
        sinogram_interring_stepsize (int, optional): Axial stepsize between rings. Defaults to 4.
        sinogram_intraring_stepsize (int, optional): Stepsize of crystals within a given ring. Defaults to 4.
        num_dense_tof_bins (int, optional): Number of dense TOF bins used when partioning the emission integrals (these integrals must be partioned for TOF-based estimation). Defaults to 25.

    Returns:
        torch.Tensor: Estimated sparse single scatter simulation sinogram.
    """
    # Important quantities
    E_PET = torch.tensor(511).to(pytomography.device)
    dr = torch.tensor(object_meta.dr)
    shape = torch.tensor(object_meta.shape)
    object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
    scanner_LUT = proj_meta.scanner_lut
    total_compton_cross_section_511keV = total_compton_cross_section(E_PET)
    # Get sample image/sinogram points
    coords = get_sample_scatter_points(attenuation_image, stepsize=image_stepsize, attenuation_cutoff=attenuation_cutoff)
    coords_position = (coords - shape.unsqueeze(1).to(pytomography.device)/2 + 0.5) * dr.unsqueeze(1).to(pytomography.device)
    _, _, detector_ids_scatter = get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)
    # Begin
    idxA, idxB = detector_ids_scatter.to(pytomography.device).T
    rA = scanner_LUT.to(pytomography.device)[idxA]
    rB = scanner_LUT.to(pytomography.device)[idxB]
    # Now loop over scatter points
    probability = torch.zeros([tof_meta.num_bins, detector_ids_scatter.shape[0]]).to(pytomography.device)
    tof_bin_idxs = torch.arange(tof_meta.num_bins)
    tof_bin_positions = tof_meta.bin_positions.to(pytomography.device)
    counts = 0
    for scatter_point in range(coords.shape[1]):
        scatter_point_position = coords_position[:,scatter_point] + ((torch.rand(3) - 0.5) * dr).to(pytomography.device)
        # Compute value of attenuation coefficient at scatter point
        mu_value = attenuation_image[tuple(coords[:,scatter_point].tolist())]
        # Compute emission/transmission integrals for that scatter point
        rSD = scanner_LUT.to(pytomography.device) - scatter_point_position
        rSD_norm = torch.norm(rSD, dim=1)
        bin_edges_scaling = torch.linspace(0,1,num_dense_tof_bins+1).to(pytomography.device)
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
        ).reshape((scanner_LUT.shape[0],num_dense_tof_bins))
        transmission_integrals = parallelproj.joseph3d_fwd(
            scatter_point_position.unsqueeze(0).expand(scanner_LUT.shape[0], -1),
            scanner_LUT.to(pytomography.device),
            attenuation_image.to(pytomography.dtype).to(pytomography.device),
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
        # Loop over split TOF bins
        for tof_bin_idxs_partial in torch.tensor_split(tof_bin_idxs, N_splits):
            prob_SA = tof_efficiency(offset_SA[tof_bin_idxs_partial], bin_centers_distance_along_LOR[idxA], tof_meta) # first dim TOFbin
            prob_SB = tof_efficiency(offset_SB[tof_bin_idxs_partial], bin_centers_distance_along_LOR[idxB], tof_meta) # first dim TOFbin
            # Compute emission integrals
            emission_integralsA = (prob_SA*emission_integrals[idxA].unsqueeze(0)).sum(dim=-1)
            emission_integralsB = (prob_SB*emission_integrals[idxB].unsqueeze(0)).sum(dim=-1)
            cos_theta = - (rSA*rSB).sum(axis=1) / rSA_norm / rSB_norm
            E_new = photon_energy_after_compton_scatter_511kev(cos_theta) 
            energy_efficiency = detector_efficiency(E_new)
            # Angle of impingement upon detectors (assumes circle, maybe fix later)
            cos_thetaA_incidence = (rSA[:,:2]*rA[:,:2]).sum(axis=1) / rSA_norm / torch.norm(rA[:,:2], dim=1)
            cos_thetaB_incidence = (rSB[:,:2]*rB[:,:2]).sum(axis=1) / rSB_norm / torch.norm(rB[:,:2], dim=1)
            compton_cross_section_ratio = total_compton_cross_section(E_new) / total_compton_cross_section_511keV
            probability[tof_bin_idxs_partial] += 1/(rSB_norm**2 * rSA_norm**2) *\
            (emission_integralsA * transmission_integrals_exp[idxB] ** (compton_cross_section_ratio - 1) + emission_integralsB * transmission_integrals_exp[idxA] ** (compton_cross_section_ratio - 1)) *\
            transmission_integrals_exp[idxB] * transmission_integrals_exp[idxA] * mu_value * energy_efficiency * cos_thetaA_incidence * cos_thetaB_incidence * diff_compton_cross_section(cos_theta, E_PET) / total_compton_cross_section_511keV * np.prod(object_meta.dr)
        counts+=1
    probability = probability.ravel()
    # Get TOF bins
    TOF_bins = torch.cartesian_prod(torch.arange(tof_meta.num_bins), detector_ids_scatter[:,0])[:,0]
    # This aligns with how probability was unraveled
    detector_ids_scatter_with_TOF = torch.concatenate([detector_ids_scatter.repeat(tof_meta.num_bins,1), TOF_bins.unsqueeze(1)], dim=-1)
    scatter_sinogram_sparse = shared.listmode_to_sinogram(detector_ids_scatter_with_TOF, proj_meta.info, tof_meta=tof_meta, weights=(probability/counts).cpu())
    return scatter_sinogram_sparse

def interpolate_sparse_sinogram(
    scatter_sinogram_sparse: torch.Tensor,
    proj_meta: ProjMeta,
    idx_intraring: torch.Tensor,
    idx_ring: torch.Tensor
    ) -> torch.Tensor:
    """Interpolates a sparse SSS sinogram estimate using linear interpolation on all oblique planes.

    Args:
        scatter_sinogram_sparse (torch.Tensor): Estimated sparse SSS sinogram from the ``compute_sss_sparse_sinogram`` or ``compute_sss_sparse_sinogram_TOF`` functions
        proj_meta (ProjMeta): PET projection metadata corresponding to the sinogram
        idx_intraring (torch.Tensor): Intraring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
        idx_ring (torch.Tensor): Interring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)

    Returns:
        torch.Tensor: Interpolated SSS sinogram
    """
    lor_coordinates, sinogram_index = sinogram_coordinates(proj_meta.info)
    _, ring_coordinates = sinogram_to_spatial(proj_meta.info)
    # First interpolate r/theta in all seperate oblique planes
    intra_crystal_index_pairs_sparse = torch.combinations(torch.arange(proj_meta.info['NrCrystalsPerRing']),2).T
    intra_crystal_index_pairs = torch.combinations(idx_intraring,2).T
    inter_crystal_index_pairs = torch.cartesian_prod(idx_ring, idx_ring).T
    angular_radial_idx = lor_coordinates[intra_crystal_index_pairs_sparse[0], intra_crystal_index_pairs_sparse[1]]
    angular_radial_idx_sparse = lor_coordinates[intra_crystal_index_pairs[0], intra_crystal_index_pairs[1]]
    sinogram_plane_idx_sparse = sinogram_index[inter_crystal_index_pairs[0], inter_crystal_index_pairs[1]]
    interpolator = RBFInterpolator(
        angular_radial_idx_sparse.to(torch.float32).to(pytomography.device),
        scatter_sinogram_sparse[angular_radial_idx_sparse.T[0], angular_radial_idx_sparse.T[1]][:,sinogram_plane_idx_sparse].to(pytomography.device),
        kernel='linear',
        device=pytomography.device
    )
    interp_vals = interpolator(angular_radial_idx.to(torch.float32).to(pytomography.device))
    scatter_sinogram_interp_rtheta = torch.zeros(*scatter_sinogram_sparse.shape[:2], sinogram_plane_idx_sparse.shape[0]).to(pytomography.device)
    scatter_sinogram_interp_rtheta[angular_radial_idx.T[0], angular_radial_idx.T[1]] = interp_vals
    scatter_sinogram_interp_rtheta = scatter_sinogram_interp_rtheta.reshape(scatter_sinogram_interp_rtheta.shape[0], scatter_sinogram_interp_rtheta.shape[1], len(idx_ring), len(idx_ring))
    # Now interpolate Z using grid_sample
    z1_sparse = z2_sparse = ring_coordinates[idx_ring][:,0].cpu().numpy().astype(np.float32)
    z1 = z2 = ring_coordinates[np.arange(proj_meta.info['NrRings'])][:,0].cpu().numpy().astype(np.float32)
    idx = torch.searchsorted(torch.tensor(-z1_sparse), torch.tensor(-z1[1:-1]), side='right') - 1
    idx += -(z1_sparse[idx] - z1[1:-1]) / (z1_sparse[idx+1] - z1_sparse[idx])
    idx = torch.concatenate([torch.tensor([0]), idx, torch.tensor([z1_sparse.shape[0]-1])])
    idx = 2/idx.max() * idx  - 1
    interp_mesh = np.stack(np.meshgrid(idx,idx, indexing='ij'), axis=-1)
    interp_mesh = torch.tensor(interp_mesh).to(torch.float32).to(pytomography.device)
    # r/theta becomes batch/channel in grid_sample, which is fine
    scatter_sinogram_interp_all = grid_sample(
        scatter_sinogram_interp_rtheta.flatten(start_dim=0, end_dim=1).unsqueeze(0),
        interp_mesh.unsqueeze(0),
        align_corners=True
    ).reshape((*scatter_sinogram_interp_rtheta.shape[:2], len(z1), len(z2))).cpu()
    idx_ring1 = torch.argsort(sinogram_index.ravel()) % sinogram_index.shape[-1]
    idx_ring2 = torch.argsort(sinogram_index.ravel()) // sinogram_index.shape[-1]
    scatter_sinogram_interp_all = scatter_sinogram_interp_all[:,:,idx_ring1,idx_ring2]
    return scatter_sinogram_interp_all

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
    proj_data_mask = system_matrix.forward((attenuation_image>attenuation_image_cutoff).to(torch.float32))>0
    # Random
    if sinogram_random is not None:
        BP_random_mask = system_matrix.backward(~proj_data_mask*sinogram_random.to(system_matrix.output_device)) / norm_BP
    else:
        BP_random_mask = 0
    if len(proj_data.shape)>3: # TOF dimension added
        system_matrix.TOF = True
        proj_data_mask = proj_data_mask.unsqueeze(-1)
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
        proj_scatter_masked = system_matrix.get_projection_subset(proj_scatter, subset_idx) * system_matrix.get_projection_subset(~proj_data_mask, subset_idx)
        proj_total_masked = system_matrix.get_projection_subset(proj_data, subset_idx) * system_matrix.get_projection_subset(~proj_data_mask, subset_idx)
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
    if tof_meta is None:
        # Get sparse sinogram
        scatter_sinogram_sparse_unscaled = compute_sss_sparse_sinogram(object_meta, proj_meta, pet_image, attenuation_image, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize)
        # Interpolate sparse sinogram
        scatter_sinogram_unscaled  = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled, proj_meta, *get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)[:2])
    else:
        # Get sparse sinogram
        scatter_sinogram_sparse_unscaled = compute_sss_sparse_sinogram_TOF(object_meta, proj_meta, pet_image, attenuation_image, tof_meta, image_stepsize, attenuation_cutoff, sinogram_interring_stepsize, sinogram_intraring_stepsize, num_dense_tof_bins, N_splits)
        scatter_sinogram_unscaled = torch.empty(scatter_sinogram_sparse_unscaled.shape, dtype=torch.float32)
        # Interpolate sparse sinogram (loop over TOF bins)
        for i in range(scatter_sinogram_sparse_unscaled.shape[-1]):
            scatter_sinogram_unscaled[...,i] = interpolate_sparse_sinogram(scatter_sinogram_sparse_unscaled[:,:,:,i], proj_meta, *get_sample_detector_ids(proj_meta, sinogram_interring_stepsize, sinogram_intraring_stepsize)[:2])
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

