from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.transforms.shared import RotationTransform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.utils import pad_object, unpad_object, pad_proj, unpad_proj, rotate_detector_z
import numpy as np
from ..system_matrix import SystemMatrix
from pytomography.utils import simind_mc
from copy import copy
from typing import Sequence
import shutil

class SPECTSystemMatrix(SystemMatrix):
    r"""System matrix for SPECT imaging implemented using the rotate+sum technique.
    
    Args:
        obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
        proj2proj_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
        object_meta (SPECTObjectMeta): SPECT Object metadata.
        proj_meta (SPECTProjMeta): SPECT projection metadata.
        object_initial_based_on_camera_path (bool): Whether or not to initialize the object estimate based on the camera path; this sets voxels to zero that are outside the SPECT camera path. Defaults to False.
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
    ) -> list:
        """Sets the subsets for this system matrix given ``n_subsets`` total subsets.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 
        """
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
        """Gets the subset of projections :math:`g_m` corresponding to index :math:`m`.

        Args:
            projections (torch.tensor): full projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.tensor: subsampled projections :math:`g_m`
        """
        return projections[...,self.subset_indices_array[subset_idx],:,:]
    
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`H_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

        Args:
            subset_idx (int): Subset index

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.num_projections

    def compute_normalization_factor(self, subset_idx : int | None = None) -> torch.tensor:
        """Function used to get normalization factor :math:`H^T_m 1` corresponding to projection subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Index of subset. If none, then considers all projections. Defaults to None.

        Returns:
            torch.Tensor: normalization factor :math:`H^T_m 1`
        """
        norm_proj = torch.ones(self.proj_meta.shape).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = self.get_projection_subset(norm_proj, subset_idx)
        return self.backward(norm_proj, subset_idx)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Applies forward projection to ``object`` for a SPECT imaging system.

        Args:
            object (torch.tensor[Lx, Ly, Lz]): The object to be forward projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.

        Returns:
            torch.tensor: forward projection estimate :math:`g_m=H_mf`
        """
        # Deal with subset stuff
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        # Start projection
        object = object.to(pytomography.device)
        proj = torch.zeros(
            (N_angles,*self.proj_meta.padded_shape[1:])
            ).to(pytomography.device)
        # Loop through all angles (or groups of angles in parallel)
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            # Format Object
            object_i = pad_object(object)
            # beta = 270 - phi, and backward transform called because projection should be at +beta (requires inverse rotation of object)
            object_i = self.rotation_transform.backward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            # Apply object 2 object transforms
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_indices_i)
            proj[i] = object_i.sum(axis=0)
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Applies back projection to ``proj`` for a SPECT imaging system.

        Args:
            proj (torch.tensor): projections :math:`g` which are to be back projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
            return_norm_constant (bool): Whether or not to return :math:`H_m^T 1` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor: the object :math:`\hat{f} = H_m^T g_m` obtained via back projection.
        """
        # Deal with subset stuff
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones(self.object_meta.shape).to(pytomography.device), mode='back_project')
        # Pad proj and norm_proj (norm_proj used to compute sum_j H_ij)
        proj = pad_proj(proj)
        # First apply proj transforms before back projecting
        for transform in self.proj2proj_transforms[::-1]:
            proj = transform.backward(proj)
        # Setup for back projection
        object = torch.zeros(self.object_meta.padded_shape).to(pytomography.device)
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            # Perform back projection
            object_i = proj[i].unsqueeze(0) * boundary_box_bp
            # Apply object mappings
            for transform in self.obj2obj_transforms[::-1]:
                object_i  = transform.backward(object_i, angle_indices_i)
            # Rotate all objects by by their respective angle
            object_i = self.rotation_transform.forward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            # Add to total 
            object += object_i
        # Unpad
        object = unpad_object(object)
        return object
        
class MonteCarloHybridSPECTSystemMatrix(SPECTSystemMatrix):
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
        crystal_thickness: float,
        cover_thickness: float,
        backscatter_thickness: float,
        energy_resolution_140keV: float = 0,
        advanced_energy_resolution_model: str | None = None,
        advanced_collimator_modeling: bool = False,
    ):
        """Monte Carlo Hybrid SPECT System Matrix class that uses SIMIND to simulate scatter and total projections.
        Args:
            object_meta (ObjectMeta): SPECT ObjectMeta used in reconstruction
            proj_meta (SPECTProjMeta): SPECT projection metadata used in reconstruction
            n_events (int): Number of photons to simulate per projection angle
            n_parallel (int): Number of simulations to perform in parallel, should not exceed number of CPU cores.
            obj2obj_transforms (Sequence[Transform]): List of object to object transforms for back projection
            proj2proj_transforms (Sequence[Transform]): List of projection to projection transforms for back projection
            attenuation_map_140keV (torch.Tensor): Attenuation map at 140keV (used in MC simulation)
            energy_window_params (Sequence[str]): List of strings which constitute a typical "scattwin.win" file used by SIMIND
            primary_window_idx (int): Index from the energy_window_params list corresponding to indices used as photopeak in reconstruction. For single photopeak reconstruction, this will be a list of length 1, while for multi-photopeak reconstruction, this will be a list of length > 1.
            isotope_names (Sequence[str]): List of isotope names used in the simulation
            isotope_ratios (Sequence[float]): Proportion of all isotopes.
            collimator_type (str): Collimator type used for Monte Carlo scatter simulation (should use SIMIND name).
            crystal_thickness (float): Crystal thickness used for Monte Carlo scatter simulation (currently assumes NaI)
            cover_thickness (float): Cover thickness used for simulation. Currently assumes aluminum is used.
            backscatter_thickness (float): Equivalent backscatter thickness used for simulation. Currently assumes pyrex is used.
            energy_resolution_140keV (float): Energy resolution in percent of the detector at 140keV. Currently uses the relationship that resolution is proportional to sqrt(E) for E in keV.
            advanced_energy_resolution_model (str | None, optional): Advanced energy resolution model to use. If provided, then ``energy_resolution_140keV`` is not used. Currently only 'siemens' is supported. Defaults to None.
            advanced_collimator_modeling (bool, optional): Whether or not to use advanced collimator modeling that can be used to model septal penetration and scatter. Defaults to False.
        """
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
        self.cover_thickness = cover_thickness
        self.crystal_thickness = crystal_thickness  
        self.backscatter_thickness = backscatter_thickness
        self.energy_resolution_140keV = energy_resolution_140keV
        self.advanced_energy_resolution_model = advanced_energy_resolution_model
        self.advanced_collimator_modeling = advanced_collimator_modeling
        self.primary_window_idx = primary_window_idx
        self.n_events = n_events
        self.n_parallel = n_parallel
        
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
        """Runs the Monte Carlo scatter simulation using SIMIND and returns the simulated projections.
        Args:
            object (torch.Tensor): Object to simulate
            subset_idx (int | None, optional): Index of the subset to use. If None, then all projections are used. Defaults to None.
        Returns:
            torch.Tensor: Simulated projections
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
                crystal_thickness=self.crystal_thickness,
                cover_thickness=self.cover_thickness,
                backscatter_thickness=self.backscatter_thickness,
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
        