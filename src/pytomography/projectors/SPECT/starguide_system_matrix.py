from __future__ import annotations
import torch
import pytomography
from pytomography.utils import rotate_detector_z
from torch.nn.functional import pad
from pytomography.projectors.system_matrix import SystemMatrix
from pytomography.transforms.SPECT import SPECTPSFTransform
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import StarGuideProjMeta
from kornia.geometry.transform import Translate

class StarGuideSystemMatrix(SystemMatrix):
    r"""System matrix for the StarGuide SPECT imaging system form General Electric Healthcare.
    
    Args:
        obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
        proj2proj_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
        object_meta (SPECTObjectMeta): SPECT Object metadata.
        proj_meta (StarGuideProjMeta): Projection metadata pertaining to the StarGuide system.
        object_initial_based_on_camera_path (bool): Whether or not to initialize the object estimate based on the camera path; this sets voxels to zero that are outside the SPECT camera path. Defaults to False.
    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: StarGuideProjMeta,
        obj2obj_transforms = [],
        proj2proj_transforms = [],    
    ):
        super().__init__(object_meta, proj_meta, obj2obj_transforms, proj2proj_transforms)
        self.times = self.proj_meta.times.reshape(-1,1,1) / 1e3
    @torch.no_grad()
    
    def forward(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
    ): 
        r"""Applies forward projection to ``object``.

        Args:
            object (torch.tensor[Lx, Ly, Lz]): The object to be forward projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.

        Returns:
            torch.tensor: forward projection estimate :math:`g_m=H_mf`
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        projections = torch.zeros((N_angles, *self.proj_meta.shape[1:])).to(pytomography.device)
        for angle in torch.unique(self.proj_meta.angles[angle_indices]):
            idx = self.proj_meta.angles[angle_indices]==angle
            offsets_i = self.proj_meta.offsets[angle_indices][idx]
            obj_rotate = rotate_detector_z(object, angles=angle)
            for transform in self.obj2obj_transforms:
                # PSF transform depends on radial position
                if type(transform)==SPECTPSFTransform:
                    for i, j in enumerate(angle_indices[idx]):
                        obj_rotate[i] = transform.forward(obj_rotate[i], ang_idx=j)
                # Attenuation / other transforms that only depend on angle
                else:
                    obj_rotate = transform.forward(obj_rotate, ang_idx=angle_indices[idx][0]).unsqueeze(0).repeat(len(offsets_i),1,1,1)
            obj_translate_rot = self._translate_object(obj_rotate, offsets_i/self.object_meta.dx)
            center = int(obj_translate_rot.shape[2] / 2)
            obj_translate_cropped_rot = obj_translate_rot[:,:,center-8:center+8]
            projections[idx] = obj_translate_cropped_rot.sum(axis=1)
        return projections * self.times[angle_indices]
    
    def backward(
        self,
        proj: torch.Tensor,
        subset_idx: int | None = None
    ):
        """Applies back projection.

        Args:
            proj (torch.tensor): projections :math:`g` which are to be back projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
            return_norm_constant (bool): Whether or not to return :math:`H_m^T 1` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor: the object :math:`\hat{f} = H_m^T g_m` obtained via back projection.
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        boundary_box_bp = torch.ones(*self.object_meta.shape).to(pytomography.device)
        proj_pad = int((self.object_meta.shape[1] - self.proj_meta.shape[1]) / 2)
        object = torch.zeros(*self.object_meta.shape).to(pytomography.device)
        proj = proj * self.times[angle_indices]
        for angle in torch.unique(self.proj_meta.angles[angle_indices]):
            idx = self.proj_meta.angles[angle_indices]==angle
            offsets_i = self.proj_meta.offsets[angle_indices][idx]
            proj_i = proj[idx].unsqueeze(1)
            object_i = pad(proj_i, [0,0,proj_pad,proj_pad]) * boundary_box_bp 
            object_i = self._translate_object(object_i, -offsets_i/self.object_meta.dx)
            for transform in self.obj2obj_transforms[::-1]:
                if type(transform)==SPECTPSFTransform:
                    for i, j in enumerate(angle_indices[idx]):
                        object_i[i] = transform.forward(object_i[i], ang_idx=j)
                else:
                    object_i = transform.forward(object_i, ang_idx=angle_indices[idx][0])
            object_i = torch.stack([rotate_detector_z(o, angles=angle, negative=True) for o in object_i])
            object += object_i.sum(axis=0)
        return object
    
    def compute_normalization_factor(self, subset_idx : int | None = None):
        """Function used to get normalization factor :math:`H^T_m 1` corresponding to projection subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Index of subset. If none, then considers all projections. Defaults to None.

        Returns:
            torch.Tensor: normalization factor :math:`H^T_m 1`
        """
        norm_proj = torch.ones(*self.proj_meta.shape).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = self.get_projection_subset(norm_proj, subset_idx)
        return self.backward(norm_proj, subset_idx)
        
    def set_n_subsets(
        self,
        n_subsets: int
    ) -> list:
        """Sets the subsets for this system matrix given ``n_subsets`` total subsets.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 
        """
        indices_of_each_angle = [torch.where(self.proj_meta.angles == a)[0] for a in torch.unique(self.proj_meta.angles)]
        subset_indicies_array = []
        for i in range(n_subsets):
            subset_indicies_array.append(torch.concatenate(indices_of_each_angle[i::n_subsets]))
        self.subset_indices_array = subset_indicies_array
        
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
    
    def _translate_object(self, obj: torch.Tensor, translations: torch.Tensor):
        """Internal function that applies translations to an object with a batch size dimension.

        Args:
            obj (torch.Tensor): Object to be translated
            translations (torch.Tensor): Translations for each object in the batch

        Returns:
            torch.Tensor: Translated object
        """
        # Takes in object with batch dimension
        translation = torch.zeros(len(translations), 2).to(pytomography.device)
        translation[:,0] = translations
        obj_translated = Translate(translation)(obj.permute((0,3,1,2))).permute((0,2,3,1))
        return obj_translated
    
    