from __future__ import annotations
import torch
import torch.nn.functional as F
import pytomography
from pytomography.utils import rotate_detector_z
from pytomography.projectors.system_matrix import SystemMatrix
from pytomography.transforms.SPECT import SPECTPSFTransform
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import StarGuideProjMeta
from kornia.geometry.transform import Translate

DETECTOR_WIDTH = 16   # transaxial pixels of a StarGuide detector head

class StarGuideSystemMatrix(SystemMatrix):
    r"""System matrix for the StarGuide SPECT imaging system form General Electric Healthcare.

    Each view is a 16 pixel wide detector head at a projection angle and a transaxial offset. Forward projection rotates the object to the angle (shared by all views at that angle) and applies the angle-dependent object transforms (e.g. attenuation); then each view blurs the object with its PSF, translates it by its offset (bilinear interpolation) and sums the 16 columns under the head along the projection axis. A view only ever samples the columns under the head plus the reach of its PSF kernel, so the blur and the translation are evaluated on a slab of that width gathered from the object, with all views of an angle batched: one grouped convolution per blurring axis and one sampling call per angle. Back projection is the adjoint of the same operations, with the views of an angle summed before the shared transforms and the rotation are applied. The angle-dependent (non-PSF) transforms are applied before the PSF transforms regardless of their order in ``obj2obj_transforms``.

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
        self._groups_cache = {}

    # ------------------------------------------------------------------ helpers
    def _psf_transforms(self) -> list:
        return [t for t in self.obj2obj_transforms if type(t) == SPECTPSFTransform]

    def _shared_transforms(self) -> list:
        """Transforms that depend on the angle only (all views at an angle share them)."""
        return [t for t in self.obj2obj_transforms if type(t) != SPECTPSFTransform]

    def _psf_margin(self) -> int:
        """Half-width of the widest PSF kernel along the detector axis over all views: blurring a slab that extends this far beyond the sampled columns gives the same values as blurring the whole object. The whole width is used if a PSF layer is not of the separable kind."""
        margin = 0
        for transform in self._psf_transforms():
            for layer in transform.layers.values():
                layer_r = getattr(layer, 'layer_r', None)
                margin = self.object_meta.shape[1] if layer_r is None else max(margin, int(layer_r.kernel_size[0]) // 2)
        return margin

    def _view_groups(self, subset_idx: int | None):
        """Views of a subset grouped by projection angle, with everything the projection loops need prepared once so that they never synchronise with the device. For each angle (ascending): the angle as a 0-d CPU tensor (the rotation angle is derived from it in the tensor's dtype, as before), the positions of its views in the subset, their global view indices (PSF radius, transform ``ang_idx``), their translations in voxels and the slab geometry (see ``_slab_geometry``).

        Returns:
            tuple: (device index tensor of the subset's views, list of groups)
        """
        if subset_idx in self._groups_cache:
            return self._groups_cache[subset_idx]
        if subset_idx is None:
            idx = torch.arange(self.proj_meta.num_projections, device=pytomography.device)
        else:
            idx = self.subset_indices_array[subset_idx].to(pytomography.device)
        idx_cpu = idx.cpu()
        angles_cpu = self.proj_meta.angles.cpu()[idx_cpu]
        translations = self.proj_meta.offsets[idx] / self.object_meta.dx
        groups = []
        for angle in torch.unique(angles_cpu):
            pos = torch.nonzero(angles_cpu == angle).flatten()
            pos_device = pos.to(pytomography.device)
            group = dict(angle=angle, pos=pos_device, gidx=idx_cpu[pos].tolist(), translations=translations[pos_device], psf_kernels={})
            group.update(self._slab_geometry(group['translations']))
            groups.append(group)
        self._groups_cache[subset_idx] = (idx, groups)
        return self._groups_cache[subset_idx]

    def _slab_geometry(self, translations: torch.Tensor) -> dict:
        """Slab of object columns each view needs, and the sampling grids between the slab and the 16 detector columns.

        Detector column ``c`` of a view translated by ``s`` voxels samples the (blurred) object at column ``c - s`` (as ``kornia.geometry.transform.Translate`` does, bilinear, zero outside the object). The slab of a view therefore starts ``margin`` columns before ``floor(c_first - s)`` and is wide enough to hold the sampled columns and the PSF reach on both sides. ``cols`` are the object columns of the slab (clamped) and ``valid`` marks the ones inside the object, ``grid_fwd`` samples the detector columns from the slab and ``grid_bwd`` is its transpose (the slab columns sampled from the 16 detector columns).

        Args:
            translations (torch.Tensor): [B] translations of the views in voxels.

        Returns:
            dict: ``cols`` [B, W], ``valid`` [B, 1, W, 1], ``grid_fwd`` [B, Lx, 16, 2], ``grid_bwd`` [B, Lx, W, 2].
        """
        device, dtype = translations.device, translations.dtype
        B = translations.shape[0]
        Lx, Ly = self.object_meta.shape[0], self.object_meta.shape[1]
        margin = self._psf_margin()
        c_first = int(Ly / 2) - DETECTOR_WIDTH // 2
        W = DETECTOR_WIDTH + 2 + 2 * margin
        a0 = torch.floor(c_first - translations).to(torch.long) - margin                      # [B] first slab column
        cols = a0.unsqueeze(1) + torch.arange(W, device=device)                                # [B, W]
        valid = ((cols >= 0) * (cols < Ly)).to(dtype).reshape(B, 1, W, 1)
        rows = (2 * torch.arange(Lx, device=device, dtype=dtype) / (Lx - 1) - 1).reshape(1, Lx, 1)
        x_fwd = (c_first + torch.arange(DETECTOR_WIDTH, device=device, dtype=dtype)).unsqueeze(0) - translations.unsqueeze(1) - a0.unsqueeze(1).to(dtype)   # [B, 16] slab coordinate of each detector column
        grid_fwd = torch.empty((B, Lx, DETECTOR_WIDTH, 2), device=device, dtype=dtype)
        grid_fwd[..., 0] = (2 * x_fwd / (W - 1) - 1).unsqueeze(1)
        grid_fwd[..., 1] = rows
        x_bwd = torch.arange(W, device=device, dtype=dtype).unsqueeze(0) + a0.unsqueeze(1).to(dtype) + translations.unsqueeze(1) - c_first   # [B, W] detector coordinate of each slab column
        grid_bwd = torch.empty((B, Lx, W, 2), device=device, dtype=dtype)
        grid_bwd[..., 0] = (2 * x_bwd / (DETECTOR_WIDTH - 1) - 1).unsqueeze(1)
        grid_bwd[..., 1] = rows
        return dict(cols=cols.clamp(0, Ly - 1), valid=valid, grid_fwd=grid_fwd, grid_bwd=grid_bwd)

    def _psf_kernels(self, transform: SPECTPSFTransform, group: dict):
        """The PSF kernels of the views of a group, concatenated (zero padded to a common width) into the weights of one grouped 1D convolution per blurring axis, so that all views of the group are blurred by a single convolution. Returns None if the transform's layers are not the separable ``Seperable1DBlurNet`` kind (the views are then blurred one by one)."""
        cache = group['psf_kernels']
        if id(transform) in cache:
            return cache[id(transform)]
        layers = [transform.layers[transform.proj_meta.radii[j]] for j in group['gidx']]
        Lx = self.object_meta.shape[0]
        result = None
        if all(getattr(l, 'layer_r', None) is not None for l in layers) and all(l.layer_r.weight.shape[0] == Lx for l in layers):
            weights = []
            for axis in ('layer_r', 'layer_z'):
                convs = [getattr(l, axis) for l in layers]
                if any(c is None for c in convs):
                    weights.append(None)
                    continue
                k_max = max(c.weight.shape[-1] for c in convs)
                w = torch.zeros((len(layers) * Lx, 1, k_max), device=convs[0].weight.device, dtype=convs[0].weight.dtype)
                for i, c in enumerate(convs):
                    k = c.weight.shape[-1]
                    w[i * Lx:(i + 1) * Lx, :, (k_max - k) // 2:(k_max - k) // 2 + k] = c.weight.data
                weights.append(w)
            result = tuple(weights)
        cache[id(transform)] = result
        return result

    def _apply_psf(self, slabs: torch.Tensor, transform: SPECTPSFTransform, group: dict) -> torch.Tensor:
        """Applies ``transform`` to the slabs of all views of a group ([B, Lx, W, Lz], view ``i`` uses the PSF of view ``group['gidx'][i]``): one grouped convolution per axis over all views, or view by view if the kernels cannot be batched."""
        B, Lx, W, Lz = slabs.shape
        kernels = self._psf_kernels(transform, group)
        if kernels is None:
            return torch.stack([transform.forward(slabs[i], ang_idx=j) for i, j in enumerate(group['gidx'])])
        weight_r, weight_z = kernels
        x = slabs.permute(3, 0, 1, 2).reshape(Lz, B * Lx, W)                        # batch: z, channel: (view, plane along projection axis), length: detector axis
        x = F.conv1d(x, weight_r, padding='same', groups=B * Lx)
        if weight_z is None:
            return x.reshape(Lz, B, Lx, W).permute(1, 2, 3, 0)
        x = x.reshape(Lz, B, Lx, W).permute(3, 1, 2, 0).reshape(W, B * Lx, Lz)       # batch: detector axis, length: z
        x = F.conv1d(x, weight_z, padding='same', groups=B * Lx)
        return x.reshape(W, B, Lx, Lz).permute(1, 2, 0, 3)

    # ------------------------------------------------------------------ projections
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
        idx, groups = self._view_groups(subset_idx)
        Lx, Ly, Lz = self.object_meta.shape
        psf_transforms, shared_transforms = self._psf_transforms(), self._shared_transforms()
        projections = torch.zeros((idx.shape[0], *self.proj_meta.shape[1:]), device=pytomography.device)
        for group in groups:
            B, W = group['cols'].shape
            obj_rotate = rotate_detector_z(object, angles=group['angle'])
            for transform in shared_transforms:
                obj_rotate = transform.forward(obj_rotate, ang_idx=group['gidx'][0])
            # the columns each view needs (zero outside the object), blurred with its PSF (zero beyond the object, as when blurring the whole object)
            slabs = obj_rotate.index_select(1, group['cols'].flatten()).reshape(Lx, B, W, Lz).permute(1, 0, 2, 3) * group['valid']
            for transform in psf_transforms:
                slabs = self._apply_psf(slabs, transform, group) * group['valid']
            # translate: sample the detector columns from the slabs, then sum along the projection axis
            sampled = F.grid_sample(slabs.permute(0, 3, 1, 2).contiguous(), group['grid_fwd'], mode='bilinear', padding_mode='zeros', align_corners=True)   # [B, Lz, Lx, 16]
            projections[group['pos']] = sampled.sum(dim=2).permute(0, 2, 1)
        return projections * self.times[idx]

    @torch.no_grad()
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
        idx, groups = self._view_groups(subset_idx)
        Lx, Ly, Lz = self.object_meta.shape
        psf_transforms, shared_transforms = self._psf_transforms(), self._shared_transforms()
        object = torch.zeros((Lx, Ly, Lz), device=pytomography.device)
        proj = proj * self.times[idx]
        for group in groups:
            B, W = group['cols'].shape
            # every plane along the projection axis receives the view; transpose of the translation: sample the slab columns from the detector columns
            views = proj[group['pos']].permute(0, 2, 1).unsqueeze(2).expand(B, Lz, Lx, DETECTOR_WIDTH).contiguous()
            slabs = F.grid_sample(views, group['grid_bwd'], mode='bilinear', padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1) * group['valid']   # [B, Lx, W, Lz]
            for transform in psf_transforms[::-1]:                                        # symmetric kernels: the transpose is the same blur
                slabs = self._apply_psf(slabs, transform, group) * group['valid']
            group_object = torch.zeros((Lx, Ly, Lz), device=pytomography.device)
            group_object.index_add_(1, group['cols'].flatten(), slabs.permute(1, 0, 2, 3).reshape(Lx, B * W, Lz))
            for transform in shared_transforms[::-1]:
                group_object = transform.forward(group_object, ang_idx=group['gidx'][0])
            object += rotate_detector_z(group_object, angles=group['angle'], negative=True)
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
        self._groups_cache = {}

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
