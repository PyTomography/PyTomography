"""PyTomography's proton CT list mode file: one HDF5 file holding a :class:`~pytomography.metadata.PCT.PCTProjMeta` and
each proton's WEPL, whatever scanner or simulation the data came from (each converts into it once, outside
PyTomography)."""
from __future__ import annotations
import math
import pathlib
from typing import Sequence
import numpy as np
import torch
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PCT import PCTProjMeta

#: The ``format`` attribute of the file, and its version.
FORMAT, VERSION = 'pytomography-pct-listmode', 1
#: Each dataset and its units (stored as the dataset's ``units`` attribute).
DATASETS = {'angles': 'degrees', 'front': 'mm', 'rear': 'mm', 'view_sizes': 'protons',
            'entry': 'x mm, z mm, dx/dy, dz/dy', 'exit': 'x mm, z mm, dx/dy, dz/dy', 'wepl': 'mm', 'entry_fwhm': 'mm'}


def save_listmode(path: str | pathlib.Path, projections: torch.Tensor, proj_meta: PCTProjMeta) -> None:
    """Write protons and their WEPLs as a PyTomography proton CT list mode file (HDF5).

    The file holds the datasets of :data:`DATASETS` (``angles``, ``front``, ``rear``, ``view_sizes`` per view;
    ``entry``, ``exit`` [N, 4], ``wepl`` [N] and, when each proton has its own, ``entry_fwhm`` [N] per proton, view by
    view), each with its ``units``, and the attributes ``format``, ``version``, ``beam_energy`` (keV),
    ``beam_divergence`` (degrees per mm), ``z_center`` (mm) and, when one value holds for every proton,
    ``entry_fwhm`` (mm). See :class:`~pytomography.metadata.PCT.PCTProjMeta` for what each means.

    Args:
        path (str | pathlib.Path): The file to write (``.h5``).
        projections (torch.Tensor): Each proton's WEPL, [N], in mm.
        proj_meta (PCTProjMeta): The protons.
    """
    import h5py
    if tuple(projections.shape) != proj_meta.shape:
        raise ValueError(f'{tuple(projections.shape)} WEPLs for {proj_meta.N_protons} protons')
    arrays = {'angles': proj_meta.angles.numpy(), 'front': proj_meta.front, 'rear': proj_meta.rear,
              'view_sizes': proj_meta.view_sizes, 'entry': proj_meta.entry.numpy(), 'exit': proj_meta.exit.numpy(),
              'wepl': projections.detach().to('cpu', torch.float32).numpy()}
    with h5py.File(path, 'w') as f:
        f.attrs.update(format=FORMAT, version=VERSION, beam_energy=proj_meta.beam_energy,
                       beam_divergence=proj_meta.beam_divergence, z_center=proj_meta.z_center)
        if isinstance(proj_meta.entry_fwhm, torch.Tensor):
            arrays['entry_fwhm'] = proj_meta.entry_fwhm.numpy()
        else:
            f.attrs['entry_fwhm'] = proj_meta.entry_fwhm
        for name, a in arrays.items():
            f.create_dataset(name, data=a).attrs['units'] = DATASETS[name]


def load_listmode(
    path: str | pathlib.Path,
    dr: float | Sequence[float] = 1.0,
    shape: Sequence[int] | None = None,
    views: Sequence[int] | None = None,
    fraction: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, ObjectMeta, PCTProjMeta]:
    """Read a PyTomography proton CT list mode file (:func:`save_listmode`) into the standard form ``(projections,
    object_meta, proj_meta)``: each proton's WEPL, an object grid centred on the rotation axis and on the file's
    ``z_center``, and the protons. With ``views`` or ``fraction`` the file is read a view at a time, so that only the
    protons kept are held in memory (40 bytes each with their WEPLs and spreads).

    Args:
        path (str | pathlib.Path): The file.
        dr (float | Sequence[float], optional): Voxel size of the object grid, in mm. Defaults to 1.
        shape (Sequence[int] | None, optional): Shape of the object grid (voxels). Defaults to None: x and y cover
            every proton's straight line (an odd number of voxels), z the protons' axial range.
        views (Sequence[int] | None, optional): The views to read (indices). Defaults to None: all.
        fraction (float, optional): Fraction of each view's protons to keep, chosen at random (no units). Defaults
            to 1.
        seed (int, optional): Seed of that choice (view ``v`` uses ``seed + 7919 v``). Defaults to 0.

    Returns:
        tuple[torch.Tensor, ObjectMeta, PCTProjMeta]: The WEPLs (float32, mm), the object grid and the protons.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        if f.attrs.get('format') != FORMAT:
            raise ValueError(f'{path} is not a {FORMAT} file')
        if int(f.attrs['version']) > VERSION:
            raise ValueError(f'{path} is version {int(f.attrs["version"])}; this PyTomography reads up to {VERSION}')
        sizes = f['view_sizes'][()]
        views = np.arange(len(sizes)) if views is None else np.asarray(views, dtype=np.int64)
        per_proton = 'entry_fwhm' in f
        if fraction >= 1 and np.array_equal(views, np.arange(len(sizes))):
            entry, exit, wepl = (torch.from_numpy(f[k][()]) for k in ('entry', 'exit', 'wepl'))
            fwhm = torch.from_numpy(f['entry_fwhm'][()]) if per_proton else float(f.attrs['entry_fwhm'])
            kept = sizes
        else:
            offsets = np.concatenate([[0], np.cumsum(sizes)])
            parts, kept = [], []
            for v in views:
                lo, hi = int(offsets[v]), int(offsets[v + 1])
                keep = slice(None) if fraction >= 1 else (
                    torch.rand(hi - lo, generator=torch.Generator().manual_seed(seed + 7919 * int(v))) < fraction).numpy()
                names = ('entry', 'exit', 'wepl') + (('entry_fwhm',) if per_proton else ())
                parts.append([torch.from_numpy(f[k][lo:hi][keep]) for k in names])
                kept.append(len(parts[-1][2]))
            entry, exit, wepl = (torch.cat([p[i] for p in parts]) for i in range(3))
            fwhm = torch.cat([p[3] for p in parts]) if per_proton else float(f.attrs['entry_fwhm'])
            del parts
        proj_meta = PCTProjMeta(f['angles'][()][views], f['front'][()][views], f['rear'][()][views], entry, exit,
                                kept, float(f.attrs['beam_energy']), entry_fwhm=fwhm,
                                beam_divergence=float(f.attrs['beam_divergence']), z_center=float(f.attrs['z_center']))
    if shape is None:
        # x and y: every straight line's distance from the axis (the same in every frame); z: the protons' range
        x0, x1 = entry[:, 0].double(), exit[:, 0].double()
        fr = torch.as_tensor(np.repeat(proj_meta.front, kept))
        re = torch.as_tensor(np.repeat(proj_meta.rear, kept))
        cross = (x0 * (re - fr) - fr * (x1 - x0)).abs() / torch.hypot(x1 - x0, re - fr)
        r_fov = float(torch.quantile(cross[::max(1, len(cross) // 10**6)], 0.9999))
        zs = torch.cat([entry[::97, 1], exit[::97, 1]]).double().abs()
        dr3 = np.broadcast_to(np.asarray(dr, dtype=np.float64), (3,))
        shape = (2 * math.ceil(r_fov / dr3[0]) + 1, 2 * math.ceil(r_fov / dr3[1]) + 1,
                 max(1, 2 * math.ceil(float(torch.quantile(zs, 1 - 1e-4)) / dr3[2])))
    object_meta = ObjectMeta(dr=dr, shape=shape, centre=(0.0, 0.0, proj_meta.z_center), frame='scanner', units='RSP')
    return wepl, object_meta, proj_meta
