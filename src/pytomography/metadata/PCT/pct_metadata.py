"""Projection metadata of proton CT (pCT) list mode data: one projection per proton."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import torch
from pytomography.metadata import ProjMeta

#: FWHM of a Gaussian over its standard deviation.
FWHM_PER_SD = 2 * math.sqrt(2 * math.log(2))
#: A proton's band index (:func:`pytomography.projectors.PCT.band_paths`) is its rank in its view plus this times the
#: view, so that a view's band is the same whichever views it is loaded with.
BAND_INDEX_STRIDE = 1_000_003


class PCTProjMeta(ProjMeta):
    r"""Projection space of proton CT list mode data: one projection per proton, its water-equivalent path length
    (WEPL, :math:`\int \mathrm{RSP}\,dl` along its path, in mm), so that a reconstruction is the relative stopping
    power (RSP, no units).

    Each view is a beam along +y of its own frame, with the object turned by ``angles`` about +z, and two planes
    across the beam: the entry plane at ``y = front`` and the exit plane at ``y = rear``. A proton is where it crossed
    each plane and its direction there, as four numbers per plane: ``x``, ``z`` and the slopes ``dx/dy``, ``dz/dy``.
    That is what a pair of trackers measures (the CREATIS PCT toolkit's list mode pairs hold the same, as positions
    and unit vectors). Without a front tracker the entry is the pencil beam's ray from the beam delivery log, and
    ``entry_fwhm`` says how far the proton may have been from it: the entry is then Gaussian about the stored point,
    of that FWHM in x and in z, and its direction turns by ``beam_divergence`` per mm of offset (the beam's
    position-angle correlation). With a front tracker ``entry_fwhm`` is 0.

    The protons are stored view by view (``view_sizes``), in float32 tables of 16 bytes per plane, so that a scan of
    188 million protons takes 6.8 GB with its WEPLs; :func:`~pytomography.io.PCT.save_listmode` and
    :func:`~pytomography.io.PCT.load_listmode` write and read it as one file.
    Positions are in the object frame's units and axes: x and z along the object's x and z when the angle is 0, z
    relative to ``z_center``; :meth:`load` turns a view into the object frame (LPS: x towards the patient's left, y
    posterior, z superior), centred on the rotation axis and on ``z_center``.

    Args:
        angles (Sequence[float]): Rotation of the object about +z in each view, in degrees (equivalently, the beam
            turned by minus the angle).
        front (float | Sequence[float]): Entry plane of each view (or all of them), as its y in the view's frame, in
            mm from the rotation axis (negative: before the axis).
        rear (float | Sequence[float]): Exit plane of each view, in mm from the rotation axis.
        entry (torch.Tensor): Each proton at the entry plane: x and z (mm), and the slopes dx/dy and dz/dy (no units),
            [N, 4], view by view.
        exit (torch.Tensor): The same at the exit plane, [N, 4].
        view_sizes (Sequence[int]): Number of protons of each view, in the order they are stored.
        beam_energy (float): Kinetic energy of the protons entering the object region, in keV.
        entry_fwhm (float | torch.Tensor, optional): FWHM of each proton's entry about its stored entry point, in x and
            in z, in mm: one for every proton or [N]. Defaults to 0: the entry is measured.
        beam_divergence (float, optional): Turn of the entry direction per mm of lateral offset from the stored entry
            point, in degrees per mm. Defaults to 0.
        z_center (float, optional): Axial position of the frame's origin along the rotation axis, in mm in the
            scanner's frame (the isocentre at 0). Defaults to 0.

    Attributes:
        shape (tuple): (N,): one projection per proton.
        N_angles (int): Number of views.
        N_protons (int): Number of protons.
        view_offsets (np.ndarray): Where each view starts in the tables, [N_angles + 1].
    """
    def __init__(
        self,
        angles: Sequence[float],
        front: float | Sequence[float],
        rear: float | Sequence[float],
        entry: torch.Tensor,
        exit: torch.Tensor,
        view_sizes: Sequence[int],
        beam_energy: float,
        entry_fwhm: float | torch.Tensor = 0.0,
        beam_divergence: float = 0.0,
        z_center: float = 0.0,
    ) -> None:
        super().__init__(torch.as_tensor(np.asarray(angles, dtype=np.float64)))
        self.N_angles = len(self.angles)
        self.view_sizes = np.asarray(view_sizes, dtype=np.int64)
        self.view_offsets = np.concatenate([[0], np.cumsum(self.view_sizes)])
        self.N_protons = int(self.view_offsets[-1])
        self.shape = (self.N_protons,)
        for name, a in (('entry', entry), ('exit', exit)):
            if tuple(a.shape) != (self.N_protons, 4):
                raise ValueError(f'{name} is {tuple(a.shape)}, expected ({self.N_protons}, 4) from view_sizes')
        if len(self.view_sizes) != self.N_angles:
            raise ValueError(f'{len(self.view_sizes)} view sizes for {self.N_angles} angles')
        self.front = np.broadcast_to(np.asarray(front, dtype=np.float64), (self.N_angles,)).copy()
        self.rear = np.broadcast_to(np.asarray(rear, dtype=np.float64), (self.N_angles,)).copy()
        self.entry = entry.to(torch.float32).contiguous()
        self.exit = exit.to(torch.float32).contiguous()
        if isinstance(entry_fwhm, torch.Tensor):
            if tuple(entry_fwhm.shape) != (self.N_protons,):
                raise ValueError(f'entry_fwhm is {tuple(entry_fwhm.shape)}, expected ({self.N_protons},) or a number')
            entry_fwhm = entry_fwhm.to(torch.float32).contiguous()
        else:
            entry_fwhm = float(entry_fwhm)
        self.entry_fwhm = entry_fwhm
        self.beam_divergence = float(beam_divergence)
        self.beam_energy = float(beam_energy)
        self.z_center = float(z_center)

    def replace(self, **changes) -> 'PCTProjMeta':
        """A copy with some of the arguments changed (the tables not given are shared, not copied).

        Args:
            **changes: Arguments of :class:`PCTProjMeta` by name (``entry``, ``entry_fwhm``, ...).

        Returns:
            PCTProjMeta: The copy.
        """
        args = dict(angles=self.angles.numpy(), front=self.front, rear=self.rear, entry=self.entry, exit=self.exit,
                    view_sizes=self.view_sizes, beam_energy=self.beam_energy, entry_fwhm=self.entry_fwhm,
                    beam_divergence=self.beam_divergence, z_center=self.z_center)
        return PCTProjMeta(**{**args, **changes})

    def load(self, view: int, device, start: int = 0, stop: int | None = None) -> dict:
        """Protons ``start`` to ``stop`` of a view, in the object frame, on ``device``.

        Args:
            view (int): Index of the view.
            device (str | torch.device): Where to put them.
            start (int, optional): First proton, counted from the view's first (an index). Defaults to 0.
            stop (int | None, optional): One past the last proton (an index). Defaults to None: the view's end.

        Returns:
            dict: ``entry_points``, ``exit_points`` [n, 3] (mm) and ``entry_directions``, ``exit_directions`` [n, 3]
            (unit vectors), ``entry_fwhm`` [n] (mm), ``angle`` [n] (the view's angle, in radians, for trigonometry),
            and ``index`` [n] (the protons' band indices, :data:`BAND_INDEX_STRIDE`).
        """
        stop = int(self.view_sizes[view]) if stop is None else int(stop)
        lo, hi = int(self.view_offsets[view]) + int(start), int(self.view_offsets[view]) + stop
        n = hi - lo
        t_in = self.entry[lo:hi].to(device, non_blocking=True)
        t_out = self.exit[lo:hi].to(device, non_blocking=True)
        a = torch.full((n,), math.radians(float(self.angles[view])), dtype=torch.float32, device=device)
        c, s = torch.cos(a), torch.sin(a)

        def obj(x, y, z):                       # the view's frame to the object frame: turn by minus the angle
            return torch.stack([c * x + s * y, -s * x + c * y, z], dim=1)

        def direction(sx, sz):
            norm = torch.rsqrt(1 + sx * sx + sz * sz)
            return obj(sx * norm, norm, sz * norm)
        fy = torch.full((n,), float(self.front[view]), dtype=torch.float32, device=device)
        ry = torch.full((n,), float(self.rear[view]), dtype=torch.float32, device=device)
        fwhm = self.entry_fwhm
        fwhm = (fwhm[lo:hi].to(device, non_blocking=True) if isinstance(fwhm, torch.Tensor)
                else torch.full((n,), fwhm, dtype=torch.float32, device=device))
        index = BAND_INDEX_STRIDE * int(view) + torch.arange(int(start), stop, device=device)
        return {'entry_points': obj(t_in[:, 0], fy, t_in[:, 1]), 'entry_directions': direction(t_in[:, 2], t_in[:, 3]),
                'exit_points': obj(t_out[:, 0], ry, t_out[:, 1]), 'exit_directions': direction(t_out[:, 2], t_out[:, 3]),
                'entry_fwhm': fwhm, 'angle': a, 'index': index}

    def nbytes(self) -> int:
        """Bytes the protons' tables take in host memory (without their WEPLs)."""
        fwhm = self.entry_fwhm.numel() * 4 if isinstance(self.entry_fwhm, torch.Tensor) else 0
        return 4 * (self.entry.numel() + self.exit.numel()) + fwhm

    def __repr__(self):
        fwhm = 'per proton' if isinstance(self.entry_fwhm, torch.Tensor) else f'{self.entry_fwhm:g} mm'
        return (f'{type(self).__name__}({self.N_protons:,} protons, {self.N_angles} views, beam {self.beam_energy:g} keV, '
                f'entry FWHM {fwhm}, z_center {self.z_center:g} mm)')
