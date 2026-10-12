"""``gate.get_symmetry_histogram_from_ROOTfile`` counts a calibration scan's coincidences per symmetry class of crystal pairs.
With ``include_randoms=False`` it must leave out the random coincidences, whose two photons come from different
annihilations (their source positions differ), and with ``include_randoms=True`` keep them. From April 2024 it did the
opposite, so the GATE tutorials, which ask for ``include_randoms=False``, computed their normalization with the randoms in.
No ROOT file is needed: the function only reads branches, so a small stand-in file is used."""
from __future__ import annotations

import numpy as np
import torch

from pytomography.io.PET import gate

INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)


class _Branch:
    """What uproot gives for a branch: ``.array(library='np')``."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def array(self, library='np'):
        return self.values


def _calibration_file():
    """Six coincidences: four trues (both photons from one annihilation) and, last, two randoms."""
    x2 = [1.0, 2.0, 3.0, 4.0, 50.0, -60.0]   # the randoms' second photons come from elsewhere
    branches = dict(rsectorID1=[0, 1, 2, 3, 4, 5], rsectorID2=[8, 9, 10, 11, 12, 13],
                    moduleID1=[0, 1, 0, 1, 0, 1], moduleID2=[1, 0, 1, 0, 1, 0],
                    submoduleID1=[0] * 6, submoduleID2=[0] * 6,
                    crystalID1=[0, 5, 10, 15, 3, 7], crystalID2=[1, 6, 11, 14, 2, 9],
                    sourcePosX1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], sourcePosX2=x2,
                    sourcePosY1=[0.0] * 6, sourcePosY2=[0.0] * 6, sourcePosZ1=[0.0] * 6, sourcePosZ2=[0.0] * 6)
    return {'Coincidences': {name: _Branch(values) for name, values in branches.items()}}


def test_include_randoms_false_leaves_the_randoms_out():
    f = _calibration_file()
    with_randoms = gate.get_symmetry_histogram_from_ROOTfile(f, INFO, include_randoms=True)
    without_randoms = gate.get_symmetry_histogram_from_ROOTfile(f, INFO, include_randoms=False)
    assert with_randoms.shape[0] == 6
    assert without_randoms.shape[0] == 4
    assert torch.equal(without_randoms, with_randoms[:4])     # the trues, in their order


def test_default_keeps_the_randoms():
    f = _calibration_file()
    assert torch.equal(gate.get_symmetry_histogram_from_ROOTfile(f, INFO), gate.get_symmetry_histogram_from_ROOTfile(f, INFO, include_randoms=True))
