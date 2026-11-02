"""Proton CT list mode data in PyTomography's standard form ``(projections, object_meta, proj_meta)``, the projections
being each proton's water-equivalent path length (WEPL, mm):

* :func:`load_listmode` reads PyTomography's proton CT list mode file (HDF5), and :func:`save_listmode` writes one.

Each scanner or simulation is converted into that file once, outside PyTomography, with its own calibrations (WEPL
from energy or range, cuts, entries estimated from the beam for a scanner without a front tracker).
"""
from .listmode import load_listmode, save_listmode
