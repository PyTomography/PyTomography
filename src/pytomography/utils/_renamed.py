"""Arguments renamed in PyTomography 4 that still work, with a DeprecationWarning, until PyTomography 5.

PyTomography's arguments carry no unit suffixes: every length is in mm, every angle in degrees, every energy in keV
and every time in s (see the docs' Conventions page), and every Gaussian is given by its FWHM. A few names of
PyTomography 3 changed for that; :func:`renamed_arguments` keeps them working, as CONTRIBUTING.md's deprecation
policy asks.
"""
from __future__ import annotations

import functools
import math
import warnings
from typing import Callable

#: The FWHM of a Gaussian over its standard deviation, :math:`2\sqrt{2 \ln 2}`.
FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))


def fwhm_from_sigma(sigma: float) -> float:
    """The FWHM of a Gaussian of standard deviation ``sigma`` (same units)."""
    return sigma * FWHM_PER_SIGMA


def sigma_from_fwhm(fwhm: float) -> float:
    """The standard deviation of a Gaussian of FWHM ``fwhm`` (same units)."""
    return fwhm / FWHM_PER_SIGMA


def renamed_arguments(**renames: str | tuple[str, Callable, str]) -> Callable:
    """Decorator: keyword arguments ``old=...`` still work, as ``new``, with a DeprecationWarning.

    Args:
        **renames: ``old="new"``, or ``old=("new", convert, note)`` when the value changes too (e.g. a sigma that is
            now a FWHM): ``convert`` turns the old value into the new one, and ``note`` says how in the warning.
    """
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for old, spec in renames.items():
                if old not in kwargs:
                    continue
                new, convert, note = (spec, None, "") if isinstance(spec, str) else spec
                if new in kwargs:
                    raise TypeError(f"{function.__qualname__}() got both {old!r} and its new name {new!r}")
                value = kwargs.pop(old)
                kwargs[new] = convert(value) if convert is not None else value
                given = f"{new}={kwargs[new]:g}" if isinstance(kwargs[new], (int, float)) else new
                warnings.warn(f"{function.__qualname__}(): {old!r} is now {new!r}{note}; {old}={value!r} was used as "
                              f"{given}. The old name will be removed in PyTomography 5.0.", DeprecationWarning,
                              stacklevel=2)
            return function(*args, **kwargs)
        return wrapper
    return decorate


def sigma_renamed(new: str, units: str) -> tuple[str, Callable, str]:
    """The ``renamed_arguments`` entry for a Gaussian standard deviation that is now a FWHM."""
    return new, fwhm_from_sigma, f", the FWHM in {units} (2.355 times the old standard deviation)"
