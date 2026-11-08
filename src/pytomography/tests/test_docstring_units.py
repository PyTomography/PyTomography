"""Every documented argument that is a physical quantity says its unit.

PyTomography's names carry no unit suffixes (lengths are in mm, angles in degrees, energies in keV, times in s; see
the docs' Conventions page), so the docstring is where a reader, or an AI agent, learns the unit. This test reads the
``Args:`` section of every public function, method and class of the package, and checks that each argument whose name
says it is a length, angle, energy, time, resolution or attenuation also names a unit (or says it has none, or is an
index or a count).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

import pytomography

#: Argument names that are physical quantities (whole words of the name).
PHYSICAL = re.compile(
    r"(^|_)(fwhm|sigma|sigmas|distance|distances|radius|radii|thickness|length|diameter|spacing|pitch|doi|offset|"
    r"translation|placement|tolerance|cover|centring|step|energy|energies|duration|interval|angle|angles|theta|"
    r"phi|kvp|cor|mu|dr|dx|dy|dz|resolution|frame_duration|time|times|activity|half_life|depth)($|_)")
#: Words that name a unit, or say that there is none.
UNITS = re.compile(
    r"\b(mm|cm|keV|MeV|kV|kVp|degrees?|radians?|s|seconds?|ms|ps|ns|minutes?|hours?|Bq|kBq|MBq|mL|cps|counts?|HU|"
    r"voxels?|pixels?|columns?|samples?|bins?|rings?|angles|views|steps|percent|fraction|factor|no units|unitless|"
    r"dimensionless|index|indices|number|numbers|1/mm|1/keV|keV\^\(1/2\)|mm/ps|%|sigmas|standard deviations|"
    r"cosine)\b|°|1/mm|/mm|%", re.I)
#: Documented types that are not quantities (names, flags, functions, lists of names).
NOT_QUANTITIES = re.compile(r"^\(\s*(str|bool|Callable|Sequence\[str\]|list\[str\])(?=[\s|,)\]])")
#: Modules not part of the public API, or not documented argument by argument.
SKIP_MODULES = ("pytomography.tests", "pytomography.utils._", "pytomography.io._formats",
                "pytomography.utils.memory_calibration")


def _modules():
    # a package that needs an optional dependency (parallelproj, petsird, ...) can raise any exception when it is
    # missing, not only an ImportError, which is all walk_packages skips on its own
    for info in pkgutil.walk_packages(pytomography.__path__, "pytomography.", onerror=lambda name: None):
        name = info.name
        if any(name.startswith(s) for s in SKIP_MODULES) or any(part.startswith("_") and part != "__init__"
                                                               for part in name.split(".")[1:]):
            continue
        try:
            yield importlib.import_module(name)
        except Exception:   # optional dependencies (parallelproj, petsird, ...) missing here
            continue


def _documented_args(obj) -> dict[str, str]:
    """Argument name -> its description, from a Google-style ``Args:`` section."""
    doc = inspect.getdoc(obj) or ""
    match = re.search(r"^(Args|Arguments|Parameters):\s*$(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if not match:
        return {}
    out, name = {}, None
    for line in match.group(2).splitlines():
        head = re.match(r"^    (\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:(.*)$", line)
        if head:
            name = head.group(1).lstrip("*")
            out[name] = (head.group(2) or "") + head.group(3)
        elif name and line.startswith("        "):
            out[name] += " " + line.strip()
    return out


def _public_callables():
    seen = set()
    for module in _modules():
        for attr, obj in vars(module).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != module.__name__ or id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{attr}", obj
            elif inspect.isclass(obj):
                yield f"{module.__name__}.{attr}", obj
                for meth_name, meth in vars(obj).items():
                    if inspect.isfunction(meth) and (not meth_name.startswith("_") or meth_name == "__init__"):
                        yield f"{module.__name__}.{attr}.{meth_name}", meth


def missing_units() -> list[str]:
    out = []
    for where, obj in _public_callables():
        for arg, text in _documented_args(obj).items():
            if PHYSICAL.search(arg) and not NOT_QUANTITIES.match(text.strip()) and not UNITS.search(text):
                out.append(f"{where}: {arg}: {text.strip()[:100]}")
    return out


def test_every_physical_argument_says_its_unit():
    missing = missing_units()
    assert not missing, "arguments whose docstrings name no unit:\n" + "\n".join(missing)


if __name__ == "__main__":
    for line in missing_units():
        print(line)
