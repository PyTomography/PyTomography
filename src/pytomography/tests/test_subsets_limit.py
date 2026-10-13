"""The number of subsets has a limit: each subset of a partition by view needs at least one view. Partitioning into
more subsets than views stops with a message, before a reconstruction starts (an empty subset made a GPU projector
fail with "invalid argument"), and subsets_for_budget says what budget fits when the most there can be do not."""
from __future__ import annotations

import math

import pytest

import pytomography
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors import SystemMatrix
from pytomography.utils import memory

from test_contracts_spect import NA, tiny_spect_system


@pytest.fixture
def budget():
    yield pytomography.set_memory_budget
    pytomography.set_memory_budget(None)


def test_check_n_subsets():
    SystemMatrix._check_n_subsets(1, 224)
    SystemMatrix._check_n_subsets(224, 224)
    for n in (0, 225, 277):
        with pytest.raises(ValueError, match="use 1 to 224 subsets"):
            SystemMatrix._check_n_subsets(n, 224)


def test_spect_partition_stops_at_the_number_of_projections():
    H, f, g = tiny_spect_system(attenuation=False, psf=False)
    H.set_n_subsets(NA)
    assert len(H.subset_indices_array) == NA and all(len(s) == 1 for s in H.subset_indices_array)
    with pytest.raises(ValueError, match=f"{NA + 1} subsets of {NA} projections"):
        H.set_n_subsets(NA + 1)
    with pytest.raises(ValueError):                                 # stops before the reconstruction starts
        OSEM(PoissonLogLikelihood(H, g))(n_iters=1, n_subsets=NA + 1)


def test_subsets_for_budget_stops_at_its_maximum(budget):
    budget(8)                                                       # half of it, 4 GB, for the subsets
    assert memory.subsets_for_budget(34.6e9) == 26                  # three 34.6 GB / 26 arrays fit in 4 GB
    assert memory.subsets_for_budget(34.6e9, maximum=26) == 26
    with pytest.raises(ValueError, match="at most 20: raise it to at least 11 GB") as error:
        memory.subsets_for_budget(34.6e9, maximum=20)
    least = int(str(error.value).split("at least ")[1].split(" GB")[0])
    budget(least)                                                   # the budget it asks for is enough
    assert memory.subsets_for_budget(34.6e9, maximum=20) <= 20
    # with the memory held given, as the TOF sinogram tutorial gives it: 224 angles at most
    budget(16)
    with pytest.raises(ValueError, match="at most 224") as error:
        memory.subsets_for_budget(34.6e9, arrays=4, held_bytes=15.5e9, minimum=28, maximum=224)
    least = int(str(error.value).split("at least ")[1].split(" GB")[0])
    assert least == math.ceil((15.5e9 + 4 * 34.6e9 / 224) / 1e9)
    budget(least)
    assert memory.subsets_for_budget(34.6e9, arrays=4, held_bytes=15.5e9, minimum=28, maximum=224) <= 224


def test_set_n_subsets_sets_the_partition_and_returns_none():
    """Every system matrix's set_n_subsets is annotated as returning None, as it does (they said list), and sets
    subset_indices_array: every view in exactly one subset."""
    import importlib
    import inspect
    import torch
    classes = set()
    for name in ('system_matrix', 'SPECT', 'PET', 'CT', 'shared'):
        try:
            module = importlib.import_module(f'pytomography.projectors.{name}')
        except ImportError:                       # CT needs parallelproj 2
            continue
        classes.update(c for c in vars(module).values() if inspect.isclass(c) and issubclass(c, SystemMatrix))
    with_subsets = [c for c in classes if 'set_n_subsets' in vars(c)]
    assert len(with_subsets) >= 7
    for cls in with_subsets:
        assert inspect.signature(cls.set_n_subsets).return_annotation in (None, 'None'), cls.__name__
    system_matrix = tiny_spect_system()[0]
    assert system_matrix.set_n_subsets(3) is None
    views = torch.cat([s.cpu() for s in system_matrix.subset_indices_array])
    assert len(system_matrix.subset_indices_array) == 3 and sorted(views.tolist()) == list(range(NA))
