"""Shared pytest configuration.

Markers (declared in pyproject.toml):
  gpu   needs a CUDA device; skipped when none is available
  data  needs the tutorial data; skipped unless PYTOMOGRAPHY_DATA points at it

CI runs ``pytest -m "not gpu and not data"`` on CPU; the GPU runner runs everything.
"""
import os
from pathlib import Path

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    no_gpu = pytest.mark.skip(reason="needs a CUDA device")
    no_data = pytest.mark.skip(reason="set PYTOMOGRAPHY_DATA to the tutorial data folder to run this test")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(no_gpu)
        if "data" in item.keywords and not os.environ.get("PYTOMOGRAPHY_DATA"):
            item.add_marker(no_data)


@pytest.fixture
def data_dir() -> Path:
    """The tutorial data folder (see "Tutorial data" in the docs)."""
    return Path(os.environ["PYTOMOGRAPHY_DATA"]).expanduser()
