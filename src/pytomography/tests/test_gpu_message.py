"""PyTomography says at import when a machine's NVIDIA GPU goes unused: a broken install runs on the CPU without
failing, many times slower, and is easily taken for PyTomography's normal speed. PET and CT also say when parallelproj
is its CPU build on a GPU machine."""
import os
import subprocess
import sys

import pytest
import torch

import pytomography
from pytomography import _device_message, _parallelproj_message

CPU = torch.device("cpu")


def test_nothing_to_say_on_a_cuda_gpu():
    assert _device_message(torch.device("cuda"), "nvidia-smi", None) is None


def test_an_unused_nvidia_gpu_is_a_warning_with_the_docs_link():
    message = _device_message(CPU, "/usr/bin/nvidia-smi", None)
    assert message.startswith("WARNING: this machine has an NVIDIA GPU") and pytomography._GPU_CHECK_URL in message


def test_the_warning_names_the_likely_cause(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    assert "CPU-only build" in _device_message(CPU, "nvidia-smi", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert "driver may be too old" in _device_message(CPU, "nvidia-smi", None)
    for hidden in ("", "-1"):
        assert "CUDA_VISIBLE_DEVICES" in _device_message(CPU, "nvidia-smi", hidden)


def test_no_gpu_at_all_is_a_note_not_a_warning():
    message = _device_message(CPU, None, None)
    assert "did not find a GPU" in message and not message.startswith("WARNING")


def test_parallelproj_cpu_build_on_a_gpu_is_a_warning():
    assert _parallelproj_message("cuda", False).startswith("WARNING")
    assert _parallelproj_message("cuda", True) is None
    assert _parallelproj_message(CPU, False) is None


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="PyTomography uses the Apple Silicon GPU here")
def test_import_warns_when_the_gpu_goes_unused(tmp_path):
    """The check v3 had never printed (it compared a torch.device with a string): import PyTomography with an NVIDIA
    GPU present (a stand-in nvidia-smi on the PATH) that PyTorch cannot use (hidden by CUDA_VISIBLE_DEVICES)."""
    fake = tmp_path / ("nvidia-smi.bat" if os.name == "nt" else "nvidia-smi")
    fake.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n")
    fake.chmod(0o755)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PATH=str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    result = subprocess.run([sys.executable, "-c", "import pytomography"], env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    assert "WARNING: this machine has an NVIDIA GPU, but PyTomography is running on the CPU" in result.stderr
    assert "CUDA_VISIBLE_DEVICES='-1' hides the GPU" in result.stderr
