"""What every benchmark case shares: the command line, stage timers, memory peaks, machine details and the result file.

A case script runs in a process of its own (benchmarks/run.py starts one per case), so whatever memory one case holds,
PyTorch's and CuPy's caches included, goes back to the system when it ends and never counts against the next case.
Inside a case, each step is a stage:

    bench = Bench('spect_lu177', recons=['osem', 'bsrem'], description=...)
    with bench.stage('load'):
        projections, object_meta, proj_meta = ...
    image = bench.repeat('osem 4x8', lambda: OSEM(likelihood)(n_iters=4, n_subsets=8))
    bench.save()

A stage synchronises the GPU before its clock starts and after it stops, frees what earlier stages left in the
caches (outside the clock), and records its peaks: the RAM this process has committed (sampled every 20 ms), the GPU
memory PyTorch allocated, and the GPU memory in use on the device, which also counts CuPy, parallelproj and spectarf
(sampled every 20 ms; on a shared GPU it also counts other processes' changes).

repeat() times the same computation more than once. The first run includes one-off work (compiling kernels, filling
the system matrix's caches, such as each subset's sensitivity image); the repeats show the steady state. Both are
reported. The repeats stop after `repeats` runs or once they have taken `max_seconds`, whichever comes first, but there
is always at least one.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch

import pytomography
from pytomography.utils.memory import _host_memory

GB = 1e9


def _process_bytes() -> float:
    """Memory this process has committed (Windows: private bytes; Linux: resident size)."""
    return _host_memory()[2]


def _cuda() -> bool:
    return torch.device(pytomography.device).type == 'cuda' and torch.cuda.is_available()


def _device_used_bytes() -> float:
    free, total = torch.cuda.mem_get_info(pytomography.device)
    return float(total - free)


def free_caches() -> None:
    """Returns what Python, PyTorch and CuPy hold for reuse, so that the next stage starts from a clean slate."""
    gc.collect()
    if _cuda():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if 'cupy' in sys.modules:
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


class _Sampler(threading.Thread):
    """Samples this process's RAM and the device's used GPU memory until stopped, keeping the peaks."""

    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.cuda = _cuda()
        self.ram_peak = _process_bytes()
        self.gpu_start = self.gpu_peak = _device_used_bytes() if self.cuda else 0.0
        self._done = threading.Event()

    def sample(self) -> None:
        self.ram_peak = max(self.ram_peak, _process_bytes())
        if self.cuda:
            self.gpu_peak = max(self.gpu_peak, _device_used_bytes())

    def run(self) -> None:
        while not self._done.wait(self.interval_s):
            self.sample()

    def stop(self) -> None:
        self._done.set()
        self.join()
        self.sample()


def machine() -> dict:
    """What the numbers depend on: the GPU, its driver, the CPU, the memory and the versions of the packages."""
    info = dict(os=f'{platform.system()} {platform.release()} ({platform.version()})', python=platform.python_version(),
                cpu=_cpu_name(), cores=os.cpu_count(), ram_gb=round(_host_memory()[0] / GB, 1),
                device=str(pytomography.device))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / GB, 1), cuda=torch.version.cuda)
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            info['driver'] = out.splitlines()[0] if out else None
        except (OSError, subprocess.SubprocessError):
            info['driver'] = None
    packages = {}
    for name in ['pytomography', 'torch', 'numpy', 'parallelproj-core', 'libparallelproj', 'parallelproj', 'spectarf',
                 'cupy-cuda13x', 'cupy-cuda12x', 'numba', 'numba-cuda']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:  # conda-forge's parallelproj-core installs the module alone, without package metadata
        import parallelproj_core
        packages['parallelproj_core'] = (f"{getattr(parallelproj_core, '__version__', '?')} "
                                         f"({'CUDA' if getattr(parallelproj_core, 'cuda_enabled', False) else 'CPU'} build)")
    except Exception:
        pass
    info['packages'] = packages
    return info


def _cpu_name() -> str:
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    if os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    if sys.platform == 'darwin':
        try:
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or 'unknown'


class Bench:
    """One benchmark case: parses the command line, times its stages and writes the result file.

    Command line (every case script takes it):
        --recon NAME   which of the case's reconstructions to run (default: the first)
        --out FILE     where to write the result (JSON); default: print it
        --quick        a smoke test: the fewest iterations and no repeats (the numbers mean nothing)
        --device D     cuda or cpu (default: PyTomography's own choice)
        --repeats N    timed runs after the first (default 3)
        --max-seconds S  stop repeating once the repeats have taken this long (default 90)
        --ram-gb G     PyTomography's memory budget, as the tutorials' RAM_GB cell (default 25)
    """

    def __init__(self, case: str, recons: list[str], description: dict[str, str]):
        ap = argparse.ArgumentParser(description=f'Benchmark case {case}: ' + '; '.join(
            f'{r}: {d}' for r, d in description.items()))
        ap.add_argument('--recon', choices=recons, default=recons[0])
        ap.add_argument('--out', type=Path)
        ap.add_argument('--quick', action='store_true')
        ap.add_argument('--device', choices=['cuda', 'cpu'])
        ap.add_argument('--repeats', type=int, default=3)
        ap.add_argument('--max-seconds', type=float, default=90.0)
        ap.add_argument('--ram-gb', type=float, default=25.0,
                        help="PyTomography's memory budget (set_memory_budget), as the tutorials' RAM_GB (default 25)")
        args = ap.parse_args()
        if args.device:
            pytomography.set_device(args.device)
        pytomography.set_memory_budget(args.ram_gb)
        self.ram_budget_gb = args.ram_gb
        self.case, self.recon, self.quick, self.out = case, args.recon, args.quick, args.out
        self.repeats, self.max_seconds = (0, 0.0) if args.quick else (max(0, args.repeats), args.max_seconds)
        self.result = dict(case=case, recon=args.recon, description=description[args.recon], quick=args.quick,
                           started=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), stages=[],
                           settings=dict(memory_budget_gb=args.ram_gb), machine=machine())
        self._t0 = time.perf_counter()

    def setting(self, **values) -> None:
        """Records settings that the numbers depend on (iterations, subsets, grid, ...)."""
        self.result['settings'].update(values)

    @contextmanager
    def _measured(self):
        """Times a block with the GPU synchronised at both ends, and samples its memory."""
        cuda = _cuda()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampler = _Sampler()
        sampler.start()
        box = {}
        t0 = time.perf_counter()
        try:
            yield box
        finally:
            if cuda:
                torch.cuda.synchronize()
            box['s'] = time.perf_counter() - t0
            sampler.stop()
            box['ram_gb'] = sampler.ram_peak / GB
            box['gpu_torch_gb'] = torch.cuda.max_memory_allocated() / GB if cuda else 0.0
            box['gpu_device_gb'] = (sampler.gpu_peak - sampler.gpu_start) / GB if cuda else 0.0

    def _record(self, entry: dict) -> None:
        for k in ('ram_gb', 'gpu_torch_gb', 'gpu_device_gb'):
            entry[k] = round(entry[k], 2)
        self.result['stages'].append(entry)
        times = f"first {entry['first_s']:.2f} s" + (f", repeats {entry['repeat_median_s']:.2f} s (median of "
                                                     f"{len(entry['repeats_s'])})" if entry.get('repeats_s') else '') \
            if 'first_s' in entry else f"{entry['s']:.2f} s"
        print(f"[{self.case}/{self.recon}] {entry['stage']}: {times}; peaks: RAM {entry['ram_gb']:.1f} GB, "
              f"GPU {entry['gpu_torch_gb']:.2f} GB (PyTorch), +{entry['gpu_device_gb']:.2f} GB on the device", flush=True)

    @contextmanager
    def stage(self, name: str):
        """Times a step that runs once (reading data, building a system matrix)."""
        free_caches()
        with self._measured() as m:
            yield
        self._record(dict(stage=name, s=round(m['s'], 3), ram_gb=m['ram_gb'], gpu_torch_gb=m['gpu_torch_gb'],
                          gpu_device_gb=m['gpu_device_gb']))

    def repeat(self, name: str, fn):
        """Runs fn() once, then repeats it (see the module's docstring); returns the last result."""
        free_caches()
        runs, peaks, result = [], dict(ram_gb=0.0, gpu_torch_gb=0.0, gpu_device_gb=0.0), None
        while True:
            result = None          # the previous run's image is not kept while the next one runs
            gc.collect()
            with self._measured() as m:
                result = fn()
            runs.append(m['s'])
            for k in peaks:
                peaks[k] = max(peaks[k], m[k])
            if len(runs) > self.repeats or (len(runs) > 1 and sum(runs[1:]) >= self.max_seconds):
                break
        entry = dict(stage=name, first_s=round(runs[0], 3), repeats_s=[round(r, 3) for r in runs[1:]], **peaks)
        if runs[1:]:
            entry['repeat_median_s'] = round(statistics.median(runs[1:]), 3)
        self._record(entry)
        return result

    def save(self) -> None:
        self.result['total_s'] = round(time.perf_counter() - self._t0, 1)
        self.result['process_ram_peak_gb'] = round(max(s['ram_gb'] for s in self.result['stages']), 2) \
            if self.result['stages'] else None
        text = json.dumps(self.result, indent=1)
        if self.out:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text(text, encoding='utf-8')
        else:
            print(text)
