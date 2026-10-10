"""Run the tutorial notebooks and report which ones work.

    python docs/tools/run_tutorials.py                       # every tutorial in tutorials.yaml
    python docs/tools/run_tutorials.py --only t_dicomdata,t_siminddata
    python docs/tools/run_tutorials.py --section SPECT --write-back
    python docs/tools/run_tutorials.py --write-back-from docs/build/tutorial_runs/20261007-091700

Notebooks are executed into a run folder (docs/build/tutorial_runs/<timestamp>/), never in place.
Each run records pass/fail, the failing cell and error, wall time, peak GPU memory and the versions
used, in report.md and report.json. With --write-back, notebooks that pass are copied back into
docs/source/notebooks with their new outputs and a "pytomography_run" stamp in their metadata,
which the docs show on the tutorial page. --write-back-from does the same for an earlier run without
running anything, and skips any notebook whose cells have been edited since that run.

After each tutorial listed in docs/source/tutorials/viewer.yaml, one more cell in the same kernel exports its images
for the docs' 3D viewer into <run>/viewer/<tutorial>/ (docs/tools/viewer_export.py; --no-viewer skips it). A failed
export is reported, and never fails the tutorial; the cell is not written back.

The kernel runs with:
  --python       the interpreter that has PyTomography and its dependencies
                 (default: $PYTOMOGRAPHY_TUTORIAL_PYTHON, else this interpreter)
  --path-prefix  extra folders put first on PATH for the kernel, e.g. a conda env's Library/bin
                 so that libparallelproj is found (default: $PYTOMOGRAPHY_TUTORIAL_PATH_PREFIX)
  --out          run folder (default: $PYTOMOGRAPHY_TUTORIAL_RUNS/<timestamp>, else docs/build/tutorial_runs/);
                 keep it short on Windows, where paths are limited to 260 characters
and PYTOMOGRAPHY_DATA from the environment (required). Outputs go to PYTOMOGRAPHY_OUTPUT if it is set
(so cached steps are reused across runs), otherwise to the run folder.

Memory: every tutorial must run in at most 25 GB of RAM (Luke, 9 Oct 2026), and say how to use less (its RAM_GB
cell). A notebook whose kernel uses more than --ram-cap-gb (default 25; 0 turns it off, to measure a tutorial's real
peak) is stopped and reported "over the RAM cap". Apart from that, a run must never take the machine's memory.
Notebooks run one at a time, and each one
  - is not started unless --min-free-gb (default 16) is free;
  - is stopped if free memory falls below --min-free-gb while it runs, or if the kernel and the
    processes it starts use more than --max-kernel-gb (default 60% of RAM);
  - on Windows, also runs in a job object (holding this script and everything it starts) whose committed
    memory is capped at --max-kernel-gb, so an allocation past the cap fails inside the kernel instead of
    exhausting the machine, and any kernel still running is killed when this script exits.
"Free" is the smaller of free RAM and, on Windows, uncommitted memory. The report gives each
notebook's peak RAM, and --write-back stamps it on the tutorial page.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

import nbformat
import psutil
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

import viewer_export  # images for the docs' 3D viewer, exported in each tutorial's kernel (docs/tools/viewer_export.py)

SRCDIR = Path(__file__).resolve().parents[1] / "source"
MARKER = "__PYTOMOGRAPHY_RUN__"
RUNINFO = f"""
import json, platform, torch, pytomography
def _peak_ram_gb():
    try:
        import psutil
        m = psutil.Process().memory_info()
        if hasattr(m, "peak_pagefile"):  # Windows: peak committed memory
            return round(m.peak_pagefile / 1e9, 1)
    except ImportError:
        pass
    import resource  # ru_maxrss is in kB on Linux, bytes on macOS
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if platform.system() == "Darwin" else 1e6), 1)
_info = dict(pytomography=getattr(pytomography, "__version__", "?"), source=pytomography.__file__,
             torch=torch.__version__, python=platform.python_version(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0,
             peak_ram_gb=_peak_ram_gb())
print("{MARKER}" + json.dumps(_info))
"""


def free_memory_gb() -> float:
    """The smaller of free RAM and, on Windows, uncommitted memory (what the page file can still back)."""
    free = psutil.virtual_memory().available
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                  "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        status = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            free = min(free, status.ullAvailPageFile)
    return free / 1e9


def _memory_gb(proc: psutil.Process) -> float:
    m = proc.memory_info()
    return (getattr(m, "private", None) or m.rss) / 1e9  # committed memory on Windows, resident elsewhere


class WindowsMemoryCap:
    """A job object that caps the committed memory of the processes in it, and kills them when it is closed."""

    def __init__(self, limit_gb: float):
        import ctypes
        from ctypes import wintypes
        self._ctypes = ctypes
        self.k32 = k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x200 | 0x2000  # JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = int(limit_gb * 1e9)
        if not self.job or not k32.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "could not create a memory-capped job object")

    def add(self, pid: int) -> bool:
        handle = self.k32.OpenProcess(0x0100 | 0x0001, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not handle:
            return False
        try:
            return bool(self.k32.AssignProcessToJobObject(self.job, handle))
        finally:
            self.k32.CloseHandle(handle)

    def close(self) -> None:
        if self.job:
            self.k32.CloseHandle(self.job)  # kills anything still in the job
            self.job = None


class MemoryWatchdog(threading.Thread):
    """Stops the kernel, and everything it started, before the machine runs short of memory."""

    def __init__(self, min_free_gb: float, max_kernel_gb: float, interval_s: float = 1.0, ram_cap_gb: float = 0):
        super().__init__(daemon=True)
        self.min_free_gb, self.max_kernel_gb, self.interval_s = min_free_gb, max_kernel_gb, interval_s
        self.ram_cap_gb = ram_cap_gb      # the tutorials' cap (0: none); the other two limits protect the machine
        self.over_cap = False
        self.peak_kernel_gb = 0.0
        self.tripped: str | None = None
        self._done = threading.Event()

    def run(self) -> None:
        me = psutil.Process()
        while not self._done.wait(self.interval_s):
            try:
                procs = me.children(recursive=True)
                used = sum(_memory_gb(p) for p in procs)
            except psutil.Error:
                continue
            free = free_memory_gb()
            self.peak_kernel_gb = max(self.peak_kernel_gb, used)
            if self.ram_cap_gb and used > self.ram_cap_gb:
                self.over_cap = True
                self.tripped = (f"over the RAM cap: the kernel used {used:.1f} GB; tutorials must run in "
                                f"{self.ram_cap_gb:.0f} GB (lower its memory use, e.g. its RAM_GB cell)")
            elif free < self.min_free_gb or used > self.max_kernel_gb:
                self.tripped = (f"stopped to protect the machine's memory: the kernel used {used:.1f} GB "
                                f"with {free:.1f} GB free (limits {self.max_kernel_gb:.0f} GB used, {self.min_free_gb:.0f} GB free)")
            if self.tripped:
                for p in procs:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                return

    def stop(self) -> None:
        self._done.set()


def kernel_dir(run_dir: Path, python: str, path_prefix: str, env: dict) -> Path:
    """A private kernelspec so the run never depends on the user's installed kernels."""
    kdir = run_dir / "jupyter" / "kernels" / "pytomography-run"
    kdir.mkdir(parents=True, exist_ok=True)
    kernel_env = dict(env)
    kernel_env["PATH"] = os.pathsep.join(p for p in [path_prefix, os.environ.get("PATH", "")] if p)
    spec = {"argv": [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "PyTomography tutorial run", "language": "python", "env": kernel_env}
    (kdir / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf8")
    return run_dir / "jupyter"


def run_one(name: str, run_dir: Path, timeout: int, min_free_gb: float, max_kernel_gb: float,
            viewer_spec: dict | None = None, ram_cap_gb: float = 0) -> dict:
    result = {"notebook": name, "status": "passed", "cell": None, "error": None, "date": dt.date.today().isoformat(),
              "wall_time_s": 0.0}
    free = free_memory_gb()
    if free < min_free_gb:
        result.update(status="not started", error=f"only {free:.1f} GB of memory free; it needs {min_free_gb:.0f} GB")
        return result
    nb = nbformat.read(SRCDIR / "notebooks" / f"{name}.ipynb", as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(RUNINFO, metadata={"tags": ["run-info"]}))
    if viewer_spec:  # after the run information, so the stamp's peak memory is the tutorial's own; dropped on write-back
        sha = viewer_export.code_sha(SRCDIR / "notebooks" / f"{name}.ipynb")   # which code the cached images came from
        nb.cells.append(nbformat.v4.new_code_cell(viewer_export.cell_source(name, run_dir / "viewer", viewer_spec, sha),
                                                  metadata={"tags": ["run-info", "viewer-export"]}))
    cwd = run_dir / "cwd" / name
    cwd.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="pytomography-run", resources={"metadata": {"path": str(cwd)}})
    watchdog = MemoryWatchdog(min_free_gb, max_kernel_gb, ram_cap_gb=ram_cap_gb)
    watchdog.start()
    t0 = time.time()
    try:
        client.execute()
    except CellTimeoutError as e:
        result.update(status="timeout", error=str(e).splitlines()[0][:300])
    except DeadKernelError as e:
        result.update(status="kernel died", error=str(e)[:300])
    except CellExecutionError as e:
        result.update(status="failed", error=f"{e.ename}: {e.evalue}"[:400])
    finally:
        watchdog.stop()
    if watchdog.tripped:
        result.update(status="over the RAM cap" if watchdog.over_cap else "stopped (memory)", error=watchdog.tripped)
    elif ram_cap_gb and watchdog.peak_kernel_gb > ram_cap_gb:   # a peak between two of the watchdog's checks
        result.update(status="over the RAM cap",
                      error=f"over the RAM cap: the kernel peaked at {watchdog.peak_kernel_gb:.1f} GB; tutorials must run in {ram_cap_gb:.0f} GB")
    result["wall_time_s"] = round(time.time() - t0, 1)
    result["peak_kernel_gb"] = round(watchdog.peak_kernel_gb, 1)
    # Which cell failed, and the run information printed by the final cell
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error" and result["cell"] is None:
                result["cell"] = i
            text = out.get("text", "") if out.get("output_type") == "stream" else ""
            if MARKER in text:
                result["info"] = json.loads(text.split(MARKER, 1)[1].strip().splitlines()[0])
            if viewer_export.MARKER in text:
                result["viewer"] = json.loads(text.split(viewer_export.MARKER, 1)[1].strip().splitlines()[0])
                result["wall_time_s"] = round(result["wall_time_s"] - result["viewer"].get("seconds", 0), 1)
    nbformat.write(nb, run_dir / f"{name}.ipynb")
    return result


def cell_sources(nb) -> list:
    return [(c.cell_type, c.source) for c in nb.cells if "run-info" not in c.get("metadata", {}).get("tags", [])]


def write_back(name: str, run_dir: Path, result: dict) -> bool:
    executed = run_dir / f"{name}.ipynb"
    nb = nbformat.read(executed, as_version=4)
    path = SRCDIR / "notebooks" / f"{name}.ipynb"
    # Never overwrite edits made to the notebook after it was run
    if cell_sources(nb) != cell_sources(nbformat.read(path, as_version=4)):
        print(f"  not written back: {path.name} has changed since this run, so run it again")
        return False
    nb.cells = [c for c in nb.cells if "run-info" not in c.get("metadata", {}).get("tags", [])]
    for c in nb.cells:
        c.metadata.pop("execution", None)  # nbclient's per-cell timestamps would change on every run
    info = result.get("info", {})
    date = result.get("date") or dt.date.fromtimestamp(executed.stat().st_mtime).isoformat()
    nb.metadata["pytomography_run"] = {
        "date": date, "pytomography": info.get("pytomography"), "torch": info.get("torch"),
        "gpu": info.get("gpu"), "wall_time_s": result["wall_time_s"], "peak_gpu_gb": info.get("peak_gpu_gb"),
        "peak_ram_gb": info.get("peak_ram_gb") or result.get("peak_kernel_gb")}
    # nbformat.writes gives the layout Jupyter saves (sources as lists of lines), so diffs stay readable
    text = nbformat.writes(nb)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf8", newline="\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated notebook names")
    ap.add_argument("--section", help="run one section of tutorials.yaml (e.g. SPECT)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds allowed per cell")
    ap.add_argument("--python", default=os.environ.get("PYTOMOGRAPHY_TUTORIAL_PYTHON", sys.executable))
    ap.add_argument("--path-prefix", default=os.environ.get("PYTOMOGRAPHY_TUTORIAL_PATH_PREFIX", ""))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-back", action="store_true", help="copy passing notebooks back with their outputs")
    ap.add_argument("--write-back-from", type=Path, metavar="RUN_DIR",
                    help="copy the passing notebooks of an earlier run back, without running anything")
    ap.add_argument("--min-free-gb", type=float, default=16,
                    help="don't start a notebook, and stop it, when less memory than this is free (default 16)")
    ap.add_argument("--max-kernel-gb", type=float, default=round(0.6 * psutil.virtual_memory().total / 1e9),
                    help="stop a notebook whose kernel uses more memory than this (default 60%% of RAM)")
    ap.add_argument("--ram-cap-gb", type=float, default=25,
                    help="the tutorials' RAM cap: a notebook using more is stopped and reported (default 25; 0 turns it off)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="don't export the 3D viewer's images (by default, tutorials in viewer.yaml export them into RUN_DIR/viewer)")
    args = ap.parse_args()

    sections = yaml.safe_load(open(SRCDIR / "tutorials" / "tutorials.yaml", encoding="utf8"))["sections"]
    names = [t["notebook"] for s in sections if not args.section or s["title"] == args.section for t in s["tutorials"]]
    if args.only:
        wanted = args.only.split(",")
        names = [n for n in names if n in wanted] + [n for n in wanted if n not in names]

    if args.write_back_from:
        run_dir = args.write_back_from.resolve()
        report = json.loads((run_dir / "report.json").read_text(encoding="utf8"))
        for r in report:
            if r["notebook"] in names and r["status"] == "passed" and write_back(r["notebook"], run_dir, r):
                print(f"wrote back {r['notebook']} (run {r.get('date', '?')}, {r['wall_time_s']} s)")
        return 0

    if "PYTOMOGRAPHY_DATA" not in os.environ:
        sys.exit("Set PYTOMOGRAPHY_DATA to the tutorial data folder first.")

    # Absolute, and short: DICOM outputs are named by UID, and Windows limits paths to 260 characters
    default_root = Path(os.environ.get("PYTOMOGRAPHY_TUTORIAL_RUNS", SRCDIR.parent / "build" / "tutorial_runs"))
    run_dir = (args.out or default_root / dt.datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # A PYTOMOGRAPHY_OUTPUT set by the caller is kept, so slow cached steps (e.g. PET normalisation) are reused across runs
    env = {"PYTOMOGRAPHY_DATA": os.environ["PYTOMOGRAPHY_DATA"],
           "PYTOMOGRAPHY_OUTPUT": os.environ.get("PYTOMOGRAPHY_OUTPUT", str(run_dir / "outputs")),
           "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
    os.environ["JUPYTER_PATH"] = str(kernel_dir(run_dir, args.python, args.path_prefix, env))

    print(f"memory: {free_memory_gb():.0f} GB free; each notebook is stopped below {args.min_free_gb:.0f} GB free "
          f"or above {args.max_kernel_gb:.0f} GB used" + (f"; tutorials over the {args.ram_cap_gb:.0f} GB RAM cap are stopped "
          "and reported" if args.ram_cap_gb else ""), flush=True)
    if os.name == "nt":
        # Every process this one starts (kernels, the venv launcher's interpreter, SIMIND) inherits the job, so the
        # cap holds even between watchdog checks; the job is closed, killing any kernel left, when this script exits.
        try:
            cap = WindowsMemoryCap(args.max_kernel_gb + 1)  # +1 GB for this script
            if not cap.add(os.getpid()):
                print("  (could not join the memory-capped job; the watchdog still applies)", flush=True)
        except OSError as e:
            print(f"  (no hard memory cap: {e}; the watchdog still applies)", flush=True)
    viewer_specs = {} if args.no_viewer else viewer_export.load_specs(SRCDIR)
    results = []
    for name in names:
        print(f"running {name} ...", flush=True)
        r = run_one(name, run_dir, args.timeout, args.min_free_gb, args.max_kernel_gb, viewer_specs.get(name), args.ram_cap_gb)
        results.append(r)
        print(f"  {r['status']} in {r['wall_time_s']} s" + (f" (cell {r['cell']}: {r['error']})" if r["error"] else ""), flush=True)
        if "viewer" in r:
            v = r["viewer"]
            print(f"  viewer: {v['status']}" + (f", {v['bytes'] / 1e6:.1f} MB in {v['folder']}" if "bytes" in v else f": {v.get('error')}"), flush=True)
        if args.write_back and r["status"] == "passed":
            write_back(name, run_dir, r)

    (run_dir / "report.json").write_text(json.dumps(results, indent=1), encoding="utf8")
    lines = ["| Tutorial | Status | Time (s) | Peak GPU (GB) | Peak RAM (GB) | Error |", "|---|---|---|---|---|---|"]
    for r in results:
        err = (f"cell {r['cell']}: " if r["cell"] is not None else "") + (r["error"] or "")
        info = r.get("info", {})
        lines.append(f"| {r['notebook']} | {r['status']} | {r['wall_time_s']} | {info.get('peak_gpu_gb', '')} | "
                     f"{info.get('peak_ram_gb') or r.get('peak_kernel_gb', '')} | {err.replace('|', '/')} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))
    print(f"\nExecuted notebooks and report: {run_dir}")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
