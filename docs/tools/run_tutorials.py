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

The kernel runs with:
  --python       the interpreter that has PyTomography and its dependencies
                 (default: $PYTOMOGRAPHY_TUTORIAL_PYTHON, else this interpreter)
  --path-prefix  extra folders put first on PATH for the kernel, e.g. a conda env's Library/bin
                 so that libparallelproj is found (default: $PYTOMOGRAPHY_TUTORIAL_PATH_PREFIX)
  --out          run folder (default: $PYTOMOGRAPHY_TUTORIAL_RUNS/<timestamp>, else docs/build/tutorial_runs/);
                 keep it short on Windows, where paths are limited to 260 characters
and PYTOMOGRAPHY_DATA from the environment (required). Outputs go to PYTOMOGRAPHY_OUTPUT if it is set
(so cached steps are reused across runs), otherwise to the run folder.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

SRCDIR = Path(__file__).resolve().parents[1] / "source"
MARKER = "__PYTOMOGRAPHY_RUN__"
RUNINFO = f"""
import json, platform, torch, pytomography
_info = dict(pytomography=getattr(pytomography, "__version__", "?"), source=pytomography.__file__,
             torch=torch.__version__, python=platform.python_version(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0)
print("{MARKER}" + json.dumps(_info))
"""


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


def run_one(name: str, run_dir: Path, timeout: int) -> dict:
    nb = nbformat.read(SRCDIR / "notebooks" / f"{name}.ipynb", as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(RUNINFO, metadata={"tags": ["run-info"]}))
    cwd = run_dir / "cwd" / name
    cwd.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="pytomography-run", resources={"metadata": {"path": str(cwd)}})
    result = {"notebook": name, "status": "passed", "cell": None, "error": None, "date": dt.date.today().isoformat()}
    t0 = time.time()
    try:
        client.execute()
    except CellTimeoutError as e:
        result.update(status="timeout", error=str(e).splitlines()[0][:300])
    except DeadKernelError as e:
        result.update(status="kernel died", error=str(e)[:300])
    except CellExecutionError as e:
        result.update(status="failed", error=f"{e.ename}: {e.evalue}"[:400])
    result["wall_time_s"] = round(time.time() - t0, 1)
    # Which cell failed, and the run information printed by the final cell
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error" and result["cell"] is None:
                result["cell"] = i
            text = out.get("text", "") if out.get("output_type") == "stream" else ""
            if MARKER in text:
                result["info"] = json.loads(text.split(MARKER, 1)[1].strip().splitlines()[0])
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
        "gpu": info.get("gpu"), "wall_time_s": result["wall_time_s"], "peak_gpu_gb": info.get("peak_gpu_gb")}
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

    results = []
    for name in names:
        print(f"running {name} ...", flush=True)
        r = run_one(name, run_dir, args.timeout)
        results.append(r)
        print(f"  {r['status']} in {r['wall_time_s']} s" + (f" (cell {r['cell']}: {r['error']})" if r["error"] else ""), flush=True)
        if args.write_back and r["status"] == "passed":
            write_back(name, run_dir, r)

    (run_dir / "report.json").write_text(json.dumps(results, indent=1), encoding="utf8")
    lines = ["| Tutorial | Status | Time (s) | Peak GPU (GB) | Error |", "|---|---|---|---|---|"]
    for r in results:
        err = (f"cell {r['cell']}: " if r["cell"] is not None else "") + (r["error"] or "")
        lines.append(f"| {r['notebook']} | {r['status']} | {r['wall_time_s']} | {r.get('info', {}).get('peak_gpu_gb', '')} | {err.replace('|', '/')} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))
    print(f"\nExecuted notebooks and report: {run_dir}")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
