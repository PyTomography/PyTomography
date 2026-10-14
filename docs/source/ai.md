# Using these docs with AI assistants

Coding assistants such as Claude often answer from what they saw during training, which may be an older version of PyTomography. These docs publish machine-readable versions of every page so an assistant can work from the current release instead.

## What is available

| File | What it contains |
|---|---|
| <a href="llms.txt"><code>llms.txt</code></a> | A short index of the documentation with a link to every page. |
| <a href="llms-full.txt"><code>llms-full.txt</code></a> | The full text of every guide, tutorial and top-level API page in one file. |
| `_md/<page>.md` | A Markdown version of a single page. Tutorials are converted from notebooks, with figures left out. Every page has a **Copy page as Markdown** button that copies it for you. |

## Giving the docs to an assistant

- **In a chat:** press **Copy page as Markdown** on the page you are working from and paste it in with your question.
- **In an agent such as Claude Code:** point it at `llms.txt` and ask it to read the pages it needs, for example: *"Read https://pytomography.readthedocs.io/en/latest/llms.txt, then help me reconstruct my Lu-177 DICOM data."*
- **For a whole project:** save `llms-full.txt` next to your code so the agent can search it.

(gpu-check)=
## Check that PyTomography uses the GPU

Do this first on any machine, before the first reconstruction. If the machine has an NVIDIA GPU, PyTomography must run on it. A broken install falls back to the CPU without failing, and is then many times slower, so a user can mistake that for PyTomography's normal speed.

```python
import shutil, subprocess, torch, pytomography

smi = shutil.which("nvidia-smi")
print("NVIDIA GPU:", subprocess.run([smi, "-L"], capture_output=True, text=True).stdout.strip() if smi else "none found")
print("PyTorch", torch.__version__, "built for CUDA", torch.version.cuda)   # None: a CPU-only PyTorch
print("PyTorch sees the GPU:", torch.cuda.is_available())
print("PyTomography's device:", pytomography.device)                       # should be cuda
try:
    import parallelproj_core                                                # the PET and CT projector
    print("parallelproj", parallelproj_core.__version__, "CUDA build:", bool(parallelproj_core.cuda_enabled))
except Exception:
    print("parallelproj 2 not installed (only PET and CT need it)")
```

If `nvidia-smi` lists a GPU but `pytomography.device` is not `cuda`, or (for PET and CT) parallelproj is not its CUDA build, the install is broken. Stop, show the user this output, and fix the install before running anything heavy. Don't carry on on the CPU unless the user chooses to. The usual causes:

| What the check shows | Cause | Fix |
|---|---|---|
| `torch.version.cuda` is `None` | A CPU-only PyTorch. On Windows, `pip install torch` installs one. | Reinstall PyTorch with the command for the machine's CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). |
| `torch.version.cuda` is set, but `torch.cuda.is_available()` is `False` | The NVIDIA driver is older than PyTorch's CUDA version, or `CUDA_VISIBLE_DEVICES` is empty or `-1`. | `nvidia-smi` shows the newest CUDA version the driver supports: update the driver or install the matching PyTorch build. Unset the variable. |
| PyTorch sees the GPU but `pytomography.device` is `cpu` | The code calls `pytomography.set_device("cpu")`. | Remove the call, unless the user chose the CPU. |
| `CUDA build: False` | The CPU build of libparallelproj, so PET and CT project on the CPU. | Install the build matching the CUDA version, for example `conda install -c conda-forge "libparallelproj=*=cuda130*" parallelproj` (see [Installation](install.md)). |

On a machine with no NVIDIA GPU, running on the CPU is expected. Apple Silicon GPUs (`mps`) are experimental.

## Loading data

Every scan reader returns the same three things in one standard form, and images go onto its grid with `.on`:

```python
from pytomography import io

projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)   # every read_<format>
ct = io.load_image(files_CT).on(object_meta)                           # any image, on the grid
photon_keV = proj_meta.energy_windows[0].photon_keV
mu = io.attenuation_map(files_CT, object_meta, energy_keV=photon_keV)  # 1/mm
io.save_dicom(recon, folder, object_meta)                              # placed in the patient
```

Lengths are mm, grids are LPS with k running foot to head, and `object_meta.affine` places every voxel. Before writing a
reader call, look the source up on the Architecture page's <a href="architecture.html#io">Data and conventions</a>
tab: its conventions, what its reader does, the vendor rules it relies on and the tutorials that use it. The
<a href="architecture.html#io-standard">standard form</a> is one table there, and the same data are JSON at
<a href="_static/arch/io-conventions.json"><code>_static/arch/io-conventions.json</code></a>.

## Planning memory before a reconstruction

Before running a large reconstruction, have the agent read [Memory and speed](memory.md). It shows how to call `estimate_memory()` and `plan_memory()`, which read the machine's free memory and give the fastest settings that fit. It also links the memory model behind them as data (`_static/memory/model.json`, with a JavaScript evaluator and tables of PET scanners and GPUs), for planning for a machine the agent cannot run Python on.

## Tips that avoid common mistakes

- Tell the assistant which PyTomography version you have installed (`pytomography.__version__`). v4 changed every reader (lengths in mm), the PET requirements and the PSF interface.
- Ask it to check tensor shapes and devices against the docstrings. Projections, objects and attenuation maps each have a fixed axis order, and mixing CPU and GPU tensors is the most common error.
- Ask for code that runs on a small subset of angles first, then scale up.
- Ask it to print `estimate_memory()` before the heavy step and to call `pytomography.set_memory_budget(...)` for the machine, so that a reconstruction does not run out of memory halfway.
