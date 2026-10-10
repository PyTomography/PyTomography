"""Export a tutorial's images for the docs' 3D viewer, from inside the finished tutorial's own kernel.

docs/tools/run_tutorials.py runs one extra cell after every tutorial listed in docs/source/tutorials/viewer.yaml.
That cell calls `run(name, out_dir, globals(), spec)`, which evaluates the yaml's expressions in the notebook's
namespace and writes, into out_dir/<name>/:

  <layer>.nii.gz  one per layer: NIfTI-1, 16-bit integers with a scale factor, reoriented so that the array axes are
                  the scanner's (RAS) axes with an axis-aligned sform. The viewer draws each on its own grid.
  manifest.json   what the viewer reads: title, credit, and per layer its kind, units, default colormap and range,
                  voxel size, shape, size in bytes and sha256.
  thumb.png       an anterior maximum-intensity projection of the SPECT or PET over a line integral of the anatomy.

The cell never raises, so a failed export never fails the tutorial; it prints one line starting with MARKER that the
runner reads. The expressions can use the helpers below as `vx.<name>`, e.g. `vx.ct_dicom(files_CT)`.

The cache: each image is also kept as the tutorial computed it (full precision, with its affine) in
<run>/viewer_cache/<name>/, keyed by its expressions in viewer.yaml. Editing anything else in viewer.yaml (ranges,
colormaps, crop, voxel_mm, slices, labels, scale groups, the size budget...) then needs no new run of the tutorial:

    python docs/tools/viewer_export.py rebuild RUN_DIR                 # every tutorial cached in RUN_DIR
    python docs/tools/viewer_export.py rebuild RUN_DIR t_dicomdata     # some of them

rewrites RUN_DIR/viewer/<name>/ from the cache with the current viewer.yaml, and names the tutorials that need a new
run: an image's expression is new or has changed, or the notebook's code has changed since the run (each cache keeps a
fingerprint of the code cells it ran; edits to text cells don't count). `rebuild --force` uses the cache anyway.

Run this file with a folder to check an export:  python docs/tools/viewer_export.py OUT_DIR/t_dicomdata
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import sys
import time
import traceback
from pathlib import Path

try:  # the runner only needs cell_source() and load_specs(); the kernel has numpy
    import numpy as np
except ImportError:
    np = None

MARKER = "__PYTOMOGRAPHY_VIEWER__"
FORMAT = "pytomography-viewer/1"
ROLE = {"spect": "overlay", "pet": "overlay", "ct": "base", "mr": "base", "mu": "base", "image": "base"}
DEFAULT_MAX_MB = 15.0


# ---------- helpers for the expressions in viewer.yaml ----------

def centred(dr_mm, shape) -> np.ndarray:
    """PyTomography's object grid with no scanner frame: voxel i's centre at (i - (N-1)/2) * dr, in mm (LPS)."""
    dr = np.broadcast_to(np.asarray(dr_mm, float), (3,))
    A = np.diag([*dr, 1.0])
    A[:3, 3] = [-(n - 1) / 2 * d for n, d in zip(shape, dr)]
    return A


def flip_axis(A, n: int, axis: int) -> np.ndarray:
    """The affine of an image flipped along one array axis of length n (index k becomes n - 1 - k)."""
    F = np.eye(4)
    F[axis, axis], F[axis, 3] = -1.0, n - 1
    return np.asarray(A, float) @ F


def object_affine(object_meta) -> np.ndarray:
    """PyTomography's object grid for PET and CT, whose object_meta.dr is in mm."""
    return centred(object_meta.dr, object_meta.shape)


def simind_affine(object_meta) -> np.ndarray:
    """A SIMIND reconstruction's grid; SIMIND's object_meta.dr is in cm."""
    return centred([10 * d for d in object_meta.dr], object_meta.shape)


def spect_affine(file_NM: str) -> np.ndarray:
    """The SPECT reconstruction grid of a DICOM projection file (LPS mm)."""
    from pytomography.io.SPECT import dicom
    return dicom._get_affine_spect_projections(file_NM)


def ct_dicom(files) -> tuple:
    """A CT series in HU and its affine (LPS mm), as PyTomography opens it, on the CPU."""
    from pytomography.io.shared.dicom import _get_affine_multifile, open_multifile
    files = [str(f) for f in files]
    return _to_numpy(open_multifile(files)), _get_affine_multifile(files)


def nifti_centred(path, step_mm: float = 2.0) -> tuple:
    """A NIfTI image placed as PyTomography's GATE helpers place it (centred, LPS; gate.get_attenuation_map_nifti),
    averaged down to about step_mm, a few slices at a time so a large MR never sits in memory whole."""
    import nibabel as nib
    img = nib.load(str(path))
    d = np.asarray(img.header["pixdim"][1:4], float)
    n = img.shape[:3]
    f = [max(1, int(round(step_mm / x))) for x in d]
    m = [k // fk for k, fk in zip(n, f)]
    out = np.empty(m, np.float32)
    for k in range(m[2]):
        block = np.asarray(img.dataobj[: m[0] * f[0], : m[1] * f[1], k * f[2]:(k + 1) * f[2]], np.float32)
        out[:, :, k] = block.reshape(m[0], f[0], m[1], f[1], f[2]).mean(axis=(1, 3, 4))
    A = np.diag([-d[0], -d[1], d[2], 1.0])
    A[:3, 3] = [(n[0] - 1) / 2 * d[0], (n[1] - 1) / 2 * d[1], -(n[2] - 1) / 2 * d[2]]
    return out, A @ _block_matrix(f)


# ---------- the export ----------

def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _block_matrix(f) -> np.ndarray:
    """Voxel i of an image averaged in blocks of f covers voxels i*f ... i*f+f-1; its centre is i*f + (f-1)/2."""
    B = np.diag([*map(float, f), 1.0])
    B[:3, 3] = [(fk - 1) / 2 for fk in f]
    return B


def _canonical(arr: np.ndarray, A: np.ndarray):
    """Permute and flip the array so its axes run along +x, +y, +z (RAS) and the affine is diagonal and positive."""
    M = A[:3, :3]
    perm = [int(np.argmax(np.abs(M[:, i]))) for i in range(3)]
    if sorted(perm) != [0, 1, 2]:
        raise ValueError("the image's axes don't map one-to-one onto the scanner's axes")
    for i, j in enumerate(perm):
        off = np.abs(np.delete(M[:, i], j))
        if off.max() > 1e-3 * abs(M[j, i]):
            raise ValueError("the image is rotated against the scanner axes (oblique); the viewer needs axis-aligned images")
    order = [perm.index(j) for j in range(3)]           # array axis that runs along scanner axis j
    arr = np.transpose(arr, order)
    A = A[:, order + [3]]
    for j in range(3):
        if A[j, j] < 0:
            n = arr.shape[j]
            arr = np.flip(arr, axis=j)
            A[:, 3] = A[:, 3] + (n - 1) * A[:, j]
            A[:, j] = -A[:, j]
    A[:3, :3] = np.diag(np.diag(A[:3, :3]))
    return arr, A


def _crop(arr, A, lo, hi):
    lo = np.clip(np.asarray(lo, int), 0, arr.shape)
    hi = np.clip(np.asarray(hi, int), lo + 1, arr.shape)
    T = np.eye(4)
    T[:3, 3] = lo
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], A @ T


def _downsample(arr, A, f):
    f = [max(1, int(x)) for x in f]
    if f == [1, 1, 1]:
        return arr, A
    m = [n // k for n, k in zip(arr.shape, f)]
    a = np.ascontiguousarray(arr[: m[0] * f[0], : m[1] * f[1], : m[2] * f[2]], dtype=np.float32)
    a = a.reshape(m[0], f[0], m[1], f[1], m[2], f[2]).mean(axis=(1, 3, 5), dtype=np.float32)
    return a, A @ _block_matrix(f)


def _parse_slices(spec):
    out = []
    for s in spec:
        parts = [int(p) if p.strip() else None for p in str(s).split(":")] if ":" in str(s) else [int(s), int(s) + 1]
        out.append(slice(*parts))
    return tuple(out)


def _quantize(arr: np.ndarray):
    """16-bit integers with a scale factor: exact for integer data such as HU, else to 1/65535 of the maximum."""
    lo, hi = float(arr.min()), float(arr.max())
    integral = np.issubdtype(arr.dtype, np.integer) or bool(np.all(np.mod(arr, 1) == 0))
    if lo >= 0:
        if integral and hi <= 65535:
            return arr.astype(np.uint16), 1.0, 0.0
        slope = hi / 65535 if hi > 0 else 1.0
        return np.round(arr / slope).astype(np.uint16), slope, 0.0
    if integral and lo >= -32768 and hi <= 32767:
        return arr.astype(np.int16), 1.0, 0.0
    slope = max(abs(lo), abs(hi)) / 32767
    return np.round(arr / slope).astype(np.int16), slope, 0.0


def _nifti_bytes(q: np.ndarray, A: np.ndarray, slope: float, inter: float, descrip: str) -> bytes:
    import nibabel as nib
    h = nib.Nifti1Header()
    h.set_data_shape(q.shape)
    h.set_data_dtype(q.dtype)
    h.set_qform(A, code=1)
    h.set_sform(A, code=1)
    h.set_xyzt_units("mm")
    h["scl_slope"], h["scl_inter"] = slope, inter
    h["vox_offset"] = 352
    h["descrip"] = descrip[:79].encode("ascii", "replace")
    raw = h.binaryblock + b"\0\0\0\0" + np.asarray(q, q.dtype.newbyteorder("<")).tobytes(order="F")
    return gzip.compress(raw, compresslevel=9, mtime=0)


def _inside(over: dict, base: dict | None):
    """Which of an overlay's voxels lie inside the anatomy: CT above -500 HU, other images above 2% of their maximum.
    Reconstructions can have hot voxels at the edge of their field of view, outside the patient or phantom; they
    shouldn't set the colour scale or the starting point. Both images are axis-aligned, so the lookup is per axis."""
    if base is None:
        return None
    thr = -500 if base["kind"] == "ct" else 0.02 * float(base["arr"].max())
    body = base["arr"] > thr
    idx, ok = [], []
    for j in range(3):
        mm = over["A"][j, 3] + over["A"][j, j] * np.arange(over["arr"].shape[j])
        k = np.round((mm - base["A"][j, 3]) / base["A"][j, j]).astype(int)
        ok.append((k >= 0) & (k < body.shape[j]))
        idx.append(np.clip(k, 0, body.shape[j] - 1))
    mask = body[np.ix_(*idx)] & ok[0][:, None, None] & ok[1][None, :, None] & ok[2][None, None, :]
    return mask if mask.any() else None


def _density(arr, kind):
    return np.maximum(0, (arr + 1000) / 1000) if kind == "ct" else np.maximum(0, arr / max(float(arr.max()), 1e-12))


def _thumbnail(layers, path: Path, height_px: int = 320) -> None:
    """Anterior view: the overlay's maximum along y over the base's line integral along y; patient's right on the left."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    base = next((L for L in layers if L["role"] == "base"), None)
    over = next((L for L in layers if L["role"] == "overlay"), None)
    frame = base or over
    ext = lambda L: [L["A"][0, 3] - L["A"][0, 0] / 2, L["A"][0, 3] + (L["arr"].shape[0] - 0.5) * L["A"][0, 0],
                     L["A"][2, 3] - L["A"][2, 2] / 2, L["A"][2, 3] + (L["arr"].shape[2] - 0.5) * L["A"][2, 2]]
    e = ext(frame)
    w_mm, h_mm = e[1] - e[0], e[3] - e[2]
    width_px = int(min(480, max(120, height_px * w_mm / h_mm)))
    fig = Figure(figsize=(width_px / 100, height_px / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_axis_off()
    if base is not None:
        drr = _density(base["arr"], base["kind"]).sum(axis=1)
        drr = (drr / max(float(drr.max()), 1e-12)) ** 0.7
        ax.imshow(drr.T * (0.55 if over is not None else 1.0), origin="lower", extent=ext(base), cmap="gray", vmin=0, vmax=1,
                  interpolation="bilinear")
    if over is not None:
        # as the viewer's 3D view draws it: colour opacity rises with intensity, so cold background stays clear
        mip = over["arr"].max(axis=1).astype(np.float32).T
        lo, hi = over["range"]
        t = np.clip((mip - lo) / max(hi - lo, 1e-12), 0, 1)
        rgba = colormaps[over["colormap"]](t)
        rgba[..., 3] = np.where(mip > lo, (0.9 if base is not None else 1.0) * np.sqrt(t), 0)
        ax.imshow(rgba, origin="lower", extent=ext(over), interpolation="bilinear")
    ax.set_xlim(e[0], e[1])
    ax.set_ylim(e[2], e[3])
    ax.set_aspect("auto")
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", pil_kwargs={"optimize": True})
    path.write_bytes(buf.getvalue())


def _layer_key(ls: dict, spec: dict) -> str:
    """What a layer's image depends on: its expressions and the spec's setup code. Display settings are left out, so
    changing them re-exports from the cache without a new run."""
    what = {"array": ls["array"], "affine": ls.get("affine"), "setup": spec.get("setup", "")}
    return hashlib.sha1(json.dumps(what, sort_keys=True).encode("utf8")).hexdigest()[:16]


def cache_dir(out_dir, name: str) -> Path:
    """<run>/viewer_cache/<name>, next to the export folder <run>/viewer."""
    return Path(out_dir).parent / "viewer_cache" / name


def code_sha(notebook) -> str:
    """A fingerprint of a notebook's code: its code cells' sources, in order. Text cells don't change the images."""
    nb = json.loads(Path(notebook).read_text(encoding="utf8"))
    code = [("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in nb["cells"]
            if c.get("cell_type") == "code" and "run-info" not in c.get("metadata", {}).get("tags", [])]
    return hashlib.sha1(json.dumps(code).encode("utf8")).hexdigest()[:16]


def _cache_write(folder: Path, name: str, spec: dict, raw: list, notebook_sha: str | None = None) -> None:
    """Keep each evaluated image (float64 as float32, other types as they are) and its affine, and an index of them."""
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for ls, arr, A, space in raw:
        key = _layer_key(ls, spec)
        keep = arr.astype(np.float32) if arr.dtype == np.float64 else arr
        np.savez_compressed(folder / f"{key}.npz", arr=keep, A=A, space=np.array(space or ""))
        entries.append({"key": key, "name": ls["name"], "array": ls["array"], "affine": ls.get("affine"),
                        "shape": list(arr.shape), "dtype": str(keep.dtype)})
    for old in folder.glob("*.npz"):                    # images of expressions that are gone
        if old.stem not in {e["key"] for e in entries}:
            old.unlink()
    import pytomography
    index = {"tutorial": name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "pytomography": getattr(pytomography, "__version__", "?"), "notebook_code_sha": notebook_sha, "layers": entries}
    (folder / "index.json").write_text(json.dumps(index, indent=1), encoding="utf8")


def _cache_read(folder: Path, spec: dict) -> list:
    """The cached images for spec's layers. A layer with `when` that isn't cached didn't apply to the run; any other
    missing layer means its expression is new or changed, and the tutorial must run again."""
    index = json.loads((folder / "index.json").read_text(encoding="utf8"))
    have = {e["key"] for e in index["layers"]}
    raw, missing = [], []
    for ls in spec["layers"]:
        key = _layer_key(ls, spec)
        if key in have:
            with np.load(folder / f"{key}.npz") as z:
                raw.append((ls, z["arr"], z["A"], str(z["space"]) or None))
        elif not ls.get("when"):
            missing.append(ls["name"])
    if missing:
        raise LookupError(f"no cached image for {', '.join(missing)}: the expression is new or changed, so run the "
                          "tutorial again")
    return raw


def export(name: str, out_dir, namespace: dict, spec: dict, notebook_sha: str | None = None) -> dict:
    """Evaluate spec's layers in namespace, keep them in the cache, and write the viewer files. Returns a summary;
    raises on failure."""
    t0 = time.time()
    env = dict(namespace)
    env["vx"] = sys.modules[__name__]
    env["np"] = np
    if spec.get("setup"):
        exec(spec["setup"], env)
    raw = []
    for ls in spec["layers"]:
        if ls.get("when") and not eval(ls["when"], env):   # a layer for one version of the tutorial only
            continue
        got = eval(ls["array"], env)
        space = None
        if isinstance(got, tuple):
            arr, A = got[0], got[1]
            if len(got) > 2:
                space = got[2]
        else:
            arr, A = got, eval(ls["affine"], env)
        raw.append((ls, _to_numpy(arr), np.asarray(_to_numpy(A), float).reshape(4, 4).copy(), space))
    try:
        _cache_write(cache_dir(out_dir, name), name, spec, raw, notebook_sha)
        cached = True
    except Exception:                                   # a full disk shouldn't cost the export
        cached = False
    result = _write(name, out_dir, spec, raw, t0)
    result["cached"] = cached
    return result


def rebuild(name: str, out_dir, spec: dict) -> dict:
    """Rewrite out_dir/<name>/ from the cache, with the current spec: no tutorial run. Raises LookupError when an
    image has to be computed again."""
    return _write(name, out_dir, spec, _cache_read(cache_dir(out_dir, name), spec), time.time())


def _write(name: str, out_dir, spec: dict, raw: list, t0: float) -> dict:
    """The viewer files from evaluated images: [(layer spec, array, affine, space from the expression or None)]."""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.nii.gz")) + [out / "manifest.json", out / "thumb.png"]:
        old.unlink(missing_ok=True)
    layers = []
    for ls, arr, A, space in raw:
        kind = ls["kind"]
        A = np.asarray(A, float).copy()
        space = space or ls.get("space", "lps")
        if ls.get("affine_scale"):                      # e.g. 10 for an affine in cm
            A[:3, :] *= float(ls["affine_scale"])
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"layer {ls['name']}: expected a 3D image, got shape {arr.shape}")
        if ls.get("slices"):
            sl = _parse_slices(ls["slices"])
            first = [s.indices(n)[0] for s, n in zip(sl, arr.shape)]
            arr = arr[sl]
            T = np.eye(4)
            T[:3, 3] = first
            A = A @ T
        if space == "lps":                              # DICOM LPS to NIfTI RAS
            A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ A
        A_in = A.copy()                                 # the tutorial's own voxel indices, for start_voxel
        arr, A = _canonical(arr, A)
        layers.append({"spec": ls, "kind": kind, "role": ls.get("role", ROLE.get(kind, "base")), "arr": arr, "A": A,
                       "A_in": A_in})

    base = next((L for L in layers if L["role"] == "base"), None)
    for L in layers:
        arr, A, ls = L["arr"], L["A"], L["spec"]
        if ls.get("crop") == "body" or (ls.get("crop") is None and L["role"] == "base" and L["kind"] == "ct"):
            thr = -500 if L["kind"] == "ct" else 0.02 * float(np.nanmax(arr))
            found = np.argwhere(np.nan_to_num(arr, nan=-np.inf) > thr)
            if len(found):
                margin = [int(math.ceil(10 / abs(A[j, j]))) for j in range(3)]
                arr, A = _crop(arr, A, found.min(0) - margin, found.max(0) + 1 + np.asarray(margin))
        elif L["role"] == "overlay" and base is not None and ls.get("crop", "base") == "base":
            # nothing outside the anatomy is shown, so don't ship it
            corners = np.array([[i, j, k, 1] for i in (0, base["arr"].shape[0] - 1) for j in (0, base["arr"].shape[1] - 1)
                                for k in (0, base["arr"].shape[2] - 1)], float).T
            bmm = base["A"] @ corners
            vox = np.linalg.inv(A) @ bmm
            arr, A = _crop(arr, A, np.floor(vox[:3].min(1)).astype(int) - 2, np.ceil(vox[:3].max(1)).astype(int) + 3)
        if ls.get("voxel_mm"):                          # one size, or one per axis
            vm = ls["voxel_mm"] if isinstance(ls["voxel_mm"], list) else [ls["voxel_mm"]] * 3
            arr, A = _downsample(arr, A, [max(1, round(float(vm[j]) / abs(A[j, j]))) for j in range(3)])
        elif ls.get("downsample"):
            arr, A = _downsample(arr, A, [int(ls["downsample"])] * 3)
        arr = np.nan_to_num(np.asarray(arr, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        L["arr"], L["A"] = (np.round(arr) if L["kind"] == "ct" else arr), A   # whole HU: exact in 16 bits

    max_bytes = float(spec.get("max_mb", DEFAULT_MAX_MB)) * 1e6
    for attempt in range(4):
        for L in layers:
            q, slope, inter = _quantize(L["arr"])
            L["file"] = L["spec"].get("file") or f"{L['spec']['name'].lower().replace(' ', '-')}.nii.gz"
            L["blob"] = _nifti_bytes(q, L["A"], slope, inter, f"PyTomography docs viewer: {name} {L['spec']['name']}")
            L["slope"], L["dtype"] = slope, str(q.dtype)
        total = sum(len(L["blob"]) for L in layers)
        if total <= max_bytes:
            break
        # over budget: average the largest anatomical image down by 2 and try again; through the slices first while they
        # are no thicker than 1.5 pixels, which keeps axial views sharp, then in plane
        shrinkable = [L for L in layers if L["role"] == "base" and min(np.abs(np.diag(L["A"])[:3])) < 3.0] or layers
        big = max(shrinkable, key=lambda L: len(L["blob"]))
        if attempt == 3:
            raise ValueError(f"the images take {total / 1e6:.1f} MB, more than the {max_bytes / 1e6:.0f} MB budget")
        d = np.abs(np.diag(big["A"])[:3])
        big["arr"], big["A"] = _downsample(big["arr"], big["A"], [1, 1, 2] if d[2] <= 1.5 * min(d[0], d[1]) else [2, 2, 1])
        if big["kind"] == "ct":
            big["arr"] = np.round(big["arr"])

    base = next((L for L in layers if L["role"] == "base"), None)
    for L in layers:
        if L["role"] == "overlay":
            L["inside"] = _inside(L, base)
            # the hottest voxel inside the anatomy
            L["hot"] = float(L["arr"][L["inside"]].max()) if L["inside"] is not None else float(L["arr"].max())
        else:
            L["hot"] = float(L["arr"].max())
        # images in the same units share one colour scale (scale_group in viewer.yaml overrides; "none" opts out),
        # so the viewer's Image switch compares like with like
        g = L["spec"].get("scale_group", f"{L['role']}:{L['spec'].get('units', '')}")
        L["group"] = None if g == "none" else g
    group_hot = {}
    for L in layers:
        if L["group"]:
            group_hot[L["group"]] = max(group_hot.get(L["group"], -np.inf), L["hot"])
    man_layers = []
    for L in layers:
        ls, arr = L["spec"], L["arr"]
        (out / L["file"]).write_bytes(L["blob"])
        mn, mx = float(arr.min()), float(arr.max())
        hot = group_hot[L["group"]] if L["group"] else L["hot"]
        if ls.get("range"):
            rng = [float(v) for v in ls["range"]]
        elif L["role"] == "overlay" or L["kind"] == "mu":  # absolute: 0 to the hottest voxel (inside the anatomy)
            rng = [0.0, hot]
        elif L["kind"] == "mr":
            rng = [float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))]
        else:
            rng = [mn, mx]
        L["range"] = rng
        L["colormap"] = ls.get("colormap", "inferno" if L["role"] == "overlay" else "gray")
        entry = {"name": ls["name"], "label": ls.get("label", ls["name"]), "kind": L["kind"], "role": L["role"],
                 "file": L["file"], "units": ls.get("units", ""), "colormap": L["colormap"], "range": [round(v, 6) for v in rng],
                 "shape": list(arr.shape), "voxel_mm": [round(float(L["A"][j, j]), 4) for j in range(3)],
                 "min": round(mn, 6), "max": round(mx, 6), "dtype": L["dtype"], "scale": L["slope"],
                 "bytes": len(L["blob"]), "sha256": hashlib.sha256(L["blob"]).hexdigest()}
        if L["kind"] == "ct":
            entry["window"] = ls.get("window", "soft")
        if L["group"] and sum(1 for M in layers if M["group"] == L["group"]) > 1:
            entry["group"] = L["group"]
        if ls.get("opacity") is not None:
            entry["opacity"] = float(ls["opacity"])
        man_layers.append(entry)

    start = None
    over = next((L for L in layers if L["role"] == "overlay"), None)
    if spec.get("start_voxel") is not None:             # a voxel of the first layer that has one, as the tutorial indexes it
        L = over or layers[0]
        start = [round(float(v), 1) for v in (L["A_in"] @ np.array([*map(float, spec["start_voxel"]), 1.0]))[:3]]
    elif over is not None:                              # the hottest spot inside the anatomy
        from scipy.ndimage import uniform_filter
        hot = uniform_filter(over["arr"], 3)
        if over.get("inside") is not None:
            hot = np.where(over["inside"], hot, -np.inf)
        ijk = np.unravel_index(int(np.argmax(hot)), over["arr"].shape)
        start = [round(float(v), 1) for v in (over["A"] @ np.array([*ijk, 1.0]))[:3]]
    _thumbnail(layers, out / "thumb.png")
    import pytomography
    manifest = {"format": FORMAT, "tutorial": name, "title": spec.get("title", name), "description": spec.get("description", ""),
                "credit": spec.get("credit", ""), "credit_url": spec.get("credit_url", ""),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pytomography": getattr(pytomography, "__version__", "?"), "start_mm": start, "layers": man_layers,
                "thumb": "thumb.png", "bytes": sum(e["bytes"] for e in man_layers)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return {"status": "exported", "folder": str(out), "layers": [e["name"] for e in man_layers],
            "bytes": manifest["bytes"], "seconds": round(time.time() - t0, 1)}


def run(name: str, out_dir, namespace: dict, spec: dict, notebook_sha: str | None = None) -> None:
    """The runner's cell: export, and print one MARKER line with the result. Never raises."""
    try:
        result = export(name, out_dir, namespace, spec, notebook_sha)
    except Exception as e:  # the tutorial itself passed; report the export's failure without failing it
        result = {"status": "export failed", "error": f"{type(e).__name__}: {e}"[:400],
                  "trace": traceback.format_exc(limit=3)[-800:]}
    print(MARKER + json.dumps(result))


def cell_source(name: str, out_dir, spec: dict, notebook_sha: str | None = None) -> str:
    """The code of the extra cell that the runner appends to a tutorial. notebook_sha (code_sha of the notebook being
    run) goes into the cache, so a later rebuild knows whether the notebook's code has changed since."""
    tools = str(Path(__file__).resolve().parent)
    return (f"import sys as _s\n_s.path.insert(0, {tools!r})\nimport viewer_export as _vx\n"
            f"_vx.run({name!r}, {str(out_dir)!r}, globals(), __import__('json').loads({json.dumps(spec)!r}), {notebook_sha!r})\n"
            "del _s, _vx\n")


def load_specs(srcdir: Path) -> dict:
    """viewer.yaml as {notebook: spec}."""
    import yaml
    path = Path(srcdir) / "tutorials" / "viewer.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf8")).get("tutorials", {}) or {}


def check(folder) -> int:
    """Re-read an export folder: checksums, sizes, and that each file decodes to its shape."""
    import nibabel as nib
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    ok = True
    for L in man["layers"]:
        blob = (folder / L["file"]).read_bytes()
        img = nib.Nifti1Image.from_bytes(gzip.decompress(blob))
        same = hashlib.sha256(blob).hexdigest() == L["sha256"] and len(blob) == L["bytes"] and list(img.shape) == L["shape"]
        ok &= same
        print(f"{L['file']:24s} {len(blob) / 1e6:6.2f} MB  shape {img.shape}  voxel {L['voxel_mm']} mm  "
              f"{L['dtype']} x {L['scale']:.3g}  range {L['range']}  {'ok' if same else 'MISMATCH'}")
    print(f"total {man['bytes'] / 1e6:.2f} MB; start {man['start_mm']}")
    return 0 if ok else 1


def rebuild_run(run_dir, names=None, srcdir=None, force: bool = False) -> int:
    """`rebuild` for a run folder: every tutorial in RUN_DIR/viewer_cache (or the names given), with viewer.yaml."""
    run_dir = Path(run_dir)
    srcdir = Path(srcdir) if srcdir else Path(__file__).resolve().parents[1] / "source"
    specs = load_specs(srcdir)
    cached = sorted(p.parent.name for p in (run_dir / "viewer_cache").glob("*/index.json"))
    todo = names or cached
    rerun = []
    for name in todo:
        if name not in specs:
            print(f"{name}: not in viewer.yaml, skipped")
            continue
        if name not in cached:
            print(f"{name}: not cached in this run; run the tutorial")
            rerun.append(name)
            continue
        index = json.loads((run_dir / "viewer_cache" / name / "index.json").read_text(encoding="utf8"))
        made, now = index.get("notebook_code_sha"), code_sha(srcdir / "notebooks" / f"{name}.ipynb")
        if made and made != now and not force:
            print(f"{name}: the notebook's code has changed since this run ({index['created']}); run it again")
            rerun.append(name)
            continue
        if not made:
            print(f"{name}: this cache doesn't record the notebook's code, so a change since {index['created']} "
                  "can't be detected")
        try:
            r = rebuild(name, run_dir / "viewer", specs[name])
            print(f"{name}: rebuilt, {r['bytes'] / 1e6:.1f} MB, {', '.join(r['layers'])}")
        except LookupError as e:
            print(f"{name}: {e}")
            rerun.append(name)
    if rerun:
        print("run again: " + ",".join(rerun))
    return 1 if rerun else 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "rebuild":
        rest = [a for a in sys.argv[3:] if a != "--force"]
        sys.exit(rebuild_run(sys.argv[2], rest or None, force="--force" in sys.argv[3:]))
    sys.exit(check(sys.argv[1]))
