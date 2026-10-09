"""Download the data the tutorials read.

.. code-block:: python

    from pytomography import datasets

    folder = datasets.fetch("SPECT/Lu177-NEMA-SymT2")  # 11 MB the first time; afterwards it returns at once
    output = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

Every dataset is a folder under the data folder: ``PYTOMOGRAPHY_DATA``, or ``pytomography_data`` in your home folder.
:func:`fetch` downloads only the dataset it is asked for, checks every part against the size and checksum pinned in
the registry, unpacks it and returns its folder. An interrupted download continues where it stopped, and data already
in the folder (from an earlier download, or copied there by hand) is checked and used instead of downloaded again.
The same works from a shell::

    python -m pytomography.datasets list
    python -m pytomography.datasets fetch SPECT/Lu177-PSMA-GEDisc

The datasets, their sources, licences and checksums are listed in ``pytomography/datasets/registry.py``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from . import registry
from ._download import ChecksumError, DownloadError, download, size_text
from ._parts import ADOPT, INSTALL, PIECES, Context, describe, disk_size, download_size, host
from ._store import DOWNLOADS, MARKER, Marker, lock, part_id, problems, user_files
from ._zip import Missing

__all__ = ["fetch", "path", "data_dir", "output_dir", "available", "info", "verify", "cache_key",
           "DatasetNotAvailable", "DownloadError", "ChecksumError"]

PathLike = Union[str, os.PathLike]


class DatasetNotAvailable(RuntimeError):
    """The dataset is in the registry but cannot be downloaded yet."""


def data_dir() -> Path:
    """The tutorial data folder: ``PYTOMOGRAPHY_DATA``, or ``pytomography_data`` in your home folder."""
    return Path(os.environ.get("PYTOMOGRAPHY_DATA") or "~/pytomography_data").expanduser()


def output_dir(name: str = "") -> Path:
    """The folder for a tutorial's results, created if needed: ``PYTOMOGRAPHY_OUTPUT`` (or ``pytomography_outputs``
    in the current folder), then ``name``. Results never go into the data folder."""
    folder = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT") or "pytomography_outputs").expanduser() / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _root(folder: Optional[PathLike]) -> Path:
    return Path(folder).expanduser() if folder else data_dir()


def _entry(name: str) -> dict:
    try:
        return registry.DATASETS[name]
    except (KeyError, TypeError):
        lower = {k.lower(): k for k in registry.DATASETS}
        close = [lower[c] for c in difflib.get_close_matches(str(name).lower(), lower, n=3, cutoff=0.5)]
        hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"No dataset is called {name!r}.{hint} datasets.available() lists them all.") from None


def _selected_parts(name: str, entry: dict, extras: Union[str, Sequence[str]]) -> List[dict]:
    known = entry.get("extras", {})
    if isinstance(extras, str):
        extras = list(known) if extras == "all" else [extras]
    for x in extras:
        if x not in known:
            raise ValueError(f"{name} has no extra called {x!r}. Its extras: {', '.join(known) or 'none'}.")
    return list(entry["parts"]) + [p for x in extras for p in known[x]["parts"]]


def _known_ids(entry: dict) -> set:
    return {part_id(p) for p in entry["parts"]} | {part_id(p) for x in entry.get("extras", {}).values()
                                                   for p in x["parts"]}


def _state(name: str, entry: dict, root: Path) -> str:
    if entry.get("status") == "pending":
        return "not published yet"
    folder = root.joinpath(*name.split("/"))
    marker = Marker(folder, name)
    if entry["parts"] and all(part_id(p) in marker.parts for p in entry["parts"]):
        return "downloaded"
    if marker.parts:
        return "partly downloaded"
    return "in the folder, not checked yet" if user_files(folder) else "not downloaded"


def _doi(entry: dict) -> str:
    url = entry.get("url") or ""
    return "doi:" + url[len("https://doi.org/"):] if url.startswith("https://doi.org/") else url


def fetch(name: str, *, extras: Union[str, Sequence[str]] = (), data_dir: Optional[PathLike] = None,
          force: bool = False, verify: str = "size", progress: bool = True, workers: int = 4) -> Path:
    """Download a tutorial dataset if it is not there yet, and return its folder.

    Args:
        name: the dataset: its folder under the data folder, such as ``"SPECT/Lu177-NEMA-SymT2"``. :func:`available`
            lists them.
        extras: optional parts to add, by name, or ``"all"``. :func:`info` lists a dataset's extras.
        data_dir: the data folder. Defaults to :func:`data_dir`: ``PYTOMOGRAPHY_DATA``, or ``~/pytomography_data``.
        force: download and unpack everything again.
        verify: how to check a dataset that is already there. ``"size"`` compares file sizes with what was written,
            which is instant; ``"hash"`` re-reads every file and compares checksums; ``"none"`` trusts the record.
            Files that fail are downloaded again. New downloads are always checked against the registry.
        progress: print what is downloaded, a progress bar, and the licence and citation of the data.
        workers: the most connections to open at once.

    Returns:
        The dataset's folder.

    Raises:
        DatasetNotAvailable: the dataset cannot be downloaded yet, and its folder is empty. (If it already holds
            files, they are used as they are, without checks.)
        DownloadError: a server could not be reached after several retries. Run fetch() again to continue.
        ChecksumError: downloaded data does not match the registry.
    """
    entry = _entry(name)
    if verify not in ("size", "hash", "none"):
        raise ValueError(f"verify must be 'size', 'hash' or 'none', not {verify!r}")
    root = _root(data_dir)
    folder = root.joinpath(*name.split("/"))
    if entry.get("status") == "pending":
        if not user_files(folder):
            raise DatasetNotAvailable(f"{name} cannot be downloaded yet. {entry.get('note', '')}".strip())
        if progress:
            print(f"{name} cannot be downloaded yet, so the files already in its folder are used as they are,"
                  " unchecked.", flush=True)
        return folder
    parts = _selected_parts(name, entry, extras)
    ctx = Context(name, folder, workers, progress)

    def broken(marker: Marker) -> Dict[str, Optional[List[str]]]:
        """Parts that are not in place: None if never fetched, else what is wrong with their files."""
        out = {}
        for part in parts:
            record = marker.parts.get(part_id(part))
            if record is None:
                out[part_id(part)] = None
            elif verify != "none":
                found = problems(folder, record["files"], verify == "hash")
                if found:
                    out[part_id(part)] = found
        return out

    # Most calls find the dataset complete, which needs no lock and nothing written
    if not force and verify != "hash" and (folder / MARKER).is_file() and not broken(Marker(folder, name)):
        ctx.say(f"{name}: already downloaded")
        return folder

    with lock(folder, name, progress):
        marker = Marker(folder, name)
        if force:
            marker.data["parts"] = {}
        todo = broken(marker)
        for found in todo.values():
            if found:
                ctx.say(f"{name}: {len(found)} {'file' if len(found) == 1 else 'files'} changed or missing, such as"
                        f" {found[0]}; fetching again")
        if todo and not force and user_files(folder):
            for part in parts:
                pid = part_id(part)
                if pid not in todo or todo[pid] is not None:
                    continue
                try:
                    files = ADOPT[part["kind"]](part, ctx)
                except (OSError, ValueError, zipfile.BadZipFile, Missing):
                    files = None  # can't tell; download it instead
                if files is not None:
                    marker.add(pid, part["kind"], "already in the folder", files)
                    del todo[pid]
            if not todo:
                ctx.say(f"{name}: the files already in its folder match the registry")
        if todo:
            _fetch_parts(name, entry, [p for p in parts if part_id(p) in todo], marker, ctx, root)
        known = _known_ids(entry)
        stale = [pid for pid in marker.parts if pid not in known]
        if stale:  # parts of an older registry entry: remove the files that the current parts did not rewrite
            keep = {rel for pid, record in marker.parts.items() if pid in known for rel in record["files"]}
            for pid in stale:
                for rel in marker.parts[pid]["files"]:
                    if rel not in keep:
                        folder.joinpath(*rel.split("/")).unlink(missing_ok=True)
                marker.drop(pid)
        try:
            (folder / DOWNLOADS).rmdir()
        except OSError:
            pass  # absent, or holds an interrupted download
    return folder


def _fetch_parts(name: str, entry: dict, parts: List[dict], marker: Marker, ctx: Context, root: Path) -> None:
    pieces = {part_id(p): PIECES[p["kind"]](p, ctx) for p in parts}
    everything = [piece for group in pieces.values() for piece in group]
    missing = sum(p.size for p in everything if not (p.dest.exists() and p.dest.stat().st_size == p.size))
    needed = missing + sum(disk_size(p) for p in parts if p["kind"] != "file")
    free = shutil.disk_usage(ctx.folder).free
    if needed > free:
        raise OSError(f"{name} needs {size_text(needed)} of free disk space in {root}, and {size_text(free)} is free."
                      " Free some space, or set PYTOMOGRAPHY_DATA to a folder on a larger disk.")
    fetched = missing + sum(p["size"] for p in parts if p["kind"] == "idc")
    sources = " and ".join(dict.fromkeys(host(p) for p in parts))
    ctx.say(f"{name}: {size_text(fetched)} from {sources} ({_doi(entry)})" if fetched else
            f"{name}: nothing left to download")
    download(everything, ctx.workers, ctx.progress)
    count = 0
    for part in parts:
        files, source = INSTALL[part["kind"]](part, ctx, pieces[part_id(part)])
        marker.add(part_id(part), part["kind"], source, files)
        count += len(files)
    ctx.say(f"{name}: {count} {'file' if count == 1 else 'files'} checked and in place, in {ctx.folder}")
    ctx.say(f"Licence: {entry['licence']}." + (f" If you use these data, please cite:\n  {entry['cite']}"
                                                if entry.get("cite") else ""))


def path(name: str, data_dir: Optional[PathLike] = None) -> Path:
    """The folder of a dataset that :func:`fetch` has downloaded, without checking or downloading anything.

    Raises FileNotFoundError if it has not been downloaded."""
    entry = _entry(name)
    root = _root(data_dir)
    if _state(name, entry, root) != "downloaded":
        raise FileNotFoundError(f"{name} has not been downloaded to {root}. Run datasets.fetch({name!r}).")
    return root.joinpath(*name.split("/"))


def info(name: str, data_dir: Optional[PathLike] = None) -> dict:
    """What a dataset is, where it comes from, its licence and citation, what fetch() downloads for it, and whether
    it is downloaded."""
    entry = _entry(name)
    root = _root(data_dir)
    out = {"name": name}
    out.update({k: entry.get(k) for k in ("title", "source", "url", "licence", "cite", "tutorials", "status", "note")})
    out["download_bytes"] = sum(download_size(p) for p in entry["parts"])
    out["disk_bytes"] = sum(disk_size(p) for p in entry["parts"])
    out["parts"] = [describe(p) for p in entry["parts"]]
    out["extras"] = {x: {"title": e.get("title", ""), "download_bytes": sum(download_size(p) for p in e["parts"])}
                     for x, e in entry.get("extras", {}).items()}
    out["folder"] = root.joinpath(*name.split("/"))
    out["state"] = _state(name, entry, root)
    return out


def available(data_dir: Optional[PathLike] = None):
    """Every dataset, with its download size, licence and state, as a pandas DataFrame."""
    import pandas as pd

    root = _root(data_dir)
    rows = [{"dataset": name, "title": e["title"], "licence": e["licence"],
             "download": "" if e.get("status") == "pending" else size_text(sum(download_size(p) for p in e["parts"])),
             "state": _state(name, e, root)} for name, e in registry.DATASETS.items()]
    return pd.DataFrame(rows, columns=["dataset", "download", "licence", "state", "title"])


def verify(*names: str, level: str = "size", data_dir: Optional[PathLike] = None) -> Dict[str, List[str]]:
    """Check downloaded datasets against what :func:`fetch` recorded when it wrote them.

    Args:
        names: the datasets to check; every downloaded dataset if none are given.
        level: ``"size"`` compares file sizes; ``"hash"`` re-reads every file and compares checksums.

    Returns:
        {dataset: problems} for every dataset checked. An empty list means the dataset is complete and unchanged."""
    if level not in ("size", "hash"):
        raise ValueError(f"level must be 'size' or 'hash', not {level!r}")
    root = _root(data_dir)
    out = {}
    for name in names or list(registry.DATASETS):
        entry = _entry(name)
        folder = root.joinpath(*name.split("/"))
        marker = Marker(folder, name)
        if not marker.parts:
            if names:
                out[name] = ["not downloaded"]
            continue
        found = []
        known = _known_ids(entry)
        for pid, record in marker.parts.items():
            if pid not in known:
                found.append("downloaded from an older version of the registry; fetch() updates it")
            found += problems(folder, record["files"], level == "hash")
        found += [f"not downloaded: {describe(p)}" for p in entry["parts"] if part_id(p) not in marker.parts]
        out[name] = found
    return out


def cache_key(*names: str) -> str:
    """A short hash of what the registry pins for these datasets. It changes only when one of them changes, so it
    can key a cache of the data, as the tests on GitHub do."""
    data = [[name, sorted(part_id(p) for p in _entry(name)["parts"])] for name in names]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16]


def _tutorial_datasets(notebook: str) -> List[str]:
    """The datasets a tutorial notebook, such as "t_dicomdata", reads."""
    found = [name for name, e in registry.DATASETS.items() if notebook in e.get("tutorials", [])]
    if not found:
        tutorials = sorted({t for e in registry.DATASETS.values() for t in e.get("tutorials", [])})
        close = difflib.get_close_matches(notebook, tutorials, n=3)
        hint = f" Did you mean {' or '.join(close)}?" if close else ""
        raise ValueError(f"No dataset lists the tutorial {notebook!r}: it reads no data, or its name is wrong.{hint}")
    return found
