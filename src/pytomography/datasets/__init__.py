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

The datasets, their sources, licences and checksums are listed in :mod:`pytomography.datasets.registry`.

* :func:`fetch`: download a dataset if needed, and return its folder;
* :func:`path`: the folder of a dataset already downloaded;
* :func:`available` and :func:`info`: every dataset, and what one dataset is;
* :func:`verify`: check downloaded datasets for missing or changed files;
* :func:`data_dir` and :func:`output_dir`: the data folder, and the folder for a tutorial's results;
* :func:`cache_key`: a key that changes when the registry's entry for a dataset changes;
* :class:`DatasetNotAvailable`, :class:`DownloadError` and :class:`ChecksumError`: what :func:`fetch` raises.
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
from ._parts import ADOPT, INSTALL, PIECES, Context, describe, disk_size, download_size, host, pending, target
from ._store import DOWNLOADS, MARKER, Marker, lock, part_id, problems, user_files
from ._zip import Missing

__all__ = ["fetch", "path", "data_dir", "output_dir", "available", "info", "verify", "cache_key",
           "DatasetNotAvailable", "DownloadError", "ChecksumError"]

PathLike = Union[str, os.PathLike]


class DatasetNotAvailable(RuntimeError):
    """Raised when a dataset is in the registry but cannot be downloaded yet, and its folder is empty."""


def data_dir() -> Path:
    """Returns the tutorial data folder: ``PYTOMOGRAPHY_DATA``, or ``pytomography_data`` in your home folder.

    Returns:
        Path: The data folder (it may not exist yet).
    """
    return Path(os.environ.get("PYTOMOGRAPHY_DATA") or "~/pytomography_data").expanduser()


def output_dir(name: str = "") -> Path:
    """Returns the folder for a tutorial's results, and creates it if needed.

    The folder is ``name`` under ``PYTOMOGRAPHY_OUTPUT``, or under ``pytomography_outputs`` in the current folder.
    Results never go into the data folder.

    Args:
        name (str, optional): The subfolder, such as a dataset's name. Defaults to '': the output folder itself.

    Returns:
        Path: The folder.
    """
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


def _all_parts(entry: dict) -> List[dict]:
    return list(entry["parts"]) + [p for x in entry.get("extras", {}).values() for p in x["parts"]]


def _known_ids(entry: dict) -> set:
    return {part_id(p) for p in entry["parts"]} | {part_id(p) for x in entry.get("extras", {}).values()
                                                   for p in x["parts"]}


def _state(name: str, entry: dict, root: Path) -> str:
    if entry.get("status") == "pending":
        return "not published yet"
    folder = root.joinpath(*name.split("/"))
    marker = Marker(folder, name)
    parts = [p for p in entry["parts"] if not pending(p)]    # a part whose record is not out cannot be downloaded
    if parts and all(part_id(p) in marker.parts for p in parts):
        return "downloaded"
    if marker.parts:
        return "partly downloaded"
    return "in the folder, not checked yet" if user_files(folder) else "not downloaded"


def _doi(entry: dict) -> str:
    url = entry.get("url") or ""
    return "doi:" + url[len("https://doi.org/"):] if url.startswith("https://doi.org/") else url


def fetch(name: str, *, extras: Union[str, Sequence[str]] = (), data_dir: Optional[PathLike] = None,
          force: bool = False, verify: str = "size", progress: bool = True, workers: int = 4) -> Path:
    """Downloads a tutorial dataset if it is not there yet, and returns its folder.

    A dataset that cannot be downloaded yet (its record is not published) is used as it is if its folder already
    holds files, without checks, and the dataset's files that come with PyTomography, such as masks, are copied in. A
    part of a dataset whose record is not published yet (a pending part, such as a tutorial's regions) is not
    downloaded either: a copy in the folder is used, and ``fetch`` says when there is none.

    Args:
        name (str): The dataset: its folder under the data folder, such as ``"SPECT/Lu177-NEMA-SymT2"``.
            :func:`available` lists them.
        extras (str | Sequence[str], optional): Optional parts to add, by name, or ``"all"``. :func:`info` lists a
            dataset's extras. Defaults to no extras.
        data_dir (str | os.PathLike | None, optional): The data folder. Defaults to None: :func:`data_dir`,
            ``PYTOMOGRAPHY_DATA`` or ``~/pytomography_data``.
        force (bool, optional): Download and unpack everything again. Defaults to False.
        verify (str, optional): How to check a dataset that is already there. ``"size"`` compares file sizes with what
            was written, which is instant; ``"hash"`` re-reads every file and compares checksums; ``"none"`` trusts
            the record. Files that fail are downloaded again. New downloads are always checked against the registry.
            Defaults to ``"size"``.
        progress (bool, optional): Print what is downloaded, a progress bar, and the licence and citation of the
            data. Defaults to True.
        workers (int, optional): The most connections to open at once. Defaults to 4.

    Returns:
        Path: The dataset's folder.

    Raises:
        ValueError: If no dataset is called ``name``, it has no extra of a name in ``extras``, or ``verify`` is not
            one of the values above.
        DatasetNotAvailable: If the dataset cannot be downloaded yet, and its folder is empty.
        DownloadError: If a download failed after every retry, or a server sent something other than the file asked
            for. Run ``fetch`` again to continue.
        ChecksumError: If downloaded data does not match the registry.
        OSError: If the data folder's disk does not have the free space the download needs.
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
        _install_package_parts(name, entry, folder, progress)
        return folder
    parts = _selected_parts(name, entry, extras)
    ctx = Context(name, folder, workers, progress)
    waiting = [p for p in parts if pending(p)]
    parts = [p for p in parts if not pending(p)]
    for part in waiting:
        if not folder.joinpath(*target(part).split("/")).is_file():
            ctx.say(f"{name}: {target(part)} cannot be downloaded yet: it will be in {part['record']}, which is not"
                    " published yet. The tutorials that read it need a copy in the dataset's folder.")

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
            keep |= {target(p) for p in _all_parts(entry) if pending(p)}   # the copies of parts not published yet
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


def _install_package_parts(name: str, entry: dict, folder: Path, progress: bool) -> None:
    """The files of a pending dataset that come with PyTomography (its package parts, such as a mask of the
    phantom's regions): copied into its folder, or replaced there if they differ, since they need no download."""
    ctx = Context(name, folder, 1, progress)
    for part in entry["parts"]:
        if part["kind"] == "package" and ADOPT["package"](part, ctx) is None:
            files, _ = INSTALL["package"](part, ctx, [])
            ctx.say(f"{name}: {', '.join(files)} copied from the PyTomography package")


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
    ctx.say(f"{name}: {count} {'file' if count == 1 else 'files'} checked and in place in the data folder")
    ctx.say(f"Licence: {entry['licence']}." + (f" If you use these data, please cite:\n  {entry['cite']}"
                                                if entry.get("cite") else ""))


def path(name: str, data_dir: Optional[PathLike] = None) -> Path:
    """Returns the folder of a dataset that :func:`fetch` has downloaded, without checking or downloading anything.

    Args:
        name (str): The dataset, such as ``"SPECT/Lu177-NEMA-SymT2"``.
        data_dir (str | os.PathLike | None, optional): The data folder. Defaults to None: :func:`data_dir`.

    Returns:
        Path: The dataset's folder.

    Raises:
        ValueError: If no dataset is called ``name``.
        FileNotFoundError: If :func:`fetch` has not downloaded every part of the dataset that can be downloaded; always
            for a dataset that cannot be downloaded yet.
    """
    entry = _entry(name)
    root = _root(data_dir)
    if _state(name, entry, root) != "downloaded":
        raise FileNotFoundError(f"{name} has not been downloaded to {root}. Run datasets.fetch({name!r}).")
    return root.joinpath(*name.split("/"))


def info(name: str, data_dir: Optional[PathLike] = None) -> dict:
    """Returns what a dataset is, where it comes from, its licence, what :func:`fetch` downloads, and its state.

    Args:
        name (str): The dataset, such as ``"SPECT/Lu177-NEMA-SymT2"``.
        data_dir (str | os.PathLike | None, optional): The data folder. Defaults to None: :func:`data_dir`.

    Returns:
        dict: ``name``; the registry's ``title``, ``source``, ``url``, ``licence``, ``cite``, ``tutorials``,
        ``status`` and ``note`` (None where it has none); ``download_bytes`` and ``disk_bytes`` of its parts;
        ``parts`` (a description of each); ``extras`` (each extra's ``title`` and ``download_bytes``); ``folder``
        (a ``Path``); and ``state``: ``'downloaded'``, ``'partly downloaded'``, ``'in the folder, not checked yet'``,
        ``'not downloaded'`` or ``'not published yet'``.

    Raises:
        ValueError: If no dataset is called ``name``.
    """
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
    """Returns every dataset of the registry, with its download size, licence and state, as a pandas DataFrame.

    Args:
        data_dir (str | os.PathLike | None, optional): The data folder. Defaults to None: :func:`data_dir`.

    Returns:
        pandas.DataFrame: One row per dataset, with the columns ``dataset``, ``download`` (its size as text; empty
        for a dataset that cannot be downloaded yet), ``licence``, ``state`` and ``title``.
    """
    import pandas as pd

    root = _root(data_dir)
    rows = [{"dataset": name, "title": e["title"], "licence": e["licence"],
             "download": "" if e.get("status") == "pending" else size_text(sum(download_size(p) for p in e["parts"])),
             "state": _state(name, e, root)} for name, e in registry.DATASETS.items()]
    return pd.DataFrame(rows, columns=["dataset", "download", "licence", "state", "title"])


def verify(*names: str, level: str = "size", data_dir: Optional[PathLike] = None) -> Dict[str, List[str]]:
    """Checks downloaded datasets against what :func:`fetch` recorded when it wrote them.

    Args:
        *names (str): The datasets to check. Defaults to every dataset that is downloaded, wholly or in part.
        level (str, optional): ``"size"`` compares file sizes; ``"hash"`` re-reads every file and compares checksums.
            Defaults to ``"size"``.
        data_dir (str | os.PathLike | None, optional): The data folder. Defaults to None: :func:`data_dir`.

    Returns:
        dict[str, list[str]]: The problems of each dataset checked: missing or changed files, parts not downloaded,
        and parts from an older version of the registry (``['not downloaded']`` for a dataset named in ``names`` that
        is not downloaded at all). An empty list means the dataset is complete and unchanged.

    Raises:
        ValueError: If ``level`` is not ``"size"`` or ``"hash"``, or no dataset is called one of ``names``.
    """
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
        found += [f"not downloaded: {describe(p)}" for p in entry["parts"]
                  if part_id(p) not in marker.parts and not pending(p)]
        out[name] = found
    return out


def cache_key(*names: str) -> str:
    """Returns a short hash of what the registry pins for these datasets, to key a cache of their data.

    The key changes only when the registry's parts of one of the datasets change, so it can key a cache of the data,
    as the tests on GitHub do.

    Args:
        *names (str): The datasets.

    Returns:
        str: 16 hexadecimal digits.

    Raises:
        ValueError: If no dataset is called one of ``names``.
    """
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
