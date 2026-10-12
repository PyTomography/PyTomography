"""What fetch() keeps in a dataset folder: a marker that records the parts already in place and the files each one
wrote, and a lock so that two processes never download the same dataset at once."""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List

from ._download import file_digest, replace

MARKER = ".pytomography-dataset.json"
LOCK = ".pytomography-dataset.lock"
DOWNLOADS = ".download"  # partial downloads and archives waiting to be unpacked
_NOT_IDENTITY = ("url", "urls", "note")  # a new mirror or note does not change what a part is
WINDOWS = os.name == "nt"


def part_id(part: dict) -> str:
    """A short hash of what a part contains, so that a changed registry entry is noticed and fetched again."""
    core = {k: v for k, v in part.items() if k not in _NOT_IDENTITY}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


class Marker:
    """``.pytomography-dataset.json``: {"dataset", "format", "parts": {part id: {"kind", "source", "at", "files"}}},
    where "files" maps each path in the dataset folder to [size, "<algorithm>:<hex>"]."""

    def __init__(self, folder: Path, name: str):
        self.path = folder / MARKER
        self.data = {"dataset": name, "format": 1, "parts": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf8"))
            if data.get("format") == 1 and isinstance(data.get("parts"), dict):
                self.data = data
        except (OSError, ValueError):
            pass

    @property
    def parts(self) -> Dict[str, dict]:
        return self.data["parts"]

    def add(self, pid: str, kind: str, source: str, files: Dict[str, list]) -> None:
        self.parts[pid] = {"kind": kind, "source": source, "files": files,
                           "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.save()

    def drop(self, pid: str) -> None:
        if self.parts.pop(pid, None) is not None:
            self.save()

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, separators=(",", ":")), encoding="utf8")  # compact: C145 lists 9,330 files
        replace(tmp, self.path)


def problems(folder: Path, files: Dict[str, list], checksums: bool = False) -> List[str]:
    """Files of a part that are missing, or whose size (or, with ``checksums``, content) has changed."""
    out = []
    for rel, (size, checksum) in files.items():
        path = folder.joinpath(*rel.split("/"))
        if not path.is_file():
            out.append(f"{rel}: missing")
        elif path.stat().st_size != size:
            out.append(f"{rel}: {path.stat().st_size:,} bytes instead of {size:,}")
        elif checksums:
            algorithm, _, expected = checksum.partition(":")
            if file_digest(path, algorithm) != expected:
                out.append(f"{rel}: content changed ({algorithm} does not match)")
    return out


def _long_paths_enabled() -> bool:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def check_paths(folder: Path, paths) -> None:
    """On Windows, refuse to write files whose paths would pass the 260-character limit, which programs reading the
    data would then also hit; unless long paths are enabled. Raised before anything is written."""
    if not WINDOWS:
        return
    longest = max(paths, key=len, default="")
    length = len(os.path.abspath(folder)) + 1 + len(longest) + len(".unpacking")
    if length > 259 and not _long_paths_enabled():
        raise OSError(
            f"Some files would have paths of {length} characters, such as {os.path.abspath(folder / longest)}, and"
            " Windows allows 260"
            " unless long paths are enabled. Set PYTOMOGRAPHY_DATA to a shorter folder, such as D:\\pytomography_data,"
            " or enable long paths (https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation),"
            " then run fetch() again. What was downloaded is kept in this folder for the next try.")


def user_files(folder: Path) -> bool:
    """Whether ``folder`` holds anything besides what fetch() itself keeps there."""
    return folder.is_dir() and any(p.name not in (MARKER, LOCK, DOWNLOADS, MARKER + ".tmp") for p in folder.iterdir())


@contextlib.contextmanager
def lock(folder: Path, name: str, progress: bool = True) -> Iterator[None]:
    """Hold the dataset's lock file; waits while another process holds it, such as a parallel test worker."""
    folder.mkdir(parents=True, exist_ok=True)
    f = open(folder / LOCK, "a+b")
    try:
        told = False
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if progress and not told:
                    print(f"{name}: waiting for another process that is downloading it", flush=True)
                    told = True
                time.sleep(0.5)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()
