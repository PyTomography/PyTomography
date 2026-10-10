"""Zip archives: reading members of a zip on a server from the byte ranges that were downloaded, and unpacking that
cannot write outside the dataset folder."""
from __future__ import annotations

import bisect
import contextlib
import fnmatch
import hashlib
import io
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from ._download import BUFFER, Progress, replace
from ._store import check_paths


class Missing(Exception):
    """zipfile asked for a byte of the remote zip that has not been downloaded.

    Not an OSError on purpose: zipfile turns OSErrors from reading the end of a zip into "File is not a zip file"."""

    def __init__(self, offset: int):
        super().__init__(f"byte {offset:,} of the zip has not been downloaded")
        self.offset = offset


class PiecesFile(io.RawIOBase):
    """A read-only view of a zip of ``size`` bytes made from the pieces of it that were downloaded.

    ``pieces`` are (offset of the piece in the zip, local file holding it). zipfile reads it as it would the whole
    archive, as long as the central directory and the members it opens are in pieces; reading a byte that no piece
    holds raises :class:`Missing`."""

    def __init__(self, size: int, pieces: Iterable[Tuple[int, Path]]):
        super().__init__()
        self._size = size
        self._pieces = sorted((start, Path(path), Path(path).stat().st_size) for start, path in pieces)
        self._starts = [p[0] for p in self._pieces]
        self._files = {}
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        if base + offset < 0:
            raise ValueError("negative seek position")
        self._pos = base + offset
        return self._pos

    def readinto(self, buffer):
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        i = bisect.bisect_right(self._starts, self._pos) - 1
        if i < 0 or self._pos >= self._pieces[i][0] + self._pieces[i][2]:
            raise Missing(self._pos)
        start, path, length = self._pieces[i]
        n = min(n, start + length - self._pos)  # stop at the end of the piece; the next read starts the next one
        f = self._files.get(path)
        if f is None:
            f = self._files[path] = open(path, "rb")
        f.seek(self._pos - start)
        got = f.readinto(memoryview(buffer)[:n])
        self._pos += got
        return got

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
        super().close()


@contextlib.contextmanager
def open_pieces(size: int, pieces: Iterable[Tuple[int, Path]]) -> Iterator[zipfile.ZipFile]:
    """zipfile on a :class:`PiecesFile`. Closes the pieces on exit, which zipfile leaves open for a file object."""
    stream = io.BufferedReader(PiecesFile(size, pieces), BUFFER)
    try:
        with zipfile.ZipFile(stream) as zf:
            yield zf
    finally:
        stream.close()


def safe_path(name: str) -> str:
    """``name`` as a relative path that stays inside the folder it is unpacked into, or ValueError."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if (name.startswith(("/", "\\")) or not parts or any(p == ".." for p in parts)
            or any(":" in p or "\0" in p for p in parts)):
        raise ValueError(f"unsafe path in the archive: {name!r}")
    return "/".join(parts)


def select(zf: zipfile.ZipFile, prefix: str = "", exclude: Iterable[str] = ()) -> List[Tuple[zipfile.ZipInfo, str]]:
    """(member, path in the dataset folder) for every file under ``prefix``, without the prefix, minus ``exclude``.

    ``exclude`` holds glob patterns matched against the path without the prefix; ``*`` also matches ``/``."""
    exclude = list(exclude)
    out = []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir() or not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in exclude):
            continue
        out.append((info, safe_path(rel)))
    return out


def member_ends(zf: zipfile.ZipFile, index_start: int) -> Dict[str, int]:
    """Where the bytes of each member end in the zip: the next member's header, or the central directory."""
    infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
    return {a.filename: (b.header_offset if b is not None else index_start)
            for a, b in zip(infos, infos[1:] + [None])}


def unpack(zf: zipfile.ZipFile, members: List[Tuple[zipfile.ZipInfo, str]], folder: Path,
           progress: bool = True) -> Dict[str, list]:
    """Unpack ``members`` into ``folder``; returns {path: [size, "sha256:<hex>"]} of every file written.

    zipfile checks each member's CRC-32 as it is read. Files are written under a temporary name and renamed when
    complete, so an interrupted unpack never leaves a truncated file under its real name."""
    check_paths(folder, [rel for _, rel in members])
    written = {}
    total = sum(info.file_size for info, _ in members)
    bar = Progress(total, progress and total >= 100e6, "Unpacking ")  # a bar only where unpacking takes a while
    for info, rel in members:
        target = folder.joinpath(*rel.split("/"))  # inside folder: safe_path() refused anything else
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".unpacking")
        h = hashlib.sha256()
        with zf.open(info) as src, open(tmp, "wb") as dst:
            while buf := src.read(BUFFER):
                h.update(buf)
                dst.write(buf)
                bar.add(len(buf))
        replace(tmp, target)
        written[rel] = [info.file_size, "sha256:" + h.hexdigest()]
    bar.close()
    return written


def matches(members: List[Tuple[zipfile.ZipInfo, str]], folder: Path) -> Optional[Dict[str, list]]:
    """If every member is already in ``folder`` with the right size and CRC-32, their manifest; otherwise None."""
    if not members:
        return None
    for info, rel in members:
        path = folder.joinpath(*rel.split("/"))
        if not path.is_file() or path.stat().st_size != info.file_size:
            return None
    written = {}
    for info, rel in members:
        crc, h = 0, hashlib.sha256()
        with open(folder.joinpath(*rel.split("/")), "rb") as f:
            while buf := f.read(BUFFER):
                crc = zlib.crc32(buf, crc)
                h.update(buf)
        if crc != info.CRC:
            return None
        written[rel] = [info.file_size, "sha256:" + h.hexdigest()]
    return written
