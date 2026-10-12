"""Downloads that resume, retry and check what they got, using only the standard library.

A piece is a byte range of a file on a server, saved to a local file. Large pieces download as chunks of ``CHUNK``
bytes over a few parallel connections. Each chunk is a ``.part`` file that a later call continues with an HTTP Range
request, so an interrupted download picks up where it stopped. When every chunk is complete they are joined into the
piece while its checksum is computed, and the piece gets its final name only if its size and checksum match.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import List, Optional

CHUNK = 128 << 20  # bytes per request when a piece is split over several connections
BUFFER = 1 << 20
TIMEOUT = 60  # seconds a connection may stay silent before it is dropped and resumed
RETRIES = 5  # failed attempts in a row, without any progress, before a server is given up
BACKOFF = 2.0  # seconds before the first retry; doubles after each failure, up to 60 s
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)

_NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout, HTTPException, ssl.SSLError)


class DownloadError(OSError):
    """A download failed after every retry, or a server sent something other than the file that was asked for."""


class ChecksumError(DownloadError):
    """Downloaded bytes do not match the size or checksum pinned in the registry."""


class Cancelled(Exception):
    """Another download of the same fetch failed, or the user pressed Ctrl+C."""


def _user_agent() -> str:
    try:
        from importlib.metadata import version
        return f"pytomography/{version('pytomography')} (pytomography.datasets)"
    except Exception:
        return "pytomography (pytomography.datasets)"


USER_AGENT = _user_agent()


def size_text(n: float) -> str:
    """Bytes as text, in decimal units like the hosts use: 11.0 MB, 1.43 GB."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.{2 if unit == 'GB' and n < 1e11 else 1}f} {unit}"
    return f"{int(n)} B"


def new_hash(algorithm: str):
    return hashlib.new(algorithm, usedforsecurity=False)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = new_hash(algorithm)
    with open(path, "rb") as f:
        while buf := f.read(BUFFER):
            h.update(buf)
    return h.hexdigest()


def replace(src: Path, dst: Path) -> None:
    """os.replace, retried briefly on Windows, where a virus scanner may still hold a file that was just written."""
    for attempt in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


class Progress:
    """One progress bar for everything a call downloads, redrawn at most twice a second.

    In a terminal or a notebook the bar redraws in place; in a log (CI) it prints a line every 10%."""

    def __init__(self, total: int, enabled: bool = True, label: str = ""):
        self.total, self.enabled, self.label = total, enabled and total > 0, label
        self.done = self._first = 0
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._drawn = 0.0
        self._inline = sys.stdout.isatty() or "ipykernel" in sys.modules
        self._next_tenth = 1

    def start_at(self, done: int) -> None:
        """Bytes already on disk from an earlier, interrupted call."""
        self.done = self._first = done

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            now = time.monotonic()
            if self.enabled and (now - self._drawn >= 0.5 or self.done >= self.total):
                self._drawn = now
                self._draw(now)

    def _draw(self, now: float) -> None:
        fraction = min(self.done / self.total, 1.0)
        rate = (self.done - self._first) / max(now - self._start, 1e-3)
        line = (f"{self.label}[{'#' * int(30 * fraction):<30}] {size_text(self.done)}/{size_text(self.total)}"
                f" {size_text(rate)}/s")
        if self._inline:
            print("\r" + line + "   ", end="", flush=True)
        elif fraction * 10 >= self._next_tenth or fraction >= 1:
            print(line, flush=True)
            self._next_tenth = int(fraction * 10) + 1

    def close(self) -> None:
        if self.enabled and self._inline and self._drawn:
            print(flush=True)


@dataclass
class Piece:
    """Bytes ``start`` to ``start + size`` of a file on a server, to be saved as ``dest``.

    ``urls`` are mirrors of the same file, tried in order. ``checksum`` is ``"sha256:<hex>"`` or ``"md5:<hex>"``;
    ``None`` only when pinning a new piece, which then gets ``digest`` set to its sha256. ``total`` is the size of the
    whole file on the server, which every response is checked against."""
    urls: List[str]
    start: int
    size: int
    dest: Path
    checksum: Optional[str]
    total: Optional[int] = None
    used: Optional[str] = None  # the URL that served the last chunk
    digest: Optional[str] = None

    def chunks(self) -> List[tuple]:
        """(first byte, end byte, local file) of every chunk; one chunk if the piece is small."""
        bounds = range(self.start, self.start + self.size, CHUNK) if self.size > CHUNK else [self.start]
        out = []
        for lo in bounds:
            hi = min(lo + CHUNK, self.start + self.size) if self.size > CHUNK else self.start + self.size
            out.append((lo, hi, self.dest.with_name(f"{self.dest.name}.{lo}-{hi}.part")))
        return out


def _skip(response, start: int, total: Optional[int], url: str) -> int:
    """Check a response to a Range request; returns how many bytes to drop from its start."""
    if response.status == 206:
        m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
        if not m or int(m.group(1)) != start:
            raise DownloadError(f"{url}: the server sent the wrong bytes ({response.headers.get('Content-Range')!r}"
                                f" for a request from byte {start:,})")
        if total is not None and m.group(3) != "*" and int(m.group(3)) != total:
            raise ChecksumError(f"{url} is {int(m.group(3)):,} bytes, not {total:,}: the file on the server has"
                                " changed since it was pinned in the registry")
        return 0
    length = response.headers.get("Content-Length")
    if total is not None and length is not None and int(length) != total:
        raise ChecksumError(f"{url} is {int(length):,} bytes, not {total:,}: the file on the server has changed"
                            " since it was pinned in the registry")
    return start  # the server ignored the Range header and sends the whole file


def _retry_after(headers) -> Optional[float]:
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _get(url: str, lo: int, hi: int, path: Path, total: Optional[int], progress: Progress,
         cancel: threading.Event) -> None:
    """Fill ``path`` with bytes [lo, hi) of ``url``, continuing from the bytes it already holds."""
    want = hi - lo
    have = path.stat().st_size if path.exists() else 0
    failures, delay = 0, BACKOFF
    while have < want:
        if cancel.is_set():
            raise Cancelled()
        before, pause, error = have, None, None
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={lo + have}-{hi - 1}", "User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                skip = _skip(response, lo + have, total, url)
                with open(path, "ab") as f:
                    while have < want:
                        if cancel.is_set():
                            raise Cancelled()
                        buf = response.read(BUFFER)
                        if not buf:
                            break
                        if skip:
                            if len(buf) <= skip:
                                skip -= len(buf)
                                continue
                            buf, skip = buf[skip:], 0
                        buf = buf[:want - have]
                        f.write(buf)
                        have += len(buf)
                        progress.add(len(buf))
            if have < want:
                error = "the connection closed early"
        except urllib.error.HTTPError as e:
            if e.code == 416:
                raise ChecksumError(f"{url} is shorter than expected: the file on the server has changed since it"
                                    " was pinned in the registry") from None
            if e.code not in RETRY_STATUS:
                raise DownloadError(f"{url}: HTTP {e.code} {e.reason}") from None
            pause, error = _retry_after(e.headers), f"HTTP {e.code} {e.reason}"
        except _NETWORK_ERRORS as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise DownloadError(f"{url}: {reason}. Python could not check the server's certificate; on macOS, run"
                                    " 'Install Certificates.command' in your Python folder.") from None
            error = f"{type(e).__name__}: {reason}"
        if have >= want:
            break
        if have > before:
            failures, delay = 0, BACKOFF  # it progressed, so count failures from zero again
        failures += 1
        if failures > RETRIES:
            raise DownloadError(f"{url}: {error} (gave up after {RETRIES} retries)")
        if cancel.wait(min(pause if pause is not None else delay, 120)):
            raise Cancelled()
        delay = min(delay * 2, 60)


def _ranges_supported(url: str) -> bool:
    """Whether the server answers Range requests, so a large piece can use several connections."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT,
                                                   "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status == 206
    except Exception:
        return True  # unknown; the chunks themselves will find out


def _join(piece: Piece, chunks: List[tuple]) -> None:
    """Join the downloaded chunks into ``piece.dest`` and check it, deleting each chunk once it is copied.

    Interrupted joins continue: ``.joining`` is cut back to the last whole chunk and the rest is appended."""
    algorithm, _, expected = (piece.checksum or "sha256:").partition(":")
    h = new_hash(algorithm)
    if len(chunks) == 1:
        joined = chunks[0][2]
        with open(joined, "rb") as f:
            while buf := f.read(BUFFER):
                h.update(buf)
    else:
        joined = piece.dest.with_name(piece.dest.name + ".joining")
        bounds = [lo - piece.start for lo, _, _ in chunks] + [piece.size]
        done = joined.stat().st_size if joined.exists() else 0
        done = max(b for b in bounds if b <= done)
        with open(joined, "r+b" if joined.exists() else "wb") as out:
            out.truncate(done)
            while out.tell() < done:
                h.update(out.read(min(BUFFER, done - out.tell())))
            for lo, hi, part in chunks:
                if lo - piece.start < done:
                    part.unlink(missing_ok=True)  # copied before an interruption
                    continue
                with open(part, "rb") as f:
                    while buf := f.read(BUFFER):
                        h.update(buf)
                        out.write(buf)
                part.unlink()
    size, digest = joined.stat().st_size, h.hexdigest()
    if size != piece.size or (expected and digest != expected):
        joined.unlink()
        for _, _, part in chunks:
            part.unlink(missing_ok=True)
        what = f"{size:,} bytes instead of {piece.size:,}" if size != piece.size else f"{algorithm} {digest}"
        raise ChecksumError(f"{piece.dest.name}: the download is {what}, which does not match the registry. It was"
                            " deleted; run fetch() again. If this keeps happening, the file on the server has changed:"
                            " please open an issue at https://github.com/PyTomography/PyTomography/issues")
    replace(joined, piece.dest)
    piece.digest = f"{algorithm}:{digest}"


def download(pieces: List[Piece], workers: int = 4, progress: bool = True, label: str = "") -> None:
    """Download every piece that is not at its destination yet, over at most ``workers`` connections at a time."""
    todo = [p for p in pieces if not (p.dest.exists() and p.dest.stat().st_size == p.size)]
    if not todo:
        return
    jobs, plans, first = [], [], 0
    for piece in todo:
        piece.dest.parent.mkdir(parents=True, exist_ok=True)
        chunks = piece.chunks()
        if len(chunks) > 1 and not _ranges_supported(piece.urls[0]):
            # one connection, which skips what is already there if the server ignores Range requests
            whole = piece.dest.with_name(f"{piece.dest.name}.{piece.start}-{piece.start + piece.size}.part")
            chunks = [(piece.start, piece.start + piece.size, whole)]
        names = {c[2].name for c in chunks}
        for stale in piece.dest.parent.glob(glob.escape(piece.dest.name) + ".*.part"):
            if stale.name not in names:
                stale.unlink()  # chunks of another chunk size, from an older version
        joining = piece.dest.with_name(piece.dest.name + ".joining")
        copied = joining.stat().st_size if joining.exists() else 0
        for lo, hi, part in chunks:
            if part.exists() and part.stat().st_size > hi - lo:
                part.unlink()
            if lo - piece.start + (hi - lo) <= copied:
                first += hi - lo
            else:
                first += part.stat().st_size if part.exists() else 0
                jobs.append((piece, lo, hi, part))
        plans.append((piece, chunks))

    bar = Progress(sum(p.size for p in todo), progress, label)
    bar.start_at(first)
    cancel = threading.Event()

    def run(piece, lo, hi, part):
        errors = []
        for url in piece.urls:
            try:
                _get(url, lo, hi, part, piece.total, bar, cancel)
                piece.used = url
                return
            except Cancelled:
                raise
            except DownloadError as e:
                errors.append(e)
                if isinstance(e, ChecksumError):
                    part.unlink(missing_ok=True)  # this mirror holds another file: start afresh from the next
        raise errors[-1] if len(errors) == 1 else DownloadError("; ".join(str(e) for e in errors))

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(run, *job) for job in jobs]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        failed = [f for f in done if f.exception() is not None]
        if failed:
            cancel.set()
            wait(pending)
            bar.close()
            raise failed[0].exception()
    except KeyboardInterrupt:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)
        bar.close()
        raise
    finally:
        pool.shutdown(wait=True)
    bar.close()
    for piece, chunks in plans:
        _join(piece, chunks)
