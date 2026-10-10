"""Tests of pytomography.datasets on the CPU, against a local HTTP server that can drop connections, ignore Range
requests, fail, or serve the wrong bytes. Nothing here touches the internet; test_datasets_links.py checks the real
hosts."""
import hashlib
import http.server
import io
import json
import re
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

from pytomography import datasets
from pytomography.datasets import _download, _parts, _store, registry
from pytomography.datasets.__main__ import main as cli
from pytomography.datasets._store import MARKER, part_id

REPO = Path(__file__).resolve().parents[3]


# -- a local server ---------------------------------------------------------------------------------------------

class Server:
    """Serves ``files`` over HTTP on localhost, honouring Range requests unless ``ranges`` is False.

    ``drops`` lists byte counts after which the next responses close the connection; ``statuses`` lists error codes
    to answer the next requests with; ``delay`` slows every response. ``requests`` logs (path, Range header)."""

    def __init__(self):
        self.files, self.requests, self.drops, self.statuses = {}, [], [], []
        self.ranges, self.delay = True, 0.0
        self.lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                with server.lock:
                    server.requests.append((self.path, self.headers.get("Range")))
                    status = server.statuses.pop(0) if server.statuses else None
                    drop = server.drops.pop(0) if server.drops and status is None else None
                time.sleep(server.delay)
                if status is not None:
                    self.send_response(status)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = server.files.get(self.path)
                if data is None:
                    self.send_error(404)
                    return
                start, end = 0, len(data)
                m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
                if m and server.ranges:
                    start = int(m.group(1))
                    end = min(int(m.group(2)) + 1 if m.group(2) else len(data), len(data))
                    if start >= len(data):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:end]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if drop is not None:
                    self.wfile.write(body[:drop])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"

    def add(self, path: str, data: bytes) -> str:
        self.files[path] = data
        return self.url(path)

    def gets(self, path: str):
        return [r for p, r in self.requests if p == path]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Retries without waiting, and small chunks so that small files use several connections."""
    monkeypatch.setattr(_download, "BACKOFF", 0.01)
    monkeypatch.setattr(_download, "TIMEOUT", 10)
    monkeypatch.setattr(_download, "CHUNK", 4096)


def use(monkeypatch, entries: dict) -> None:
    monkeypatch.setattr(registry, "DATASETS", entries)


def zip_bytes(files: dict, method=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def blob(n: int, seed: int = 0) -> bytes:
    """n bytes that do not compress, so archives are as large as their contents."""
    out, h = bytearray(), hashlib.sha256(str(seed).encode()).digest()
    while len(out) < n:
        h = hashlib.sha256(h).digest()
        out += h
    return bytes(out[:n])


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry(parts, **extra) -> dict:
    return {"title": "Test data", "source": "a test", "url": "https://example.org", "licence": "CC BY 4.0",
            "cite": "A. Tester. Test data (2026).", "tutorials": ["t_test"], "parts": parts, **extra}


CONTENT = {"T/a.txt": b"alpha" * 1000, "T/sub/b.bin": blob(20000), "T/old.pkl": b"pickle", "other/c.txt": b"c"}


def zip_part(server, path="/T.zip", files=CONTENT, **extra) -> dict:
    data = zip_bytes(files)
    return {"kind": "zip", "url": server.add(path, data), "size": len(data), "md5": md5(data),
            "unpacked": sum(len(v) for v in files.values()), **extra}


# -- downloading and unpacking ------------------------------------------------------------------------------------

def test_fetch_unpacks_checks_and_records(server, monkeypatch, tmp_path, capsys):
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/", exclude=["*.pkl"])])})
    folder = datasets.fetch("X/T", data_dir=tmp_path)
    assert folder == tmp_path / "X" / "T"
    assert (folder / "a.txt").read_bytes() == CONTENT["T/a.txt"]
    assert (folder / "sub" / "b.bin").read_bytes() == CONTENT["T/sub/b.bin"]
    assert not (folder / "old.pkl").exists() and not (folder / "other").exists()  # excluded, and outside the prefix
    assert not (folder / ".download").exists()
    marker = json.loads((folder / MARKER).read_text())
    [record] = marker["parts"].values()
    assert record["files"]["a.txt"] == [5000, "sha256:" + sha256(CONTENT["T/a.txt"])]
    out = capsys.readouterr().out
    assert "Licence: CC BY 4.0" in out and "A. Tester. Test data (2026)." in out

    before = len(server.requests)
    assert datasets.fetch("X/T", data_dir=tmp_path) == folder
    assert len(server.requests) == before  # the second call downloads nothing
    assert "already downloaded" in capsys.readouterr().out


def test_interrupted_download_resumes_with_a_range_request(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CHUNK", 1 << 20)  # one connection
    part = zip_part(server)
    use(monkeypatch, {"X/T": entry([part])})
    server.drops = [3000]
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert server.gets("/T.zip") == [f"bytes=0-{part['size'] - 1}", f"bytes=3000-{part['size'] - 1}"]
    assert (tmp_path / "X/T/T/a.txt").read_bytes() == CONTENT["T/a.txt"]


def test_server_that_ignores_range_requests(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CHUNK", 1 << 20)
    use(monkeypatch, {"X/T": entry([zip_part(server)])})
    server.ranges, server.drops = False, [3000]
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)  # the retry gets the whole file and skips 3000 bytes
    assert (tmp_path / "X/T/T/sub/b.bin").read_bytes() == CONTENT["T/sub/b.bin"]


def test_large_files_use_several_connections(server, monkeypatch, tmp_path):
    part = zip_part(server)
    assert part["size"] > 4 * _download.CHUNK
    use(monkeypatch, {"X/T": entry([part])})
    server.drops = [100, 100]  # two chunks are interrupted, and continue
    datasets.fetch("X/T", data_dir=tmp_path, progress=False, workers=3)
    ranges = server.gets("/T.zip")
    assert "bytes=0-0" in ranges  # the check that the server answers Range requests
    starts = sorted(int(r.split("=")[1].split("-")[0]) for r in ranges if r != "bytes=0-0")
    assert len(starts) >= part["size"] // _download.CHUNK
    assert (tmp_path / "X/T/T/sub/b.bin").read_bytes() == CONTENT["T/sub/b.bin"]


def test_wrong_checksum_deletes_the_download(server, monkeypatch, tmp_path):
    part = dict(zip_part(server), md5="0" * 32)
    use(monkeypatch, {"X/T": entry([part])})
    with pytest.raises(datasets.ChecksumError, match="does not match the registry"):
        datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    folder = tmp_path / "X" / "T"
    assert not (folder / "T").exists() and not (folder / MARKER).exists()
    assert not list((folder / ".download").iterdir())  # no partial files left to resume from


def test_file_changed_on_the_server(server, monkeypatch, tmp_path):
    part = zip_part(server)
    use(monkeypatch, {"X/T": entry([dict(part, size=part["size"] + 1)])})
    with pytest.raises(datasets.ChecksumError, match="changed since it was pinned"):
        datasets.fetch("X/T", data_dir=tmp_path, progress=False)


def test_retries_after_server_errors(server, monkeypatch, tmp_path):
    use(monkeypatch, {"X/T": entry([zip_part(server)])})
    server.statuses = [503, 429]
    datasets.fetch("X/T", data_dir=tmp_path, progress=False, workers=1)
    assert (tmp_path / "X/T/T/a.txt").exists()


def test_gives_up_after_repeated_failures(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "RETRIES", 2)
    use(monkeypatch, {"X/T": entry([zip_part(server)])})
    server.statuses = [503] * 50
    with pytest.raises(datasets.DownloadError, match="gave up after 2 retries"):
        datasets.fetch("X/T", data_dir=tmp_path, progress=False, workers=1)


def test_mirrors_are_tried_in_order(server, monkeypatch, tmp_path):
    part = zip_part(server)
    part["urls"] = [server.url("/missing.zip"), part.pop("url")]
    use(monkeypatch, {"X/T": entry([part])})
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert (tmp_path / "X/T/T/a.txt").exists()
    [record] = json.loads((tmp_path / "X/T" / MARKER).read_text())["parts"].values()
    assert record["source"].endswith("/T.zip")


def test_interrupted_join_continues(tmp_path):
    data = blob(10000)
    piece = _download.Piece(["unused"], 0, len(data), tmp_path / "f", "sha256:" + sha256(data))
    chunks = [(0, 4096, tmp_path / "f.0-4096.part"), (4096, 8192, tmp_path / "f.4096-8192.part"),
              (8192, 10000, tmp_path / "f.8192-10000.part")]
    for lo, hi, path in chunks[1:]:
        path.write_bytes(data[lo:hi])
    (tmp_path / "f.joining").write_bytes(data[:4096 + 1000])  # chunk 1 was copied and deleted; chunk 2 was half-way
    _download._join(piece, chunks)
    assert (tmp_path / "f").read_bytes() == data
    assert not any(p.exists() for _, _, p in chunks) and not (tmp_path / "f.joining").exists()


# -- members of a larger zip, from byte ranges ------------------------------------------------------------------

def three_folder_zip():
    files = {f"{d}/{i}.bin": blob(3000 + 500 * i, seed=ord(d) * 10 + i) for d in "ABC" for i in range(3)}
    data = zip_bytes(files, zipfile.ZIP_STORED)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        offsets = {i.filename: i.header_offset for i in zf.infolist()}
        index = zf.start_dir
    return files, data, offsets, index


def range_part(server, data, index, ranges, prefix, **extra):
    return {"kind": "zip_range", "url": server.add("/big.zip", data), "size": len(data),
            "index": [index, sha256(data[index:])], "prefix": prefix,
            "ranges": [[lo, hi, sha256(data[lo:hi])] for lo, hi in ranges], "unpacked": 0, **extra}


def test_zip_range_downloads_only_its_block(server, monkeypatch, tmp_path):
    files, data, offsets, index = three_folder_zip()
    lo, hi = offsets["B/0.bin"], offsets["C/0.bin"]
    use(monkeypatch, {"X/B": entry([range_part(server, data, index, [(lo, hi)], "B/")])})
    folder = datasets.fetch("X/B", data_dir=tmp_path, progress=False)
    assert sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")) == ["0.bin", "1.bin", "2.bin"]
    assert (folder / "2.bin").read_bytes() == files["B/2.bin"]
    served = sum(int(m.group(2)) - int(m.group(1)) + 1 for r in server.gets("/big.zip")
                 for m in [re.match(r"bytes=(\d+)-(\d+)", r)])
    assert served < (hi - lo) + (len(data) - index) + 2  # the block and the central directory, nothing else


def test_zip_range_skips_an_excluded_member(server, monkeypatch, tmp_path):
    files, data, offsets, index = three_folder_zip()
    ranges = [(offsets["B/0.bin"], offsets["B/1.bin"]), (offsets["B/2.bin"], offsets["C/0.bin"])]
    part = range_part(server, data, index, ranges, "B/", exclude=["1.bin"])
    use(monkeypatch, {"X/B": entry([part])})
    folder = datasets.fetch("X/B", data_dir=tmp_path, progress=False)
    assert (folder / "0.bin").exists() and (folder / "2.bin").exists() and not (folder / "1.bin").exists()


def test_zip_range_that_misses_a_member_is_a_registry_error(server, monkeypatch, tmp_path):
    files, data, offsets, index = three_folder_zip()
    part = range_part(server, data, index, [(offsets["B/0.bin"], offsets["B/2.bin"])], "B/")
    use(monkeypatch, {"X/B": entry([part])})
    with pytest.raises(ValueError, match="registry error: B/2.bin"):
        datasets.fetch("X/B", data_dir=tmp_path, progress=False)


def test_unsafe_paths_in_archives_are_refused(server, monkeypatch, tmp_path):
    for bad in ("../evil.txt", "/abs.txt", "C:/drive.txt", "a/../../evil.txt"):
        use(monkeypatch, {"X/T": entry([zip_part(server, files={bad: b"x", "ok.txt": b"y"})])})
        with pytest.raises(ValueError, match="unsafe path"):
            datasets.fetch("X/T", data_dir=tmp_path / "data", progress=False, force=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]


# -- other kinds of part ------------------------------------------------------------------------------------------

def test_file_and_package_parts(server, monkeypatch, tmp_path):
    data = blob(5000)
    model = _parts.FILES / "ac225_psf_model.json"
    use(monkeypatch, {"X/F": entry([
        {"kind": "file", "url": server.add("/f/raw.dat", data), "size": len(data), "md5": md5(data),
         "to": "sub/raw.dat"},
        {"kind": "package", "file": model.name, "size": model.stat().st_size,
         "sha256": sha256(model.read_bytes())}])})
    folder = datasets.fetch("X/F", data_dir=tmp_path, progress=False)
    assert (folder / "sub" / "raw.dat").read_bytes() == data
    assert (folder / "ac225_psf_model.json").read_bytes() == model.read_bytes()


def test_damaged_package_file_is_reported(monkeypatch, tmp_path):
    model = _parts.FILES / "ac225_psf_model.json"
    use(monkeypatch, {"X/F": entry([{"kind": "package", "file": model.name, "size": model.stat().st_size,
                                     "sha256": "0" * 64}])})
    with pytest.raises(datasets.ChecksumError, match="Reinstall"):
        datasets.fetch("X/F", data_dir=tmp_path, progress=False)


def fake_idc(monkeypatch, files):
    """An idc_index module whose client writes ``files`` instead of downloading a series."""
    calls = []

    class IDCClient:
        def download_from_selection(self, downloadDir, seriesInstanceUID, dirTemplate, **kwargs):
            calls.append((seriesInstanceUID, dirTemplate))
            for name, data in files.items():
                (Path(downloadDir) / name).write_bytes(data)

    monkeypatch.setitem(sys.modules, "idc_index", types.SimpleNamespace(IDCClient=IDCClient))
    return calls


def idc_part(files, **extra):
    return {"kind": "idc", "series": "1.2.3.4", "crdc": "x", "release": "v24", "to": "series", "files": len(files),
            "size": sum(len(v) for v in files.values()),
            "sha256": _parts.content_hash(sha256(v) for v in files.values()), **extra}


def test_idc_series_is_checked(monkeypatch, tmp_path):
    files = {f"{i}.dcm": blob(100 + i, seed=i) for i in range(5)}
    calls = fake_idc(monkeypatch, files)
    use(monkeypatch, {"X/CT": entry([idc_part(files)])})
    folder = datasets.fetch("X/CT", data_dir=tmp_path, progress=False)
    assert calls == [(["1.2.3.4"], None)]
    assert sorted(p.name for p in (folder / "series").iterdir()) == sorted(files)

    use(monkeypatch, {"X/CT2": entry([idc_part(files, sha256="0" * 64)])})
    with pytest.raises(datasets.ChecksumError, match="has changed since IDC v24"):
        datasets.fetch("X/CT2", data_dir=tmp_path, progress=False)


def test_idc_needs_its_client(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "idc_index", None)  # import fails
    use(monkeypatch, {"X/CT": entry([idc_part({"a.dcm": b"a"})])})
    with pytest.raises(ImportError, match="pip install idc-index"):
        datasets.fetch("X/CT", data_dir=tmp_path, progress=False)


# -- data that is already there -----------------------------------------------------------------------------------

def test_files_already_in_the_folder_are_checked_and_kept(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_parts, "TAIL", 1024)
    part = zip_part(server, prefix="T/", exclude=["*.pkl"])
    use(monkeypatch, {"X/T": entry([part])})
    folder = tmp_path / "X" / "T"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_bytes(CONTENT["T/a.txt"])
    (folder / "sub" / "b.bin").write_bytes(CONTENT["T/sub/b.bin"])
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert all(r and not r.startswith("bytes=0-") for r in server.gets("/T.zip"))  # only the end, for its index
    [record] = json.loads((folder / MARKER).read_text())["parts"].values()
    assert record["source"] == "already in the folder"


def test_wrong_files_in_the_folder_are_replaced(server, monkeypatch, tmp_path):
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/", exclude=["*.pkl"])])})
    folder = tmp_path / "X" / "T"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_bytes(CONTENT["T/a.txt"])
    (folder / "sub" / "b.bin").write_bytes(b"\0" * len(CONTENT["T/sub/b.bin"]))  # right size, wrong content
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert (folder / "sub" / "b.bin").read_bytes() == CONTENT["T/sub/b.bin"]


def test_archive_already_in_the_folder_is_unpacked(server, monkeypatch, tmp_path):
    part = zip_part(server)
    use(monkeypatch, {"X/T": entry([part])})
    folder = tmp_path / "X" / "T"
    folder.mkdir(parents=True)
    (folder / "T.zip").write_bytes(server.files["/T.zip"])
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert server.gets("/T.zip") == []
    assert (folder / "T" / "a.txt").exists() and (folder / "T.zip").exists()  # the user's archive is kept


def test_changed_files_are_fetched_again(server, monkeypatch, tmp_path):
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/")])})
    folder = datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    (folder / "a.txt").write_bytes(b"short")
    assert datasets.verify("X/T", data_dir=tmp_path)["X/T"] == ["a.txt: 5 bytes instead of 5,000"]
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert (folder / "a.txt").read_bytes() == CONTENT["T/a.txt"]

    (folder / "a.txt").write_bytes(b"A" * 5000)  # same size: only a checksum notices
    assert datasets.verify("X/T", data_dir=tmp_path) == {"X/T": []}
    assert datasets.verify("X/T", level="hash", data_dir=tmp_path)["X/T"] == [
        "a.txt: content changed (sha256 does not match)"]
    datasets.fetch("X/T", data_dir=tmp_path, progress=False, verify="hash")
    assert (folder / "a.txt").read_bytes() == CONTENT["T/a.txt"]


def test_new_registry_entry_replaces_the_old_files(server, monkeypatch, tmp_path):
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/")])})
    folder = datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    new = {"T/a.txt": b"new alpha", "T/c.txt": b"gamma"}
    use(monkeypatch, {"X/T": entry([zip_part(server, path="/T2.zip", files=new, prefix="T/")])})
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert (folder / "a.txt").read_bytes() == b"new alpha" and (folder / "c.txt").exists()
    assert not (folder / "sub" / "b.bin").exists() and not (folder / "old.pkl").exists()
    assert len(json.loads((folder / MARKER).read_text())["parts"]) == 1


def test_parallel_calls_download_once(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CHUNK", 1 << 20)
    use(monkeypatch, {"X/T": entry([zip_part(server)])})
    server.delay = 0.3
    errors = []

    def run():
        try:
            datasets.fetch("X/T", data_dir=tmp_path, progress=False)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(server.gets("/T.zip")) == 1


# -- the rest of the API ------------------------------------------------------------------------------------------

def test_extras_download_only_when_asked(server, monkeypatch, tmp_path):
    extra = zip_part(server, path="/E.zip", files={"E/e.txt": b"extra"}, prefix="E/")
    extras = {"more": {"title": "More", "parts": [extra]}}
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/")], extras=extras)})
    folder = datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert not (folder / "e.txt").exists()
    datasets.fetch("X/T", extras=["more"], data_dir=tmp_path, progress=False)
    assert (folder / "e.txt").read_bytes() == b"extra"
    with pytest.raises(ValueError, match="no extra called 'less'"):
        datasets.fetch("X/T", extras="less", data_dir=tmp_path)


def test_unknown_and_pending_datasets(monkeypatch, tmp_path):
    use(monkeypatch, {"SPECT/Lu177-NEMA-SymT2": entry([]), "PET/Later": entry([], status="pending", note="Soon.")})
    with pytest.raises(ValueError, match="Did you mean 'SPECT/Lu177-NEMA-SymT2'"):
        datasets.fetch("spect/lu177-nema-symt2", data_dir=tmp_path)
    with pytest.raises(datasets.DatasetNotAvailable, match="cannot be downloaded yet. Soon."):
        datasets.fetch("PET/Later", data_dir=tmp_path)
    assert datasets.info("PET/Later", data_dir=tmp_path)["state"] == "not published yet"
    (tmp_path / "PET" / "Later").mkdir(parents=True)
    (tmp_path / "PET" / "Later" / "copied.root").write_bytes(b"by hand")  # e.g. from the old shared folder
    assert datasets.fetch("PET/Later", data_dir=tmp_path) == tmp_path / "PET" / "Later"


def test_path_info_and_available(server, monkeypatch, tmp_path):
    part = zip_part(server, prefix="T/")
    use(monkeypatch, {"X/T": entry([part])})
    with pytest.raises(FileNotFoundError, match="Run datasets.fetch"):
        datasets.path("X/T", data_dir=tmp_path)
    assert datasets.available(tmp_path).loc[0, "state"] == "not downloaded"
    datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert datasets.path("X/T", data_dir=tmp_path) == tmp_path / "X" / "T"
    d = datasets.info("X/T", data_dir=tmp_path)
    assert d["state"] == "downloaded" and d["download_bytes"] == part["size"] and d["disk_bytes"] == part["unpacked"]
    assert datasets.available(tmp_path).loc[0, "state"] == "downloaded"


def test_command_line(server, monkeypatch, tmp_path, capsys):
    use(monkeypatch, {"X/T": entry([zip_part(server, prefix="T/")])})
    assert cli(["fetch", "--tutorial", "t_test", "--quiet", "--data-dir", str(tmp_path)]) == 0
    assert (tmp_path / "X/T/a.txt").exists()
    capsys.readouterr()
    assert cli(["list", "--data-dir", str(tmp_path)]) == 0
    assert re.search(r"X/T\s+\S+ kB\s+CC BY 4.0\s+downloaded", capsys.readouterr().out)
    assert cli(["verify", "--data-dir", str(tmp_path)]) == 0
    assert cli(["cache-key", "X/T"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert re.fullmatch(r"[0-9a-f]{16}", key) and key == datasets.cache_key("X/T")
    assert cli(["fetch", "X/Nope", "--data-dir", str(tmp_path)]) == 1


def test_paths_too_long_for_windows_are_refused_before_writing(server, monkeypatch, tmp_path):
    monkeypatch.setattr(_store, "WINDOWS", True)
    monkeypatch.setattr(_store, "_long_paths_enabled", lambda: False)
    use(monkeypatch, {"X/T": entry([zip_part(server, files={"d/" + "n" * 250 + ".dcm": b"x"})])})
    with pytest.raises(OSError, match="Set PYTOMOGRAPHY_DATA to a shorter folder"):
        datasets.fetch("X/T", data_dir=tmp_path, progress=False)
    assert not (tmp_path / "X/T/d").exists()
    monkeypatch.setattr(_store, "_long_paths_enabled", lambda: True)
    _store.check_paths(tmp_path, ["n" * 300])


def test_data_and_output_folders(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTOMOGRAPHY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PYTOMOGRAPHY_OUTPUT", str(tmp_path / "out"))
    assert datasets.data_dir() == tmp_path / "data"
    assert datasets.output_dir("SPECT/X") == tmp_path / "out" / "SPECT" / "X" and (tmp_path / "out/SPECT/X").is_dir()
    monkeypatch.delenv("PYTOMOGRAPHY_DATA")
    assert datasets.data_dir() == Path.home() / "pytomography_data"


# -- the registry itself ------------------------------------------------------------------------------------------

KINDS = {"zip": {"url", "size", "unpacked"}, "zip_range": {"url", "size", "index", "prefix", "ranges", "unpacked"},
         "file": {"url", "size"}, "package": {"file", "size", "sha256"},
         "idc": {"series", "release", "to", "files", "size", "sha256"}}


@pytest.mark.parametrize("name", list(registry.DATASETS))
def test_registry_entry(name):
    e = registry.DATASETS[name]
    assert name.count("/") == 1 and name.split("/")[0] in ("SPECT", "PET", "CT")
    for key in ("title", "source", "url", "licence", "tutorials", "parts"):
        assert key in e, key
    assert e["url"].startswith("https://")
    if e.get("status") == "pending":
        assert e.get("note")
    else:
        assert e["parts"] and e.get("cite")
    ids = [part_id(p) for p in e["parts"]]
    assert len(ids) == len(set(ids))
    for part in e["parts"] + [p for x in e.get("extras", {}).values() for p in x["parts"]]:
        assert set(KINDS[part["kind"]]) <= set(part), part
        assert isinstance(part["size"], int) and part["size"] > 0
        assert "TODO" not in json.dumps(part)
        for url in _parts.urls(part) if "url" in part or "urls" in part else []:
            assert url.startswith("https://")
        if part["kind"] in ("zip", "file"):
            assert (re.fullmatch(r"[0-9a-f]{32}", part.get("md5", ""))
                    or re.fullmatch(r"[0-9a-f]{64}", part.get("sha256", "")))
        if part["kind"] == "zip_range":
            ends = [0]
            for lo, hi, digest in part["ranges"]:
                assert ends[-1] <= lo < hi <= part["index"][0] and re.fullmatch(r"[0-9a-f]{64}", digest)
                ends.append(hi)
            assert part["prefix"].endswith("/") and re.fullmatch(r"[0-9a-f]{64}", part["index"][1])
        if part["kind"] == "package":
            path = _parts.FILES / part["file"]
            # also catches line endings converted by git on Windows (see .gitattributes)
            assert path.stat().st_size == part["size"] and sha256(path.read_bytes()) == part["sha256"]


def test_registry_matches_the_tutorial_list():
    """Every tutorial's datasets in docs/source/tutorials/tutorials.yaml are the registry's, and the other way round."""
    tutorials_yaml = REPO / "docs" / "source" / "tutorials" / "tutorials.yaml"
    if not tutorials_yaml.exists():
        pytest.skip("no docs/source/tutorials/tutorials.yaml in this checkout")
    yaml = pytest.importorskip("yaml")
    sections = yaml.safe_load(tutorials_yaml.read_text(encoding="utf8"))["sections"]
    listed = {t["notebook"]: set(t.get("datasets", [])) for s in sections for t in s["tutorials"]}
    from_registry = {}
    for name, e in registry.DATASETS.items():
        for notebook in e["tutorials"]:
            from_registry.setdefault(notebook, set()).add(name)
    assert {k: v for k, v in listed.items() if v} == from_registry
