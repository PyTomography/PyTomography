"""What fetch() does with each kind of part in the registry: which bytes to download, how to put them into the
dataset folder, and how to recognise files that are already there."""
from __future__ import annotations

import hashlib
import posixpath
import shutil
import urllib.parse
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import _zip
from ._download import ChecksumError, DownloadError, Piece, download, file_digest, replace
from ._store import DOWNLOADS, check_paths

FILES = Path(__file__).parent / "files"
TAIL = 64 << 10  # bytes read from the end of a zip on a server to find its central directory
HOSTS = {"zenodo.org": "Zenodo", "github.com": "GitHub", "huggingface.co": "Hugging Face"}


class Context:
    """The dataset folder being fetched, and how."""

    def __init__(self, name: str, folder: Path, workers: int = 4, progress: bool = True):
        self.name, self.folder, self.workers, self.progress = name, folder, workers, progress
        self.downloads = folder / DOWNLOADS

    def say(self, text: str) -> None:
        if self.progress:
            print(text, flush=True)


def urls(part: dict) -> List[str]:
    return list(part.get("urls") or [part["url"]])


def file_name(url: str) -> str:
    return urllib.parse.unquote(posixpath.basename(urllib.parse.urlparse(url).path))


def checksum(part: dict) -> str:
    return next(f"{a}:{part[a]}" for a in ("sha256", "md5") if a in part)


def host(part: dict) -> str:
    if part["kind"] == "package":
        return "the PyTomography package"
    if part["kind"] == "idc":
        return "the NCI Imaging Data Commons"
    netloc = urllib.parse.urlparse(urls(part)[0]).netloc
    return HOSTS.get(netloc, netloc)


def download_size(part: dict) -> int:
    """Bytes fetched from a server: for a zip_range part, its ranges and the zip's central directory."""
    if part["kind"] == "zip_range":
        return part["size"] - part["index"][0] + sum(hi - lo for lo, hi, _ in part["ranges"])
    return 0 if part["kind"] == "package" else part["size"]


def disk_size(part: dict) -> int:
    """Bytes the part takes in the dataset folder."""
    return part["unpacked"] if part["kind"] in ("zip", "zip_range") else part["size"]


def describe(part: dict) -> str:
    kind, where = part["kind"], urls(part)[0] if "url" in part or "urls" in part else ""
    if kind == "zip":
        return f"{file_name(where)}, unpacked ({where})"
    if kind == "zip_range":
        spans = ", ".join(f"{lo:,}-{hi:,}" for lo, hi, _ in part["ranges"])
        return f"{part['prefix']} from {file_name(where)}, bytes {spans} ({where})"
    if kind == "file":
        return f"{target(part)} ({where})"
    if kind == "package":
        return f"{target(part)} (shipped with PyTomography)"
    return f"{part['to']}/: IDC series {part['series']} ({part['files']:,} files, as of IDC {part['release']})"


def target(part: dict) -> str:
    return part.get("to") or (part["file"] if part["kind"] == "package" else file_name(urls(part)[0]))


def content_hash(digests) -> str:
    """sha256 of the sorted sha256 digests of a set of files: independent of their names and order."""
    return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()


def _download_path(ctx: Context, name: str, digest: Optional[str]) -> Path:
    """Where a piece is downloaded to. The name holds the start of its checksum, so a download left by an older
    registry entry is never taken for this one."""
    return ctx.downloads / f"{name}.{(digest or 'unpinned').rpartition(':')[2][:12]}"


# -- zip: an archive, downloaded and unpacked ---------------------------------------------------------------------

def _local_archive(part: dict, ctx: Context) -> Optional[Path]:
    """The archive itself, if it is already in the dataset folder (the old instructions said to download it there)."""
    path = ctx.folder / file_name(urls(part)[0])
    return path if path.is_file() and path.stat().st_size == part["size"] else None


def zip_pieces(part: dict, ctx: Context) -> List[Piece]:
    if _local_archive(part, ctx):
        return []
    dest = _download_path(ctx, file_name(urls(part)[0]), checksum(part))
    return [Piece(urls(part), 0, part["size"], dest, checksum(part), part["size"])]


def zip_install(part: dict, ctx: Context, pieces: List[Piece]) -> Tuple[Dict[str, list], str]:
    if pieces:
        archive, source = pieces[0].dest, pieces[0].used or urls(part)[0]
    else:
        archive = source = _local_archive(part, ctx)
        algorithm, _, expected = checksum(part).partition(":")
        ctx.say(f"Checking {archive.name}, which is already in the dataset folder")
        if file_digest(archive, algorithm) != expected:
            raise ChecksumError(f"{archive} is not the archive in the registry ({algorithm} does not match). Delete"
                                " it and run fetch() again to download it.")
    with zipfile.ZipFile(archive) as zf:
        files = _zip.unpack(zf, _zip.select(zf, part.get("prefix", ""), part.get("exclude", ())), ctx.folder,
                            ctx.progress)
    if archive.parent == ctx.downloads:
        archive.unlink()  # never an archive the user put there
    return files, str(source)


def _remote_index(part: dict, ctx: Context) -> List[Tuple[zipfile.ZipInfo, str]]:
    """The members the part keeps, from the central directory at the end of the zip on the server: Range requests
    for its last bytes, and for more if the directory starts earlier."""
    size, start = part["size"], max(0, part["size"] - TAIL)
    for _ in range(3):
        tail = Piece(urls(part), start, size - start, ctx.downloads / f"{file_name(urls(part)[0])}.end", None, size)
        download([tail], ctx.workers, progress=False)
        try:
            with _zip.open_pieces(size, [(start, tail.dest)]) as zf:
                return [(info, rel) for info, rel in _zip.select(zf, part.get("prefix", ""), part.get("exclude", ()))]
        except _zip.Missing as e:
            start = e.offset  # the central directory starts earlier: fetch from there
        finally:
            tail.dest.unlink(missing_ok=True)
    raise DownloadError(f"could not read the central directory of {urls(part)[0]}")


def zip_adopt(part: dict, ctx: Context) -> Optional[Dict[str, list]]:
    local = _local_archive(part, ctx)
    if local is None:
        return _zip.matches(_remote_index(part, ctx), ctx.folder)
    with zipfile.ZipFile(local) as zf:  # its own index, before zip_install checks its checksum
        return _zip.matches(_zip.select(zf, part.get("prefix", ""), part.get("exclude", ())), ctx.folder)


# -- zip_range: members of a larger zip, from byte ranges of it ---------------------------------------------------

def range_pieces(part: dict, ctx: Context) -> List[Piece]:
    name, (start, sha) = file_name(urls(part)[0]), part["index"]
    index = Piece(urls(part), start, part["size"] - start, _download_path(ctx, f"{name}.index", sha),
                  "sha256:" + sha, part["size"])
    return [index] + [Piece(urls(part), lo, hi - lo, _download_path(ctx, f"{name}.{lo}-{hi}", digest),
                            "sha256:" + digest, part["size"]) for lo, hi, digest in part["ranges"]]


def range_install(part: dict, ctx: Context, pieces: List[Piece]) -> Tuple[Dict[str, list], str]:
    ranges = pieces[1:]
    with _zip.open_pieces(part["size"], [(p.start, p.dest) for p in pieces]) as zf:
        members = _zip.select(zf, part["prefix"], part.get("exclude", ()))
        ends = _zip.member_ends(zf, part["index"][0])
        for info, _ in members:
            if not any(p.start <= info.header_offset and ends[info.filename] <= p.start + p.size for p in ranges):
                raise ValueError(f"registry error: {info.filename} is outside the byte ranges given for {ctx.name}")
        files = _zip.unpack(zf, members, ctx.folder, ctx.progress)
    for p in pieces:
        p.dest.unlink()
    return files, ranges[0].used or urls(part)[0]


def range_adopt(part: dict, ctx: Context) -> Optional[Dict[str, list]]:
    index = range_pieces(part, ctx)[0]
    download([index], ctx.workers, progress=False)  # pinned and small; install reuses it if the files do not match
    with _zip.open_pieces(part["size"], [(index.start, index.dest)]) as zf:
        files = _zip.matches(_zip.select(zf, part["prefix"], part.get("exclude", ())), ctx.folder)
    if files is not None:
        index.dest.unlink()
    return files


# -- file: one file -----------------------------------------------------------------------------------------------

def file_pieces(part: dict, ctx: Context) -> List[Piece]:
    dest = _download_path(ctx, target(part).replace("/", "__"), checksum(part))
    return [Piece(urls(part), 0, part["size"], dest, checksum(part), part["size"])]


def file_install(part: dict, ctx: Context, pieces: List[Piece]) -> Tuple[Dict[str, list], str]:
    check_paths(ctx.folder, [target(part)])
    path = ctx.folder.joinpath(*target(part).split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    replace(pieces[0].dest, path)
    return {target(part): [part["size"], checksum(part)]}, pieces[0].used or urls(part)[0]


def file_adopt(part: dict, ctx: Context) -> Optional[Dict[str, list]]:
    path = ctx.folder.joinpath(*target(part).split("/"))
    algorithm, _, expected = checksum(part).partition(":")
    if path.is_file() and path.stat().st_size == part["size"] and file_digest(path, algorithm) == expected:
        return {target(part): [part["size"], checksum(part)]}
    return None


# -- package: a small file shipped with PyTomography ---------------------------------------------------------------

def package_install(part: dict, ctx: Context, pieces: List[Piece]) -> Tuple[Dict[str, list], str]:
    source = FILES / part["file"]
    if file_digest(source) != part["sha256"]:
        raise ChecksumError(f"{source} does not match the registry: this copy of PyTomography is damaged. Reinstall"
                            " it.")
    path = ctx.folder.joinpath(*target(part).split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".copying")
    shutil.copyfile(source, tmp)
    replace(tmp, path)
    return {target(part): [part["size"], "sha256:" + part["sha256"]]}, "the PyTomography package"


def package_adopt(part: dict, ctx: Context) -> Optional[Dict[str, list]]:
    path = ctx.folder.joinpath(*target(part).split("/"))
    if path.is_file() and path.stat().st_size == part["size"] and file_digest(path) == part["sha256"]:
        return {target(part): [part["size"], "sha256:" + part["sha256"]]}
    return None


# -- idc: a DICOM series from the NCI Imaging Data Commons ----------------------------------------------------------

def _series_files(folder: Path) -> Tuple[Dict[Path, str], int]:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    return {p: file_digest(p) for p in files}, sum(p.stat().st_size for p in files)


def idc_install(part: dict, ctx: Context, pieces: List[Piece]) -> Tuple[Dict[str, list], str]:
    try:
        from idc_index import IDCClient
    except ImportError:
        raise ImportError(f"{ctx.name} comes from the NCI Imaging Data Commons, through its Python client. Install it"
                          " first:\n    pip install idc-index") from None
    tmp = ctx.downloads / f"idc-{part['series'].rsplit('.', 1)[-1]}"
    tmp.mkdir(parents=True, exist_ok=True)
    ctx.say(f"Downloading IDC series {part['series']} with idc-index")
    IDCClient().download_from_selection(downloadDir=str(tmp), seriesInstanceUID=[part["series"]], dirTemplate=None,
                                        use_s5cmd_sync=True, show_progress_bar=ctx.progress, quiet=True)
    digests, size = _series_files(tmp)
    if len(digests) != part["files"] or size != part["size"] or content_hash(digests.values()) != part["sha256"]:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ChecksumError(
            f"IDC's copy of series {part['series']} has changed since IDC {part['release']}, when it was pinned in the"
            f" registry: {len(digests):,} files and {size:,} bytes instead of {part['files']:,} and {part['size']:,}"
            f"{', and other content' if len(digests) == part['files'] and size == part['size'] else ''}. Please"
            " open an issue at https://github.com/PyTomography/PyTomography/issues")
    rels = {p: posixpath.join(part["to"], p.relative_to(tmp).as_posix()) for p in digests}
    check_paths(ctx.folder, rels.values())
    files = {}
    for p, digest in digests.items():
        rel = rels[p]
        dest = ctx.folder.joinpath(*rel.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        files[rel] = [p.stat().st_size, "sha256:" + digest]
        replace(p, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return files, f"IDC series {part['series']}"


def idc_adopt(part: dict, ctx: Context) -> Optional[Dict[str, list]]:
    folder = ctx.folder.joinpath(*part["to"].split("/"))
    if not folder.is_dir():
        return None
    files = [p for p in folder.rglob("*") if p.is_file()]
    if len(files) != part["files"] or sum(p.stat().st_size for p in files) != part["size"]:
        return None
    digests, _ = _series_files(folder)
    if content_hash(digests.values()) != part["sha256"]:
        return None
    return {posixpath.join(part["to"], p.relative_to(folder).as_posix()): [p.stat().st_size, "sha256:" + d]
            for p, d in digests.items()}


PIECES = {"zip": zip_pieces, "zip_range": range_pieces, "file": file_pieces,
          "package": lambda part, ctx: [], "idc": lambda part, ctx: []}
INSTALL = {"zip": zip_install, "zip_range": range_install, "file": file_install, "package": package_install,
           "idc": idc_install}
ADOPT = {"zip": zip_adopt, "zip_range": range_adopt, "file": file_adopt, "package": package_adopt, "idc": idc_adopt}
