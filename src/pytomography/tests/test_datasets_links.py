"""Checks that every download in the dataset registry is still where the registry says, as pinned: each file answers
Range requests with the pinned size, each zip_range index has its pinned checksum and its ranges hold every member,
and each IDC series is unchanged. Talks to the data hosts, so it runs only with PYTOMOGRAPHY_NETWORK_TESTS=1."""
import urllib.request

import pytest

from pytomography.datasets import _download, _parts, _zip, registry

pytestmark = pytest.mark.network

PARTS = [(name, part) for name, e in registry.DATASETS.items()
         for part in e["parts"] + [p for x in e.get("extras", {}).values() for p in x["parts"]]]


def ids(parts):
    return [f"{name}:{_parts.describe(part).split(' (')[0]}" for name, part in parts]


DOWNLOADS = [(n, p) for n, p in PARTS if p["kind"] in ("zip", "zip_range", "file")]
RANGES = [(n, p) for n, p in PARTS if p["kind"] == "zip_range"]
ZIPS = [(n, p) for n, p in PARTS if p["kind"] == "zip"]
SERIES = [(n, p) for n, p in PARTS if p["kind"] == "idc"]


@pytest.mark.parametrize("name,part", DOWNLOADS, ids=ids(DOWNLOADS))
def test_file_is_there_with_the_pinned_size(name, part):
    for url in _parts.urls(part):
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": _download.USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 206, f"{url} does not answer Range requests"
            assert int(response.headers["Content-Range"].rsplit("/", 1)[1]) == part["size"], url


@pytest.mark.parametrize("name,part", RANGES, ids=ids(RANGES))
def test_zip_range_holds_every_member(name, part, tmp_path):
    index = _parts.range_pieces(part, _parts.Context(name, tmp_path, progress=False))[0]
    _download.download([index], progress=False)  # checks the pinned sha256
    with _zip.open_pieces(part["size"], [(index.start, index.dest)]) as zf:
        members = _zip.select(zf, part["prefix"], part.get("exclude", ()))
        ends = _zip.member_ends(zf, part["index"][0])
    assert sum(info.file_size for info, _ in members) == part["unpacked"]
    for info, _ in members:
        inside = any(lo <= info.header_offset and ends[info.filename] <= hi for lo, hi, _ in part["ranges"])
        assert inside, f"{info.filename} is outside the pinned ranges"


@pytest.mark.parametrize("name,part", ZIPS, ids=ids(ZIPS))
def test_zip_unpacks_to_the_pinned_size(name, part, tmp_path):
    members = _parts._remote_index(part, _parts.Context(name, tmp_path, progress=False))
    assert sum(info.file_size for info, _ in members) == part["unpacked"]


@pytest.mark.parametrize("name,part", SERIES, ids=ids(SERIES))
def test_idc_series_is_unchanged(name, part):
    idc_index = pytest.importorskip("idc_index")
    index = idc_index.IDCClient().index
    rows = index[index["SeriesInstanceUID"] == part["series"]]
    assert len(rows) == 1, f"IDC no longer has series {part['series']}"
    row = rows.iloc[0]
    assert row["crdc_series_uuid"] == part["crdc"], "IDC revised the series; pin the new version"
    assert int(row["instanceCount"]) == part["files"]
