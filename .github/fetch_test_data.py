"""Download the tutorial data that the regression tests read, for CI and for anyone without the full data.

    python .github/fetch_test_data.py ~/pytomography_data
    PYTOMOGRAPHY_DATA=~/pytomography_data pytest -m data

Downloads the PyTomography SPECT tutorial record from Zenodo, checks its checksum, and keeps only the folders the
tests need (about 0.2 GB). Does nothing if they are already there, so CI can cache the folder between runs.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "15314460"  # PyTomography SPECT tutorial data, CC BY 4.0
ARCHIVE = "SPECT.zip"
NEEDED = ["SPECT/Lu177-PSMA-GEDisc/"]


def main(dest: Path) -> None:
    if all((dest / folder).is_dir() for folder in NEEDED):
        print("test data already present")
        return
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}") as r:
        entry = next(f for f in json.load(r)["files"] if f["key"] == ARCHIVE)
    algorithm, expected = entry["checksum"].split(":")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE
    digest = hashlib.new(algorithm)
    print(f"downloading {ARCHIVE} ({entry['size'] / 1e9:.2f} GB) from Zenodo record {RECORD}")
    with urllib.request.urlopen(entry["links"]["self"]) as r, open(archive, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        archive.unlink()
        sys.exit(f"{ARCHIVE}: {algorithm} checksum does not match the record")
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if any(m.startswith(folder) for folder in NEEDED)]
        z.extractall(dest, members)
    archive.unlink()
    print(f"kept {len(members)} files under {', '.join(NEEDED)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
