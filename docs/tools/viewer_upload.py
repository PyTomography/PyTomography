"""Upload the tutorials' viewer images to the image host, and pin them in the docs.

    python docs/tools/viewer_upload.py RUN_DIR/viewer [MORE_DIRS ...] --prefix v4.0.0             # check, and show the plan
    python docs/tools/viewer_upload.py RUN_DIR/viewer --prefix v4.0.0 --upload                    # upload, then pin
    python docs/tools/viewer_upload.py RUN_DIR/viewer --prefix v4.0.0 --pin-only --revision SHA   # pin images already uploaded

Each folder holds one subfolder per tutorial, as docs/tools/run_tutorials.py exports them (viewer_export.py); when a
tutorial appears in several, the last folder wins. Every file is checked against its manifest (size and sha256) and
each tutorial against the size budget before anything is uploaded.

--upload puts the files on the Hugging Face dataset (--repo, default PyTomography/tutorial-images) under PREFIX/, in
one commit, with the token in HF_TOKEN; it needs `pip install huggingface_hub`. Pinning then writes
docs/source/tutorials/viewer_images.json: the host's URL template, that commit, the prefix, and a summary of each
tutorial. The docs read only that file, so each docs version keeps showing the images of its own commit. To host the
images elsewhere (GitHub Pages, say), upload them there and pin with --host and --revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HF_HOST = "https://huggingface.co/datasets/{repo}/resolve/{{revision}}/{{path}}"
INDEX = Path(__file__).resolve().parents[1] / "source" / "tutorials" / "viewer_images.json"
MAX_MB = 15.0


def collect(folders: list[Path]) -> dict[str, Path]:
    found = {}
    for folder in folders:
        for man in sorted(Path(folder).glob("*/manifest.json")):
            found[man.parent.name] = man.parent
    return found


def check(name: str, folder: Path, max_mb: float) -> tuple[dict, list[str]]:
    man = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    problems = []
    for L in man["layers"]:
        f = folder / L["file"]
        if not f.exists():
            problems.append(f"{L['file']} is missing")
            continue
        blob = f.read_bytes()
        if len(blob) != L["bytes"] or hashlib.sha256(blob).hexdigest() != L["sha256"]:
            problems.append(f"{L['file']} doesn't match its manifest")
    if not (folder / "thumb.png").exists():
        problems.append("thumb.png is missing")
    if man.get("bytes", 0) > max_mb * 1e6:
        problems.append(f"{man['bytes'] / 1e6:.1f} MB, over the {max_mb:.0f} MB budget")
    if man.get("tutorial") != name:
        problems.append(f"the manifest is for {man.get('tutorial')}")
    return man, problems


def files_of(folder: Path, man: dict) -> list[Path]:
    return [folder / "manifest.json", folder / "thumb.png"] + [folder / L["file"] for L in man["layers"]]


def upload(found: dict, mans: dict, repo: str, prefix: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to an upload token for the dataset first.")
    ops = [CommitOperationAdd(path_in_repo=f"{prefix}/{name}/{f.name}", path_or_fileobj=str(f))
           for name, folder in found.items() for f in files_of(folder, mans[name])]
    info = HfApi(token=token).create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                                            commit_message=f"Tutorial viewer images for {prefix} ({len(found)} tutorials)")
    return info.oid


def pin(found: dict, mans: dict, host: str, revision: str, prefix: str) -> None:
    tutorials = {}
    for name in sorted(found):
        man = mans[name]
        blob = (found[name] / "manifest.json").read_bytes()
        tutorials[name] = {"title": man.get("title", name), "description": man.get("description", ""),
                           "layers": [L.get("label", L["name"]) for L in man["layers"]], "bytes": man.get("bytes", 0),
                           "manifest_sha256": hashlib.sha256(blob).hexdigest(), "thumb": True}
    index = {"host": host, "revision": revision, "prefix": prefix, "tutorials": tutorials}
    INDEX.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf8", newline="\n")
    print(f"pinned {len(tutorials)} tutorials in {INDEX}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="viewer folders of tutorial runs (RUN_DIR/viewer)")
    ap.add_argument("--prefix", required=True, help="folder on the host for this release, e.g. v4.0.0")
    ap.add_argument("--repo", default="PyTomography/tutorial-images", help="Hugging Face dataset")
    ap.add_argument("--host", help="URL template with {revision} and {path} (default: the Hugging Face dataset)")
    ap.add_argument("--upload", action="store_true", help="upload to the Hugging Face dataset, then pin")
    ap.add_argument("--pin-only", action="store_true", help="pin images already on the host, at --revision")
    ap.add_argument("--revision", help="the host's commit to pin (with --pin-only)")
    ap.add_argument("--max-mb", type=float, default=MAX_MB, help="size budget per tutorial")
    args = ap.parse_args()

    found = collect(args.folders)
    if not found:
        sys.exit("No exported tutorials in those folders.")
    mans, bad, total = {}, False, 0
    for name, folder in found.items():
        man, problems = check(name, folder, args.max_mb)
        mans[name] = man
        total += man.get("bytes", 0)
        print(f"{'FAIL' if problems else 'ok  '} {name:28s} {man.get('bytes', 0) / 1e6:5.1f} MB  {folder}" +
              ("".join(f"\n       {p}" for p in problems)))
        bad |= bool(problems)
    print(f"{len(found)} tutorials, {total / 1e6:.1f} MB")
    if bad:
        print("Not uploading: fix the problems above, or export those tutorials again.")
        return 1
    host = args.host or HF_HOST.format(repo=args.repo)
    if args.pin_only:
        if not args.revision:
            sys.exit("--pin-only needs --revision, the host's commit.")
        pin(found, mans, host, args.revision, args.prefix)
    elif args.upload:
        revision = upload(found, mans, args.repo, args.prefix)
        print(f"uploaded to {args.repo} at {revision}")
        pin(found, mans, host, revision, args.prefix)
    else:
        print(f"Dry run. --upload would put {sum(len(files_of(f, mans[n])) for n, f in found.items())} files on "
              f"{args.repo} under {args.prefix}/ and pin them in {INDEX.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
