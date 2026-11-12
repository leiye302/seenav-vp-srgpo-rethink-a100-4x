"""Fetch and SHA-verify the reviewed SeeNav Route-SFT dataset from GitHub Releases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile

from reproduce import ROOT, sha

LOCK = ROOT / "dataset_release.json"
METADATA = {"manifest.json", "train.jsonl", "validation.jsonl",
            "train_batches.jsonl", "validation_batches.jsonl"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def dest_path(root, name):
    raw = Path(name)
    path = root / raw
    check(not raw.is_absolute() and path.resolve().is_relative_to(root), "unsafe dataset path")
    check(not path.is_symlink(), "dataset path cannot be a symlink")
    return path


def validate_index(spec):
    check(spec["schema"] == "seenav_route_sft_github_release_v1"
          and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec["repository"])
          and re.fullmatch(r"[A-Za-z0-9_.-]+", spec["tag"])
          and re.fullmatch(r"[0-9a-f]{64}", spec["manifest_sha256"]), "bad release lock")
    names, kinds = set(), set()
    for i, a in enumerate(spec["assets"]):
        n, p, kind = a["name"], a["path"], a["kind"]
        check(re.fullmatch(r"[A-Za-z0-9_.-]+\.zip", n) and n not in names
              and re.fullmatch(r"[0-9a-f]{64}", a["sha256"]) and 0 < a["size"] < 2**31
              and ((i == 0 and kind == "metadata" and p == n)
                   or (i > 0 and kind == "media" and p == "media/" + n)), "bad release asset")
        names.add(n)
        kinds.add(kind)
    check(kinds == {"metadata", "media"}, "incomplete release assets")
    return spec


def same(path, spec):
    return path.is_file() and path.stat().st_size == spec["size"] and sha(path) == spec["sha256"]


def fetch(root, spec, asset):
    p = dest_path(root, asset["path"])
    if p.exists():
        check(same(p, asset), "existing dataset file differs: " + asset["name"])
        return p
    part = dest_path(root, asset["path"] + ".part")
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{spec['repository']}/releases/download/{spec['tag']}/{quote(asset['name'])}"
    for i in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset == asset["size"] and same(part, asset):
            part.replace(p)
            return p
        if offset >= asset["size"]:
            offset = 0  # restart an incomplete download whose bytes failed verification
        headers = {"User-Agent": "SeeNav-Route-SFT-Downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                if response.status == 206:
                    check(append and response.headers["Content-Range"].startswith(f"bytes {offset}-"),
                          "download server returned an unexpected byte range")
                written = offset if append else 0
                with part.open("ab" if append else "wb") as f:
                    for blob in iter(lambda: response.read(2*1024**2), b""):
                        written += len(blob)
                        check(written <= asset["size"], "download exceeds locked size")
                        f.write(blob)
            if same(part, asset):
                part.replace(p)
                return p
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or i == 2:
                raise RuntimeError(f"{asset['name']}: HTTP {exc.code}") from None
        except (URLError, OSError):
            if i == 2:
                raise
        if i < 2:
            time.sleep(1+i)
    raise ValueError("download does not match its SHA-256: " + asset["name"])


def unpack_metadata(root, path, spec):
    with ZipFile(path) as z:
        check(len(z.namelist()) == len(METADATA) and set(z.namelist()) == METADATA, "bad metadata archive")
        manifest = json.loads(z.read("manifest.json"))
        entries = {**manifest["files"], "manifest.json": {"size": z.getinfo("manifest.json").file_size,
                                                          "sha256": spec["manifest_sha256"]}}
        for asset in spec["assets"][1:]:
            check(entries[asset["path"]] == {k: asset[k] for k in ("size", "sha256")}, "media lock differs")
        check(set(entries) == METADATA | {a["path"] for a in spec["assets"][1:]}, "untracked data files")
        for name in sorted(METADATA):
            output = dest_path(root, name)
            if not output.exists():
                part = dest_path(root, name + ".part")
                with z.open(name) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                check(same(part, entries[name]), "metadata checksum differs: " + name)
                part.replace(output)
            check(same(output, entries[name]), "existing metadata differs: " + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--metadata-only", action="store_true", help="Download only the small JSONL metadata (not trainable)")
    args = p.parse_args()
    spec = validate_index(json.loads(LOCK.read_bytes()))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    check(1 <= args.jobs <= 8, "--jobs must be between 1 and 8")
    metadata = fetch(root, spec, spec["assets"][0])
    unpack_metadata(root, metadata, spec)
    if not args.metadata_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for path in pool.map(lambda a: fetch(root, spec, a), spec["assets"][1:]):
                print("Verified", path.name, flush=True)
    print("Verified metadata only (no frames downloaded)" if args.metadata_only
          else f"Verified complete SFT data at {root}", flush=True)


if __name__ == "__main__":
    main()
