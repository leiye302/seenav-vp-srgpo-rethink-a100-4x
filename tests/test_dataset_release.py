import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from download_dataset import METADATA, fetch, unpack_metadata, validate_index


def attrs(raw):
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class Reply(BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class ReleaseTest(unittest.TestCase):
    def test_metadata_and_resumed_media_without_unsafe_extraction(self):
        media = b"tiny mock media asset"
        sample = b"{}\n"
        media_path = "media/images.zip"
        manifest = json.dumps({"files": {**{p: attrs(sample) for p in METADATA-{"manifest.json"}},
                                         media_path: attrs(media)}}).encode()
        packed = BytesIO()
        with ZipFile(packed, "w") as z:
            for p in METADATA:
                z.writestr(p, manifest if p == "manifest.json" else sample)
        archive = packed.getvalue()
        a = dict(name="metadata.zip", path="metadata.zip", kind="metadata", **attrs(archive))
        b = dict(name="images.zip", path=media_path, kind="media", **attrs(media))
        spec = validate_index(dict(schema="seenav_route_sft_github_release_v1", repository="example/test",
                                   tag="data-v2", assets=[a,b], manifest_sha256=attrs(manifest)["sha256"]))
        with TemporaryDirectory() as d:
            root = Path(d).resolve()
            with patch("download_dataset.urlopen", return_value=Reply(archive)):
                unpack_metadata(root, fetch(root,spec,a), spec)
            self.assertEqual((root/"train.jsonl").read_bytes(), sample)
            (root/"media").mkdir()
            (root/"media/images.zip.part").write_bytes(media[:3])

            def resume(req, **kw):
                self.assertEqual(req.get_header("Range"), "bytes=3-")
                return Reply(media[3:], 206, {"Content-Range": f"bytes 3-{len(media)-1}/{len(media)}"})

            with patch("download_dataset.urlopen", side_effect=resume):
                self.assertEqual(fetch(root,spec,b).read_bytes(), media)
            self.assertFalse((root/"media/images.zip.part").exists())
            bad = dict(b, path="../images.zip")
            with self.assertRaises(ValueError):
                validate_index(dict(spec, assets=[a,bad]))
            with self.assertRaises(ValueError):
                fetch(root,spec,bad)


if __name__ == "__main__":
    unittest.main()
