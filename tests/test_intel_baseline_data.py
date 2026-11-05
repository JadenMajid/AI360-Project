import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import pandas as pd

from src.intel_baseline.data import CATEGORIES, EXPECTED, assert_archive_ready, audit_data, extract_audio, safe_relative, validate_manifest


class ManifestTests(unittest.TestCase):
    def manifest(self):
        return pd.DataFrame([dict(CATEGORY="Good", SPLIT="TRAIN", SUBDIRS="session/good"),
                             dict(CATEGORY="Porosity", SPLIT="VAL", SUBDIRS="session/defect")])

    def test_training_defects_and_duplicates_rejected(self):
        frame = self.manifest()
        validate_manifest(frame, official=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(pd.concat([frame, frame]), official=False)
        frame.loc[0, "CATEGORY"] = "Porosity"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_manifest(frame, official=False)

    def test_official_counts_required(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_manifest(self.manifest())
        self.assertEqual(sum(sum(x.values()) for x in EXPECTED.values()), 4040)

    def test_unsafe_paths_rejected(self):
        for path in ("../audio.flac", "/etc/passwd", "session/../../audio.flac", "..\\audio.flac"):
            with self.assertRaises(ValueError):
                safe_relative(path)

    def test_missing_audio_rejected(self):
        rows = []
        defects = sorted(CATEGORIES - {"Good"})
        for split, counts in EXPECTED.items():
            for label, count in counts.items():
                for i in range(count):
                    rows.append(dict(CATEGORY=defects[i % len(defects)] if label else "Good",
                                     SPLIT=split, SUBDIRS=f"session/{split}-{label}-{i}"))
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(rows).to_csv(Path(directory) / "manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Missing"):
                audit_data(directory)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("raid/intel_robotic_welding_dataset/../escape.flac")
                member.size = 3
                stream.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                extract_audio(archive, Path(directory) / "data")
            self.assertFalse((Path(directory) / "escape.flac").exists())

    def test_incomplete_download_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets--test"
            blobs = root / "blobs"
            blobs.mkdir(parents=True)
            archive = root / "data.tar.gz"
            archive.write_bytes(b"fixture")
            partial = blobs / "blob.incomplete"
            partial.write_bytes(b"unfinished")
            with self.assertRaisesRegex(ValueError, "incomplete downloads"):
                assert_archive_ready(archive)

    def test_active_download_lock_rejected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets--test"
            root.mkdir()
            archive = root / "data.tar.gz"
            archive.write_bytes(b"fixture")
            locks = Path(directory) / ".locks" / root.name
            locks.mkdir(parents=True)
            with (locks / "download.lock").open("w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                with self.assertRaisesRegex(ValueError, "download is active"):
                    assert_archive_ready(archive)


if __name__ == "__main__":
    unittest.main()
