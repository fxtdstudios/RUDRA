"""Pure-stdlib tests for deterministic real-video manifest construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.build_video_manifest import build_manifest


class VideoManifestTests(unittest.TestCase):
    def test_builds_consecutive_paired_clips_and_ignores_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdr = root / "hdr"
            sdr = root / "sdr"
            hdr.mkdir()
            sdr.mkdir()
            for frame in range(10, 16):
                name = f"tif_{frame:07d}_Scene_A_{frame:05d}.png"
                (hdr / name).touch()
                (sdr / name).touch()
            # Same source frame, different ingestion counter: ignored.
            (hdr / "tif_9999999_Scene_A_00012.png").touch()
            (sdr / "tif_9999999_Scene_A_00012.png").touch()
            # A still image must never be turned into a fake video.
            (hdr / "exr_0001_studio_hdri.png").touch()
            (sdr / "exr_0001_studio_hdri.png").touch()

            output = root / "clips.jsonl"
            stats = build_manifest(hdr, output, sdr, clip_length=3, clip_stride=3)
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(stats["clips"], 2)
            self.assertEqual(stats["hdr_duplicates_ignored"], 1)
            self.assertEqual(stats["hdr_stills_excluded"], 1)
            self.assertEqual(records[0]["frame_numbers"], [10, 11, 12])
            self.assertEqual(records[1]["frame_numbers"], [13, 14, 15])
            self.assertEqual(len(records[0]["hdr_frames"]), 3)
            self.assertEqual(len(records[0]["sdr_frames"]), 3)


if __name__ == "__main__":
    unittest.main()
