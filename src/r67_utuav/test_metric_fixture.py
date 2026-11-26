"""TrackEval oracle, distractor and empty-output fixture for the UTUAV MOT adapter."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config
import run_trackeval as rte


class MetricFixture(unittest.TestCase):
    def test_oracle_distractor_empty(self):
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="utuav_q3_metric_", dir=tmp_root) as tmp:
            root = Path(tmp)
            processed = root / "processed"
            split, video = "UTUAVQ3-fixture", "seq"
            seq = processed / split / video
            (seq / "gt").mkdir(parents=True)
            (processed / "seqmaps").mkdir()
            (processed / "seqmaps" / f"{split}.txt").write_text("name\nseq\n")
            (seq / "seqinfo.ini").write_text("[Sequence]\nname=seq\nseqLength=2\n"
                "imWidth=100\nimHeight=100\nframeRate=30\nimDir=img\nimExt=.jpg\n")
            (seq / "gt/gt.txt").write_text(
                "1,1,10,10,10,10,1,1,1\n2,1,11,10,10,10,1,1,1\n"
                "1,2,50,50,10,10,1,8,1\n2,2,50,50,10,10,1,8,1\n")
            base = processed / "trackers" / split
            for name, content in {
                "oracle": "1,7,10,10,10,10,0.9\n2,7,11,10,10,10,0.9\n"
                          "1,9,50,50,10,10,0.9\n2,9,50,50,10,10,0.9\n",
                "empty": "",
            }.items():
                folder = base / name / "data"
                folder.mkdir(parents=True)
                (folder / "seq.txt").write_text(content)
            with patch.object(config, "PROCESSED_DIR", processed), \
                 patch.object(config, "RESULTS_DIR", root / "results"), \
                 contextlib.redirect_stdout(io.StringIO()):
                oracle, _ = rte.run(split, "oracle", [video], "mot")
                empty, _ = rte.run(split, "empty", [video], "mot")
            def combined(result, tracker):
                return rte.flatten_results(result, tracker).set_index("video").loc["COMBINED_SEQ"]
            good, blank = combined(oracle, "oracle"), combined(empty, "empty")
            self.assertAlmostEqual(float(good["HOTA"]), 1.0)
            self.assertAlmostEqual(float(good["IDF1"]), 1.0)
            self.assertEqual(int(good["FP"]), 0)
            self.assertAlmostEqual(float(blank["HOTA"]), 0.0)
            self.assertEqual(int(blank["FN"]), 2)


if __name__ == "__main__":
    unittest.main()
