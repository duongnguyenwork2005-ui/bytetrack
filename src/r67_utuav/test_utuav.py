"""Q3 converter and timeline fixtures, including distractors and invalid source IDs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


def xml(objects=""):
    return ("<annotation><filename>v1_026341.jpg</filename><size><width>3840</width>"
            "<height>2160</height></size>" + objects + "</annotation>").encode()


def obj(cls, tid, x1=1, x2=11):
    return (f"<object><name>{cls}</name><bndbox><xmin>{x1}</xmin><ymin>2</ymin>"
            f"<xmax>{x2}</xmax><ymax>12</ymax></bndbox><attributes><attribute>"
            f"<name>track_id</name><value>{tid}</value></attribute></attributes></object>")


class ConverterFixture(unittest.TestCase):
    def test_dynamic_and_distractor(self):
        rows, counts = prepare.convert_xml(xml(obj("motorbike", 0) + obj("car_stat", 1)),
                                           "B", 26341, 1, "v1")
        self.assertEqual(counts, {"motorbike": 1, "car_stat": 1})
        self.assertEqual(rows, ["1,1,1.0000,2.0000,10.0000,10.0000,1,1,1",
                                "1,2,1.0000,2.0000,10.0000,10.0000,1,8,1"])

    def test_duplicate_before_distractor_filter(self):
        with self.assertRaisesRegex(ValueError, "duplicate/invalid"):
            prepare.convert_xml(xml(obj("car", 0) + obj("car_stat", 0)), "B", 26341, 1, "v1")

    def test_two_pixel_source_boundary_is_preserved(self):
        rows, _ = prepare.convert_xml(xml(obj("car", 0, 3799, 3842)), "B", 26341, 1, "v1")
        self.assertEqual(rows[0].split(",")[2:6], ["3799.0000", "2.0000", "43.0000", "10.0000"])

    def test_empty_frame_and_time_step(self):
        self.assertEqual(prepare.convert_xml(xml(), "B", 26341, 1, "v1"), ([], {}))
        self.assertAlmostEqual(1 / prepare.FPS, 1001 / 30000)
        self.assertEqual(round(1.2 * prepare.FPS), 36)


if __name__ == "__main__":
    unittest.main()
