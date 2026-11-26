"""Check corrected R66 raw field aggregation against the installed evaluator."""
import unittest

import numpy as np
from trackeval.metrics import CLEAR, HOTA, Identity

from sher_metric import combine, roi_contains


def fixture(similarity):
    # A present object and one tracker for each float; an empty frame for None.
    gt = [np.array([0], dtype=int) if v is not None else np.array([], dtype=int) for v in similarity]
    data = dict(num_timesteps=len(gt), gt_ids=gt, tracker_ids=gt,
                num_gt_dets=sum(map(len, gt)), num_tracker_dets=sum(map(len, gt)),
                num_gt_ids=int(any(map(len, gt))), num_tracker_ids=int(any(map(len, gt))),
                similarity_scores=[np.array([[v]]) if v is not None else np.empty((0, 0)) for v in similarity])
    return {m.get_name(): m.eval_sequence(data) for m in [HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})]}


class TestSherMetric(unittest.TestCase):
    def check_combination(self, cases):
        seq = {key: fixture(values) for key, values in cases.items()}
        raw = {key: {**{f: m["HOTA"][f] for f in ["HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA"]},
                     **{f: m["Identity"][f] for f in ["IDTP", "IDFN", "IDFP"]},
                     **{f: m["CLEAR"][f] for f in ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW"]}}
               for key, m in seq.items()}
        expected = {metric.get_name(): metric.combine_sequences({k: m[metric.get_name()] for k, m in seq.items()})
                    for metric in [HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})]}
        result = combine(raw)
        for key in ["HOTA", "AssA", "DetA"]:
            self.assertAlmostEqual(result[key], expected["HOTA"][key].mean())
        self.assertAlmostEqual(result["IDF1"], expected["Identity"]["IDF1"])
        for name, key in [("MOTA", "MOTA"), ("FP", "CLR_FP"), ("FN", "CLR_FN"), ("IDSW", "IDSW")]:
            self.assertAlmostEqual(result[name], expected["CLEAR"][key])

    def test_zero_tp(self):
        self.check_combination({"x": [0.]})

    def test_no_objects(self):
        self.check_combination({"empty": [None, None]})

    def test_cross_sequence_weighting_and_empty_frame(self):
        self.check_combination({"many": [1., None, 0.7, 0.6], "one": [0.41], "empty": [None]})

    def test_off_image_centres_and_mask_hole(self):
        mask = np.ones((600, 800), dtype=np.uint8)
        mask[400, 3] = 0
        x = np.array([-10, -5, -2, 794.9, 795, 40, 40])
        y = np.array([390, 390, 390, 390, 390, -11, 590])
        self.assertEqual(roi_contains(mask, x, y, 10, 20).tolist(),
                         [False, True, False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
