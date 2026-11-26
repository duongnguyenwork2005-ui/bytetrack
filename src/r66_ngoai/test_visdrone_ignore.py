"""Boundary fixtures for the official VisDrone dropObjects.m geometry."""
import unittest

import numpy as np

from visdrone_ignore import ignored_fraction


class IgnoreRegionTest(unittest.TestCase):
    def test_half_area_is_dropped(self):
        box = np.array([[1, 1, 2, 2]], dtype=float)
        region = np.array([[1, 1, 1, 2]], dtype=float)
        self.assertEqual(ignored_fraction(box, region, 5, 5).tolist(), [0.5])

    def test_overlapping_regions_form_union(self):
        box = np.array([[1, 1, 2, 2]], dtype=float)
        regions = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float)
        self.assertEqual(ignored_fraction(box, regions, 5, 5).tolist(), [1.0])

    def test_rounding_and_image_edge(self):
        box = np.array([[3.5, 3.5, 2, 2]], dtype=float)
        region = np.array([[3.5, 3.5, 2, 2]], dtype=float)
        self.assertEqual(ignored_fraction(box, region, 5, 5).tolist(), [0.25])

    def test_empty_regions(self):
        boxes = np.array([[1, 1, 2, 2], [3, 3, 2, 2]], dtype=float)
        self.assertEqual(ignored_fraction(boxes, np.empty((0, 4)), 5, 5).tolist(), [0.0, 0.0])

    def test_integral_query_matches_direct_pixel_count(self):
        rng = np.random.default_rng(67)
        for _ in range(100):
            height, width = 9, 11
            regions = rng.uniform(-2, 10, size=(3, 4))
            regions[:, 2:] = rng.uniform(0.5, 5, size=(3, 2))
            boxes = rng.uniform(-2, 10, size=(4, 4))
            boxes[:, 2:] = rng.uniform(0.5, 5, size=(4, 2))
            actual = ignored_fraction(boxes, regions, height, width)

            # Direct implementation of the inclusive MATLAB region map and
            # the exclusive upper-left corner in the integral-image query.
            pixels = np.zeros((height + 1, width + 1), dtype=bool)
            for rx, ry, rw, rh in regions:
                x, y, w, h = [max(1, int(np.floor(v + 0.5))) for v in (rx, ry, rw, rh)]
                for yy in range(y, min(height, y + h) + 1):
                    for xx in range(x, min(width, x + w) + 1):
                        pixels[yy, xx] = True
            expected = []
            for bx, by, bw, bh in boxes:
                x, y, w, h = [max(1, int(np.floor(v + 0.5))) for v in (bx, by, bw, bh)]
                x, y = min(width, x), min(height, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                expected.append(pixels[y + 1:y2 + 1, x + 1:x2 + 1].sum() / (w * h))
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
