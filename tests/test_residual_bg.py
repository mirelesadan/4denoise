"""Background aggregation contracts for stacks and scan grids."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd


class ResidualBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.stack = fd.HyperData(np.ones((3, 9, 9), dtype=float))
        self.scan = fd.HyperData(np.ones((2, 2, 9, 9), dtype=float))
        self.peaks = np.array([[4.0, 4.0], [3.0, 6.0]])

    def test_3d_stack_returns_one_value_per_peak_or_pattern(self):
        centers = np.broadcast_to(self.peaks, (3, 2, 2))
        per_peak = self.stack.get_residualBg(
            centers=centers, r_spots=(1, 2), bg_method='rings',
        )
        pooled = self.stack.get_residualBg(
            centers=centers, r_spots=(1, 2), bg_method='rings_mean',
        )
        self.assertEqual(per_peak.shape, (3, 2))
        self.assertEqual(pooled.shape, (3,))
        np.testing.assert_allclose(per_peak, 1)
        np.testing.assert_allclose(pooled, 1)

    def test_4d_scan_and_shared_centers(self):
        per_peak = self.scan.get_residualBg(
            centers=self.peaks, r_spots=(1, 2), bg_method='rings',
        )
        pooled = self.scan.get_residualBg(
            centers=self.peaks, r_spots=(1, 2), bg_method='rings_mean',
        )
        self.assertEqual(per_peak.shape, (2, 2, 2))
        self.assertEqual(pooled.shape, (2, 2))
        np.testing.assert_allclose(per_peak, 1)

    def test_ragged_peak_counts_remain_ragged(self):
        centers = [self.peaks[:1], self.peaks, self.peaks[:1]]
        values = self.stack.get_residualBg(
            centers=centers, r_spots=(1, 2), bg_method='rings',
        )
        self.assertIsInstance(values, list)
        self.assertEqual([len(entry) for entry in values], [1, 2, 1])

    def test_remove_bg_requires_one_value_per_pattern(self):
        with self.assertRaisesRegex(ValueError, "bg_method='rings_mean'"):
            self.stack.remove_bg(
                np.zeros((9, 9)), residual_bg_frac=1,
                centers=self.peaks, r_spots=(1, 2), bg_method='rings',
            )

    def test_missing_centers_and_reference_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'centers or ref_coords'):
            self.stack.get_residualBg(r_spots=(1, 2))


if __name__ == '__main__':
    unittest.main()
