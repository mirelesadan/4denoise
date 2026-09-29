"""Regression tests for diffraction-peak template matching."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


def disk_pattern(shape, peaks, background=0.0, radius=2):
    pattern = np.full(shape, background, dtype=float)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for y, x, amplitude in peaks:
        pattern[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] += amplitude
    return pattern


class PeakDetectionTests(unittest.TestCase):
    def test_template_is_zero_sum_and_flat_background_gives_no_peaks(self):
        kernel = fd.ReciprocalSpace._peak_template(6, 1, 1, -0.5)
        self.assertAlmostEqual(float(np.sum(kernel)), 0, places=12)
        self.assertGreater(np.max(kernel), 0)
        self.assertLess(np.min(kernel), 0)

        peaks = fd.ReciprocalSpace(np.full((41, 41), 100.0)).get_peaks(
            radius=6, min_distance=3, threshold_abs=1,
        )
        self.assertEqual(peaks.shape, (0, 2))

    def test_offset_background_does_not_change_detected_positions(self):
        pattern = disk_pattern((41, 41), [(20, 20, 50)], background=5)
        kwargs = dict(radius=2, min_distance=3, threshold_abs=1)
        first = fd.ReciprocalSpace(pattern).get_peaks(**kwargs)
        second = fd.ReciprocalSpace(pattern + 1000).get_peaks(**kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.any(np.all(first == (20, 20), axis=1)))

    def test_absolute_and_relative_thresholds_both_apply(self):
        pattern = disk_pattern(
            (51, 51), [(15, 15, 40), (35, 35, 10)],
        )
        dp = fd.ReciprocalSpace(pattern)
        both = dp.get_peaks(
            radius=2, min_distance=3,
            threshold_abs=1, threshold_rel=0.8,
        )
        absolute_only = dp.get_peaks(
            radius=2, min_distance=3, threshold_abs=1,
        )
        self.assertTrue(np.any(np.all(both == (15, 15), axis=1)))
        self.assertFalse(np.any(np.all(both == (35, 35), axis=1)))
        self.assertTrue(np.any(np.all(absolute_only == (35, 35), axis=1)))
        strongest = dp.get_peaks(
            radius=2, min_distance=3,
            threshold_abs=None, threshold_rel=1,
        )
        self.assertTrue(np.any(np.all(strongest == (15, 15), axis=1)))
        self.assertFalse(np.any(np.all(strongest == (35, 35), axis=1)))

    def test_nonfinite_pixel_does_not_poison_entire_fft(self):
        pattern = disk_pattern((51, 51), [(35, 35, 50)])
        pattern[4, 4] = np.nan
        peaks = fd.ReciprocalSpace(pattern).get_peaks(
            radius=2, min_distance=3, threshold_abs=1,
        )
        self.assertTrue(np.any(np.all(peaks == (35, 35), axis=1)))
        self.assertFalse(np.any(np.all(peaks == (4, 4), axis=1)))

        diagonal_nan = disk_pattern((41, 41), [(15, 15, 50)])
        diagonal_nan[12, 12] = np.nan
        diagonal_peaks = fd.ReciprocalSpace(diagonal_nan).get_peaks(
            radius=2, min_distance=3, threshold_abs=1,
        )
        self.assertTrue(np.any(np.all(diagonal_peaks == (15, 15), axis=1)))

        with self.assertRaisesRegex(ValueError, 'no finite values'):
            fd.ReciprocalSpace(np.full((11, 11), np.nan)).get_peaks(
                radius=2, min_distance=2,
            )

    def test_annular_search_uses_explicit_or_metadata_center(self):
        pattern = disk_pattern((41, 41), [(10, 15, 50)])
        beam = fd._center_beam_metadata_from_pixels(
            1, (10, 10), pattern.shape, source='alignment',
        )
        dp = fd.ReciprocalSpace(pattern, center_beam_metadata=beam)
        kwargs = dict(
            radius=2, min_distance=3, threshold_abs=1, r_range=(4, 6),
        )

        from_metadata = dp.get_peaks(**kwargs)
        from_explicit = fd.ReciprocalSpace(pattern).get_peaks(
            **kwargs, center=(10, 10),
        )
        np.testing.assert_array_equal(from_metadata, from_explicit)
        self.assertTrue(np.any(np.all(from_metadata == (10, 15), axis=1)))
        self.assertEqual(dp.get_peaks(**kwargs, center=(20, 20)).shape, (0, 2))

        stack = fd.HyperData(
            np.stack([pattern, pattern]), center_beam_metadata=beam,
        )
        stack_peaks = stack.get_peaks(**kwargs)
        self.assertEqual(len(stack_peaks), 2)
        np.testing.assert_array_equal(stack_peaks[0], from_metadata)
        np.testing.assert_array_equal(stack_peaks[1], from_metadata)

        scan = fd.HyperData(
            np.stack([pattern, pattern])[None, :, :, :],
            center_beam_metadata=beam,
        )
        scan_peaks = scan.get_peaks(**kwargs, real_mask=[[True, False]])
        np.testing.assert_array_equal(scan_peaks[0][0], from_metadata)
        self.assertEqual(scan_peaks[0][1].shape, (0, 2))

    def test_invalid_parameters_fail_before_peak_detection(self):
        dp = fd.ReciprocalSpace(np.zeros((21, 21)))
        for kwargs, message in (
            ({'radius': 0}, 'radius must be positive'),
            ({'min_distance': 0}, 'min_distance must be a positive integer'),
            ({'trench_width': 0.01}, 'trench_width is too small'),
            ({'sym_mode': 'wrong', 'n_fold': 6}, 'sym_mode must be'),
            ({'sym_mode': 'repair'}, 'sym_mode requires n_fold'),
            ({'threshold_rel': 1.1}, 'threshold_rel must be between'),
            ({'r_range': (100, 101)}, 'r_range selects no'),
            ({'center': (np.nan, 10)}, 'center must contain'),
        ):
            options = dict(radius=2, min_distance=2)
            options.update(kwargs)
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                dp.get_peaks(**options)

        beam = fd._center_beam_metadata_from_pixels(
            1, (10, 10), (21, 21), source='alignment',
        )
        beam['shape'] = (20, 20)
        with self.assertRaisesRegex(ValueError, 'stale pattern shape'):
            fd.ReciprocalSpace(
                np.zeros((21, 21)), center_beam_metadata=beam,
            ).get_peaks(radius=2, min_distance=2)

    def test_polar_pattern_is_rejected(self):
        dp = fd.ReciprocalSpace(
            np.zeros((21, 21)), polar_metadata={'is_polar': True},
        )
        with self.assertRaisesRegex(ValueError, 'Cartesian'):
            dp.get_peaks(radius=2, min_distance=2)


if __name__ == '__main__':
    unittest.main()
