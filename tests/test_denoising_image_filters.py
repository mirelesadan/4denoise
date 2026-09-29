"""Regression tests for count-valued spatial denoising methods."""

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


class GaussianFilterTests(unittest.TestCase):
    def test_integer_counts_are_not_rounded(self):
        image = np.zeros((7, 7), dtype=np.uint16)
        image[3, 3] = 1000

        result = fd.HyperData(image).denoise('gaussian', return_array=True)

        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertGreater(result[3, 3], 0)
        self.assertLess(result[3, 3], 1000)
        self.assertNotEqual(result[3, 3], round(result[3, 3]))
        self.assertEqual(image[3, 3], 1000)

    def test_rectangular_kernel_uses_y_x_order(self):
        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 1

        result = fd.HyperData(image).denoise(
            'gaussian', kernel_size=(3, 5), sigma=1,
        ).array

        self.assertEqual(result[6, 4], 0)
        self.assertGreater(result[4, 6], 0)

    def test_invalid_kernel_and_sigma(self):
        data = fd.HyperData(np.ones((5, 5)))
        for kwargs, fragment in (
            ({'kernel_size': 4}, 'kernel_size'),
            ({'kernel_size': (3, 0)}, 'kernel_size'),
            ({'kernel_size': (3, 2)}, 'kernel_size'),
            ({'sigma': -1}, 'sigma'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('gaussian', **kwargs)


class BilateralFilterTests(unittest.TestCase):
    def test_large_baseline_keeps_small_intensity_differences(self):
        image = np.random.default_rng(3).uniform(0, 10, (9, 9))
        baseline = 1e9
        plain = fd.HyperData(image).denoise(
            'bilateral', d=5, sigma_color=5, sigma_space=2,
        ).array
        shifted = fd.HyperData(image + baseline).denoise(
            'bilateral', d=5, sigma_color=5, sigma_space=2,
        ).array

        self.assertEqual(shifted.dtype, np.dtype('float64'))
        np.testing.assert_allclose(shifted - baseline, plain, atol=1e-5)

    def test_integer_input_returns_count_scale_and_float_output(self):
        image = np.full((7, 7), 1000, dtype=np.uint16)
        image[3, 3] = 1100
        result = fd.HyperData(image).denoise(
            'bilateral', d=5, sigma_color=75, sigma_space=2,
        ).array

        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertGreater(float(result.min()), 900)
        self.assertLess(float(result.max()), 1200)
        self.assertEqual(image[3, 3], 1100)

    def test_invalid_parameters(self):
        data = fd.HyperData(np.ones((5, 5)))
        for kwargs, fragment in (
            ({'d': -1}, 'd'),
            ({'d': 1.5}, 'd'),
            ({'sigma_color': 0}, 'sigma_color'),
            ({'sigma_space': -1}, 'sigma_space'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('bilateral', **kwargs)


class NonLocalMeansTests(unittest.TestCase):
    def test_uint16_input_preserves_count_scale(self):
        image = 1000 + np.random.default_rng(4).integers(0, 30, size=(9, 9), dtype=np.uint16)
        result = fd.HyperData(image).denoise(
            'non_local_means', patch_size=3, patch_distance=2,
        ).array

        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertGreater(float(result.mean()), 900)
        self.assertLess(float(result.mean()), 1100)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_zero_strength_returns_independent_volume_copy(self):
        volume = np.arange(125, dtype=np.uint16).reshape(5, 5, 5)
        result = fd.HyperData(volume).denoise('non_local_means', h=0).array

        np.testing.assert_array_equal(result, volume)
        self.assertFalse(np.shares_memory(result, volume))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_invalid_parameters(self):
        data = fd.HyperData(np.ones((5, 5)))
        for kwargs, fragment in (
            ({'h': -1}, 'h'),
            ({'patch_size': 0}, 'patch_size'),
            ({'patch_distance': -1}, 'patch_distance'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('non_local_means', **kwargs)


class AdaptiveMedianTests(unittest.TestCase):
    def test_largest_window_falls_back_to_median(self):
        image = np.full((7, 7), 10, dtype=np.uint16)
        image[3, 3] = np.iinfo(np.uint16).max
        result = fd.HyperData(image).denoise(
            'adaptive_median_filter', s=3, sMax=5,
        ).array

        self.assertEqual(result[3, 3], 10)
        self.assertEqual(result.dtype, image.dtype)
        self.assertEqual(image[3, 3], np.iinfo(np.uint16).max)

    def test_reflection_avoids_dark_edge_artifacts(self):
        image = np.full((5, 5), 10, dtype=np.uint16)
        result = fd.HyperData(image).denoise(
            'adaptive_median_filter', s=3, sMax=5,
        ).array
        np.testing.assert_array_equal(result, image)

    def test_invalid_window_sizes(self):
        data = fd.HyperData(np.ones((5, 5)))
        for kwargs, fragment in (
            ({'s': 2}, 's'),
            ({'sMax': 4}, 'sMax'),
            ({'s': 7, 'sMax': 5}, 'sMax'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('adaptive_median_filter', **kwargs)


if __name__ == '__main__':
    unittest.main()
