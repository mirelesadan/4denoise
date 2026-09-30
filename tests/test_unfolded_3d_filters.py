"""Volume-filter regressions for 4D-STEM unfold-denoise-refold routing."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


class UnfoldedVolumeFilterTests(unittest.TestCase):
    def setUp(self):
        array = np.random.default_rng(24).integers(
            950, 1050, size=(3, 4, 5, 6), dtype=np.uint16,
        )
        self.original = array.copy()
        self.data = fd.HyperData(
            array, real_units='nm', real_conv_factor=2,
            reciprocal_units='1/nm', reciprocal_conv_factor=0.25,
        )

    def _expected_refold(self, domain, transform):
        unfolded, metadata = self.data.unfold(
            domain=domain, method='serpentine', return_metadata=True,
        )
        transformed = transform(unfolded.array)
        return fd.HyperData(transformed).unfold(
            undo=True, metadata=metadata,
        ).array

    def test_median_defaults_to_all_three_unfolded_axes(self):
        for domain in ('real', 'reciprocal'):
            with self.subTest(domain=domain):
                result = self.data.denoise(
                    'median', unfold_domain=domain,
                    unfold_method='serpentine', window_size=3,
                )
                expected = self._expected_refold(
                    domain, lambda volume: median_filter(
                        volume, size=3, mode='reflect',
                    ),
                )
                np.testing.assert_array_equal(result.array, expected)
                self.assertEqual(result.dtype, self.original.dtype)
                self.assertEqual(result.real_conv_factor, 2)
                self.assertEqual(result.reciprocal_conv_factor, 0.25)
        np.testing.assert_array_equal(self.data.array, self.original)

    def test_median_can_restrict_filtering_to_each_image(self):
        result = self.data.denoise(
            'median', unfold_domain='real', unfold_method='serpentine',
            window_size=3, axes=(1, 2), return_array=True,
        )
        expected = self._expected_refold(
            'real', lambda volume: median_filter(
                volume, size=3, mode='reflect', axes=(1, 2),
            ),
        )
        np.testing.assert_array_equal(result, expected)

    def test_direct_3d_median_filters_the_volume(self):
        volume = self.data.unfold(domain='real', method='serpentine').array
        actual = fd.HyperData(volume).denoise('median', window_size=3).array
        np.testing.assert_array_equal(
            actual, median_filter(volume, size=3, mode='reflect'),
        )

    def test_total_variation_preserves_integer_count_scale(self):
        constant = fd.HyperData(np.full((2, 2, 4, 4), 1000, dtype=np.uint16))
        result = constant.denoise(
            'total_variation', unfold_domain='real', weight=1,
            max_num_iter=3,
        )
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        np.testing.assert_allclose(result.array, 1000)
        self.assertEqual(constant.array.dtype, np.dtype('uint16'))

    def test_total_variation_matches_native_scale_volume_filter(self):
        for domain in ('real', 'reciprocal'):
            with self.subTest(domain=domain):
                result = self.data.denoise(
                    'total_variation', unfold_domain=domain,
                    unfold_method='serpentine', weight=4,
                    max_num_iter=5,
                )
                expected = self._expected_refold(
                    domain, lambda volume: denoise_tv_chambolle(
                        volume.astype(np.float32), weight=4,
                        max_num_iter=5, channel_axis=None,
                    ),
                )
                np.testing.assert_allclose(result.array, expected, rtol=0, atol=1e-4)
                self.assertGreater(float(result.array.mean()), 900)
                self.assertEqual(result.real_units, 'nm')
        np.testing.assert_array_equal(self.data.array, self.original)

    def test_total_variation_zero_weight_is_an_independent_copy(self):
        volume = np.full((3, 4, 5), 1000, dtype=np.uint16)
        result = fd.HyperData(volume).denoise('total_variation', weight=0).array
        np.testing.assert_array_equal(result, volume)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertFalse(np.shares_memory(result, volume))

    def test_total_variation_rejects_invalid_parameters(self):
        for kwargs, fragment in (
            ({'weight': -1}, 'weight'),
            ({'eps': 0}, 'eps'),
            ({'max_num_iter': 0}, 'max_num_iter'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    self.data.denoise(
                        'total_variation', unfold_domain='real', **kwargs,
                    )

    def test_diffusion_uses_stack_axis_and_preserves_total_counts(self):
        array = np.zeros((2, 3, 5, 5), dtype=np.uint16)
        array[0, 1, 2, 2] = 100
        data = fd.HyperData(array)
        result = data.denoise(
            'anisotropic_diffusion', unfold_domain='real',
            unfold_method='serpentine', niter=2, kappa=1000,
            gamma=0.1,
        )
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertAlmostEqual(float(result.array.sum()), 100.0, places=4)
        self.assertGreater(result.array[0, 0, 2, 2], 0)
        self.assertGreater(result.array[0, 2, 2, 2], 0)
        self.assertEqual(data.array[0, 1, 2, 2], 100)


if __name__ == '__main__':
    unittest.main()
