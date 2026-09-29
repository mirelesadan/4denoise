"""Routing, dtype, and metadata checks for HyperData.denoise."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


def _fractional(_self, array):
    return np.asarray(array, dtype=np.float64) + 0.5


class DenoisingRoutingTests(unittest.TestCase):
    def test_integer_input_keeps_fractional_slice_results(self):
        original = np.arange(2 * 3 * 2 * 2, dtype=np.uint16).reshape(2, 3, 2, 2)
        data = fd.HyperData(
            original, real_units='nm', real_conv_factor=2,
            reciprocal_units='1/nm', reciprocal_conv_factor=0.25,
        )

        with patch.object(fd._DenoisingMethods, 'gaussian', new=_fractional):
            denoised = data.denoise('gaussian', domain='reciprocal')

        self.assertIsInstance(denoised, fd.HyperData)
        self.assertEqual(denoised.dtype, np.dtype('float64'))
        np.testing.assert_array_equal(denoised.array, original + 0.5)
        np.testing.assert_array_equal(data.array, original)
        self.assertEqual(denoised.real_units, 'nm')
        self.assertEqual(denoised.reciprocal_conv_factor, 0.25)

    def test_real_non_local_means_keeps_fractional_output(self):
        original = np.random.default_rng(17).integers(
            0, 100, size=(2, 2, 8, 8), dtype=np.uint16,
        )
        denoised = fd.HyperData(original).denoise(
            'non_local_means', domain='reciprocal',
        )
        self.assertEqual(denoised.dtype, np.dtype('float64'))
        self.assertEqual(denoised.shape, original.shape)
        self.assertTrue(np.any(denoised.array != denoised.array.astype(np.uint16)))

    def test_inconsistent_slice_dtypes_fail_instead_of_casting(self):
        original = np.zeros((2, 1, 2, 2), dtype=np.uint16)
        calls = iter((np.float32, np.float64))

        def varying_dtype(_self, image):
            return np.full(image.shape, 0.5, dtype=next(calls))

        with patch.object(fd._DenoisingMethods, 'gaussian', new=varying_dtype):
            with self.assertRaisesRegex(TypeError, 'inconsistent slice dtypes'):
                fd.HyperData(original).denoise('gaussian')

    def test_real_and_reciprocal_slices_keep_their_axis_order(self):
        original = np.arange(2 * 3 * 2 * 2, dtype=np.uint16).reshape(2, 3, 2, 2)

        def first_pixel(_self, image):
            return np.full(image.shape, float(image[0, 0]) + 0.25)

        with patch.object(fd._DenoisingMethods, 'gaussian', new=first_pixel):
            real = fd.HyperData(original).denoise('gaussian', domain='real')
            reciprocal = fd.HyperData(original).denoise('gaussian', domain='reciprocal')

        for ky in range(original.shape[2]):
            for kx in range(original.shape[3]):
                np.testing.assert_array_equal(
                    real.array[:, :, ky, kx], original[0, 0, ky, kx] + 0.25,
                )
        for ry in range(original.shape[0]):
            for rx in range(original.shape[1]):
                np.testing.assert_array_equal(
                    reciprocal.array[ry, rx], original[ry, rx, 0, 0] + 0.25,
                )

    def test_unsupported_input_dimensions_fail_before_numerical_method(self):
        three_d = fd.HyperData(np.ones((2, 3, 4)))
        four_d = fd.HyperData(np.ones((2, 3, 4, 5)))

        with self.assertRaisesRegex(ValueError, "'gaussian' expects"):
            three_d.denoise('gaussian')
        for method in ('bm4d', 'parafac2'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, '3D'):
                    four_d.denoise(method)
        volume_denoised = three_d.denoise('median', window_size=1)
        np.testing.assert_array_equal(volume_denoised.array, three_d.array)
        denoised = three_d.denoise('median', window_size=1, axes=(1, 2))
        np.testing.assert_array_equal(denoised.array, three_d.array)

    def test_invalid_arguments_and_internal_type_errors_are_distinct(self):
        data = fd.HyperData(np.ones((3, 3)))
        with self.assertRaisesRegex(TypeError, 'Invalid arguments'):
            data.denoise('gaussian', nonexistent=1)

        def broken(_self, array):
            raise TypeError('numerical failure')

        with patch.object(fd._DenoisingMethods, 'gaussian', new=broken):
            with self.assertRaisesRegex(TypeError, '^numerical failure$'):
                data.denoise('gaussian')

    def test_wrong_shape_and_non_array_results_do_not_inherit_metadata(self):
        data = fd.HyperData(np.ones((3, 3)))

        def wrong_shape(_self, array):
            return array[:2]

        with patch.object(fd._DenoisingMethods, 'gaussian', new=wrong_shape):
            with self.assertRaisesRegex(ValueError, 'changed shape'):
                data.denoise('gaussian', return_array=True)

        def missing(_self, array):
            return None

        with patch.object(fd._DenoisingMethods, 'gaussian', new=missing):
            with self.assertRaisesRegex(TypeError, 'expected an ndarray'):
                data.denoise('gaussian')

    def test_optional_decomposition_retains_direct_return_type(self):
        def factorize(_self, tensor, rank, return_decomposition=False):
            reconstruction = np.asarray(tensor, dtype=np.float64)
            return [rank, reconstruction] if return_decomposition else reconstruction

        with patch.object(fd._DenoisingMethods, 'parafac', new=factorize):
            direct = fd.HyperData(np.ones((2, 3, 4))).denoise(
                'parafac', rank=2, return_decomposition=True,
            )
            self.assertIsInstance(direct, list)
            self.assertEqual(direct[0], 2)

            four_d = fd.HyperData(np.ones((2, 3, 4, 5)))
            with self.assertRaisesRegex(ValueError, 'Slice-wise 4D'):
                four_d.denoise('parafac', rank=2, return_decomposition=True)
            with self.assertRaisesRegex(ValueError, 'unfold-denoise-refold'):
                four_d.denoise(
                    'parafac', rank=2, unfold_domain='real',
                    return_decomposition=True,
                )
            whole = four_d.denoise(
                'parafac', rank=2, domain=None, return_decomposition=True,
            )
            self.assertIsInstance(whole, list)
            self.assertEqual(whole[1].shape, four_d.shape)

    def test_domain_none_keeps_whole_tensor_and_metadata(self):
        original = np.arange(2 * 3 * 2 * 2, dtype=np.uint16).reshape(2, 3, 2, 2)
        data = fd.HyperData(
            original, real_units='nm', real_conv_factor=3,
        )
        def whole_tensor(_self, tensor, rank):
            return _fractional(_self, tensor)

        with patch.object(fd._DenoisingMethods, 'parafac', new=whole_tensor):
            denoised = data.denoise('parafac', rank=1, domain=None)
        np.testing.assert_array_equal(denoised.array, original + 0.5)
        self.assertEqual(denoised.real_units, 'nm')
        self.assertEqual(denoised.real_conv_factor, 3)

    def test_unfold_refold_preserves_dtype_calibration_and_attached_metadata(self):
        original = np.arange(2 * 3 * 2 * 2, dtype=np.uint16).reshape(2, 3, 2, 2)
        data = fd.HyperData(
            original, real_units='nm', real_conv_factor=2,
            reciprocal_units='1/nm', reciprocal_conv_factor=0.25,
        )
        with patch.object(fd._DenoisingMethods, 'bm4d', new=_fractional):
            denoised = data.denoise(
                'bm4d', unfold_domain='real', unfold_method='serpentine',
            )
        np.testing.assert_array_equal(denoised.array, original + 0.5)
        self.assertEqual(denoised.real_conv_factor, 2)
        self.assertEqual(denoised.reciprocal_units, '1/nm')

        unfolded = data.unfold(domain='real', method='serpentine')
        filtered = unfolded.denoise('median', window_size=1, axes=(1, 2))
        self.assertIsNotNone(filtered.unfold_metadata)
        np.testing.assert_array_equal(filtered.unfold(undo=True).array, original)

    def test_nonpreserving_crop_updates_real_origin(self):
        original = np.arange(6 * 6 * 2 * 2, dtype=np.uint16).reshape(6, 6, 2, 2)
        data = fd.HyperData(
            original, real_units='nm', real_conv_factor=(2, 3),
            real_origin=(10, 20),
        )
        with patch.object(fd._DenoisingMethods, 'bm4d', new=_fractional):
            denoised = data.denoise(
                'bm4d', unfold_domain='real', unfold_method='hilbert',
                unfold_kwargs={'preserve_excess': False},
            )
        self.assertEqual(denoised.shape, (4, 4, 2, 2))
        self.assertEqual(denoised.real_origin, (12, 23))
        self.assertEqual(denoised.real_conv_factor, (2, 3))
        np.testing.assert_array_equal(
            denoised.array, original[1:5, 1:5] + 0.5,
        )

    def test_nonpreserving_resize_uses_resized_calibration(self):
        original = np.arange(2 * 2 * 5 * 5, dtype=np.uint16).reshape(2, 2, 5, 5)
        data = fd.HyperData(
            original, reciprocal_units='1/nm', reciprocal_conv_factor=0.2,
        )
        with patch.object(fd._DenoisingMethods, 'bm4d', new=_fractional):
            denoised = data.denoise(
                'bm4d', unfold_domain='reciprocal', unfold_method='hilbert',
                unfold_kwargs={
                    'curve_shape_strategy': 'resize',
                    'preserve_original': False,
                },
            )
        self.assertEqual(denoised.shape, (2, 2, 4, 4))
        self.assertEqual(denoised.reciprocal_units, '1/nm')
        self.assertAlmostEqual(denoised.reciprocal_conv_factor, 0.25)
        resized = data.resize((4, 4), domain='reciprocal', method='linear')
        np.testing.assert_allclose(denoised.array, resized.array + 0.5)


if __name__ == '__main__':
    unittest.main()
