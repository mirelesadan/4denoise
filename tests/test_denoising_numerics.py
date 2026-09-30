"""Numerical regression tests for selected denoising methods."""

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


class AnisotropicDiffusionTests(unittest.TestCase):
    def test_zero_and_signed_constants_are_unchanged(self):
        for shape, value in (((5, 6), 0.0), ((4, 5, 3), -2.5)):
            with self.subTest(shape=shape):
                original = np.full(shape, value, dtype=np.float32)
                denoised = fd.HyperData(original).denoise(
                    'anisotropic_diffusion', niter=3,
                )
                np.testing.assert_array_equal(denoised.array, original)
                np.testing.assert_array_equal(original, np.full(shape, value))

    def test_impulse_diffuses_and_preserves_total_intensity(self):
        original = np.zeros((7, 7), dtype=np.float64)
        original[3, 3] = 10.0
        for option in (1, 2):
            with self.subTest(option=option):
                denoised = fd.HyperData(original).denoise(
                    'anisotropic_diffusion', niter=2, kappa=100,
                    gamma=0.2, option=option, return_array=True,
                )
                self.assertLess(denoised[3, 3], 10.0)
                self.assertGreater(denoised[3, 4], 0.0)
                self.assertGreaterEqual(float(denoised.min()), 0.0)
                self.assertAlmostEqual(float(denoised.sum()), 10.0, places=12)
        self.assertEqual(original[3, 3], 10.0)

    def test_integer_volume_uses_stable_float_substeps(self):
        original = np.zeros((5, 5, 5), dtype=np.uint16)
        original[2, 2, 2] = 100
        denoised = fd.HyperData(original).denoise(
            'anisotropic_diffusion', niter=3, kappa=100, gamma=0.2,
        )
        self.assertTrue(np.issubdtype(denoised.dtype, np.floating))
        self.assertGreater(denoised.array[2, 2, 3], 0)
        self.assertGreaterEqual(float(denoised.array.min()), 0)
        self.assertAlmostEqual(float(denoised.array.sum()), 100.0, places=4)
        self.assertEqual(original[2, 2, 2], 100)

    def test_zero_iterations_returns_independent_copy(self):
        original = np.array([[0, 2], [3, 4]], dtype=np.uint16)
        denoised = fd.HyperData(original).denoise(
            'anisotropic_diffusion', niter=0,
        )
        self.assertEqual(denoised.dtype, original.dtype)
        np.testing.assert_array_equal(denoised.array, original)
        self.assertFalse(np.shares_memory(denoised.array, original))

    def test_invalid_parameters_are_rejected(self):
        data = fd.HyperData(np.ones((4, 4)))
        for kwargs, fragment in (
            ({'niter': -1}, 'niter'),
            ({'niter': 1.5}, 'niter'),
            ({'kappa': 0}, 'kappa'),
            ({'gamma': -0.1}, 'gamma'),
            ({'option': 3}, 'option'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('anisotropic_diffusion', **kwargs)


class FourierFilterTests(unittest.TestCase):
    def test_pass_and_cut_reconstruct_signed_input(self):
        image = np.random.default_rng(5).normal(size=(9, 10))
        data = fd.HyperData(image)
        for kwargs in (
            {'r_inner': 2.0, 'sigma': 0},
            {'r_inner': 1.5, 'r_outer': 3.5, 'sigma': 0},
            {'r_inner': 1.5, 'r_outer': 3.5, 'sigma': 0.7},
            {'r_inner': 0, 'sigma': 2.5},
        ):
            with self.subTest(kwargs=kwargs):
                passed = data.denoise(
                    'fourier_filter', mode='pass', return_array=True, **kwargs,
                )
                cut = data.denoise(
                    'fourier_filter', mode='cut', return_array=True, **kwargs,
                )
                np.testing.assert_allclose(passed + cut, image, atol=1e-12)

    def test_band_cut_removes_only_selected_sinusoid(self):
        x = np.arange(16)
        image = np.broadcast_to(
            np.cos(2 * np.pi * 3 * x / 16), (16, 16),
        ).copy()
        data = fd.HyperData(image)
        kwargs = {'r_inner': 2.5, 'r_outer': 3.5, 'sigma': 0}
        passed = data.denoise('fourier_filter', mode='pass', **kwargs).array
        cut = data.denoise('fourier_filter', mode='cut', **kwargs).array
        np.testing.assert_allclose(passed, image, atol=1e-12)
        np.testing.assert_allclose(cut, 0, atol=1e-12)
        self.assertLess(float(passed.min()), 0)

    def test_default_lowpass_preserves_a_constant(self):
        image = np.full((8, 10), -3.0, dtype=np.float32)
        denoised = fd.HyperData(image).denoise('fourier_filter')
        np.testing.assert_allclose(denoised.array, image, atol=1e-6)
        self.assertEqual(denoised.dtype, np.dtype('float32'))

    def test_complex_input_remains_complex(self):
        image = np.ones((5, 6), dtype=np.complex64) * (1 + 2j)
        denoised = fd.HyperData(image).denoise('fourier_filter')
        self.assertTrue(np.iscomplexobj(denoised.array))
        np.testing.assert_allclose(denoised.array, image, atol=1e-6)

    def test_invalid_parameters_are_rejected(self):
        data = fd.HyperData(np.ones((4, 4)))
        for kwargs, fragment in (
            ({'mode': 'unknown'}, 'mode'),
            ({'r_inner': -1}, 'r_inner'),
            ({'r_inner': 3, 'r_outer': 2}, 'r_outer'),
            ({'sigma': -1}, 'sigma'),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, fragment):
                    data.denoise('fourier_filter', **kwargs)


class Parafac2ValidationTests(unittest.TestCase):
    def test_unknown_implementation_has_clear_error(self):
        data = fd.HyperData(np.ones((2, 3, 3)))
        with self.assertRaisesRegex(ValueError, 'Unknown parafac2 implementation'):
            data.denoise('parafac2', rank=1, implementation='unknown')

    def test_valid_tensorly_path_still_returns_reconstruction(self):
        original = np.random.default_rng(7).random((3, 4, 5)).astype(np.float32)
        denoised = fd.HyperData(original).denoise(
            'parafac2', rank=1, n_iter_max=2, n_iter_parafac=1,
            linesearch=False, random_state=0,
        )
        self.assertIsInstance(denoised, fd.HyperData)
        self.assertEqual(denoised.shape, original.shape)
        self.assertTrue(np.all(np.isfinite(denoised.array)))


if __name__ == '__main__':
    unittest.main()
