"""TensorLy reconstruction, selected modes, and convergence errors."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


class TensorLyErrorResultTests(unittest.TestCase):
    def setUp(self):
        self.tensor = np.random.default_rng(9).random((3, 4, 5)) + 0.1

    def assert_reconstruction_and_errors(self, method, kwargs):
        reconstruction, errors = fd.HyperData(self.tensor).denoise(
            method, return_errors=True, **kwargs,
        )
        self.assertIsInstance(reconstruction, fd.HyperData)
        self.assertEqual(reconstruction.shape, self.tensor.shape)
        self.assertTrue(np.all(np.isfinite(reconstruction.array)))
        self.assertGreaterEqual(len(errors), 1)
        self.assertTrue(np.all(np.isfinite(errors)))

    def test_cp_methods(self):
        methods = (
            ('parafac', {'rank': 1, 'n_iter_max': 3, 'init': 'random', 'random_state': 0}),
            ('randomised_parafac', {
                'rank': 1, 'n_samples': 12, 'n_iter_max': 3, 'random_state': 0,
            }),
            ('non_negative_parafac', {'rank': 1, 'n_iter_max': 3, 'random_state': 0}),
            ('non_negative_parafac_hals', {'rank': 1, 'n_iter_max': 3, 'random_state': 0}),
            ('cp_constrained', {
                'rank': 1, 'n_iter_max': 3, 'n_iter_max_inner': 2,
                'random_state': 0,
            }),
        )
        for method, kwargs in methods:
            with self.subTest(method=method):
                self.assert_reconstruction_and_errors(method, kwargs)

    def test_tucker_methods(self):
        methods = (
            ('tucker', {'rank': (2, 2, 2), 'n_iter_max': 3}),
            ('partial_tucker', {'rank': (2, 2), 'modes': (1, 2), 'n_iter_max': 3}),
            ('non_negative_tucker', {'rank': (2, 2, 2), 'n_iter_max': 3}),
            ('non_negative_tucker_hals', {'rank': (2, 2, 2), 'n_iter_max': 3}),
        )
        for method, kwargs in methods:
            with self.subTest(method=method):
                self.assert_reconstruction_and_errors(method, kwargs)

    def test_parafac2_and_robust_pca(self):
        methods = (
            ('parafac2', {
                'rank': 1, 'n_iter_max': 3, 'n_iter_parafac': 1,
                'linesearch': False, 'random_state': 0,
            }),
            ('robust_pca', {'n_iter_max': 3, 'verbose': 0}),
        )
        for method, kwargs in methods:
            with self.subTest(method=method):
                self.assert_reconstruction_and_errors(method, kwargs)

    def test_tensor_ring_errors_and_user_callback(self):
        for method, kwargs in (
            ('tensor_ring_als', {'rank': 1, 'n_iter_max': 3, 'random_state': 0}),
            ('tensor_ring_als_sampled', {
                'rank': 1, 'n_samples': 10, 'n_iter_max': 3,
                'random_state': 0,
            }),
        ):
            seen = []

            def callback(_decomposition, relative_error):
                seen.append(float(relative_error))

            with self.subTest(method=method):
                reconstruction, errors = fd.HyperData(self.tensor).denoise(
                    method, return_errors=True, callback=callback, **kwargs,
                )
                self.assertIsInstance(reconstruction, fd.HyperData)
                self.assertEqual(reconstruction.shape, self.tensor.shape)
                np.testing.assert_allclose(errors, seen)
                self.assertGreaterEqual(len(errors), 1)

    def test_four_dimensional_unfold_returns_refolded_data_and_errors(self):
        original = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3) + 0.1
        data = fd.HyperData(
            original, real_units='nm', real_conv_factor=2,
        )
        reconstruction, errors = data.denoise(
            'tucker', rank=(2, 2, 2), n_iter_max=3,
            unfold_domain='real', return_errors=True,
        )
        self.assertIsInstance(reconstruction, fd.HyperData)
        self.assertEqual(reconstruction.shape, original.shape)
        self.assertEqual(reconstruction.real_conv_factor, 2)
        self.assertGreaterEqual(len(errors), 1)

        array, array_errors = data.denoise(
            'tucker', rank=(2, 2, 2), n_iter_max=3,
            unfold_domain='real', return_errors=True, return_array=True,
        )
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, original.shape)
        self.assertGreaterEqual(len(array_errors), 1)

    def test_mask_is_unfolded_with_the_data(self):
        original = np.random.default_rng(12).random((2, 3, 4, 5))
        mask = np.ones_like(original, dtype=bool)
        mask[0, 0, 0, 0] = False
        data = fd.HyperData(original)

        reconstruction, errors = data.denoise(
            'tucker', rank=(2, 2, 2), n_iter_max=3,
            unfold_domain='real', mask=mask, return_errors=True,
        )
        self.assertEqual(reconstruction.shape, original.shape)
        self.assertGreaterEqual(len(errors), 1)

        with self.assertRaisesRegex(ValueError, 'mask must match'):
            data.denoise(
                'tucker', rank=(2, 2, 2), n_iter_max=3,
                unfold_domain='real', mask=np.ones((2, 3)),
            )

    def test_slice_wise_errors_are_rejected(self):
        data = fd.HyperData(np.ones((2, 2, 3, 3)))
        with self.assertRaisesRegex(ValueError, 'Slice-wise 4D'):
            data.denoise('parafac', rank=1, return_errors=True)

    def test_fixed_factor_tucker_does_not_claim_errors(self):
        with self.assertRaisesRegex(NotImplementedError, 'fixed_factors'):
            fd.HyperData(self.tensor).denoise(
                'tucker', rank=(2, 2, 2), fixed_factors=(0,),
                return_errors=True,
            )

    def test_existing_decomposition_payload_is_unchanged(self):
        result = fd.HyperData(self.tensor).denoise(
            'parafac', rank=1, n_iter_max=3, init='random', random_state=0,
            return_decomposition=True, return_errors=True,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2].shape, self.tensor.shape)
        self.assertGreaterEqual(len(result[3]), 1)

    def test_convergence_plot_keeps_default_return_and_metadata(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        data = fd.HyperData(
            self.tensor, reciprocal_units='mrad', reciprocal_conv_factor=0.2,
        )
        denoised = data.denoise(
            'parafac', rank=1, n_iter_max=3, init='random', random_state=0,
            convergence_plot=True, convergence_ax=ax, convergence_show=False,
        )
        self.assertIsInstance(denoised, fd.HyperData)
        self.assertEqual(denoised.shape, data.shape)
        self.assertEqual(denoised.reciprocal_units, 'mrad')
        self.assertEqual(len(ax.lines), 1)
        self.assertGreaterEqual(len(ax.lines[0].get_ydata()), 1)

    def test_convergence_plot_and_errors_share_one_history(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        denoised, errors = fd.HyperData(self.tensor).denoise(
            'robust_pca', n_iter_max=3, verbose=0, return_errors=True,
            convergence_plot=True, convergence_ax=ax, convergence_show=False,
            return_array=True,
        )
        self.assertIsInstance(denoised, np.ndarray)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), errors)
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), np.arange(1, len(errors) + 1),
        )

    def test_convergence_plot_refolds_unfolded_four_dimensional_data(self):
        original = np.random.default_rng(15).random((2, 3, 4, 5))
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        denoised = fd.HyperData(original).denoise(
            'tucker', rank=(2, 2, 2), n_iter_max=3,
            unfold_domain='real', unfold_method='serpentine',
            convergence_plot=True, convergence_ax=ax, convergence_show=False,
        )
        self.assertIsInstance(denoised, fd.HyperData)
        self.assertEqual(denoised.shape, original.shape)
        self.assertEqual(len(ax.lines), 1)

    def test_tensor_ring_plot_includes_initial_iteration(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        _, errors = fd.HyperData(self.tensor).denoise(
            'tensor_ring_als', rank=1, n_iter_max=3, random_state=0,
            return_errors=True, convergence_plot=True, convergence_ax=ax,
            convergence_show=False,
        )
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), errors)
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), np.arange(len(errors)),
        )

    def test_convergence_plot_rejects_unsupported_or_incompatible_routing(self):
        data = fd.HyperData(self.tensor)
        with self.assertRaisesRegex(ValueError, 'does not report'):
            data.denoise('median', window_size=3, convergence_plot=True)
        with self.assertRaisesRegex(ValueError, 'return_decomposition=False'):
            data.denoise(
                'parafac', rank=1, convergence_plot=True,
                return_decomposition=True,
            )
        with self.assertRaisesRegex(ValueError, 'convergence_ax requires'):
            data.denoise('parafac', rank=1, convergence_ax=plt.gca())
        with self.assertRaisesRegex(ValueError, 'Slice-wise 4D'):
            fd.HyperData(np.ones((2, 2, 3, 3))).denoise(
                'parafac', rank=1, convergence_plot=True,
            )


if __name__ == '__main__':
    unittest.main()
