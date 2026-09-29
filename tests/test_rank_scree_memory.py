"""Rank-sweep reuse, bounded residuals, and reconstruction semantics."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


def _scaled_rank(_self, tensor, rank, **_kwargs):
    """Deterministic stand-in for a rank-dependent TensorLy reconstruction."""
    return np.asarray(tensor, dtype=np.float64) * (rank / (rank + 1.0))


class RankScreeMemoryTests(unittest.TestCase):
    def test_block_norm_handles_noncontiguous_and_unsigned_arrays(self):
        original = np.array([[0, 65535], [10, 11]], dtype=np.uint16).T
        reconstruction = np.array(
            [[1, 65534], [9, 12]], dtype=np.uint16,
        ).T

        residual_norm = fd.HyperData._rank_scree_block_norm(
            original, reconstruction, block_elements=2,
        )
        original_norm = fd.HyperData._rank_scree_block_norm(
            original, block_elements=2,
        )

        self.assertAlmostEqual(residual_norm, 2.0)
        self.assertAlmostEqual(
            original_norm,
            np.linalg.norm(original.astype(np.float64)),
        )

    def test_scree_without_unfolding_has_same_metrics(self):
        original = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        data = fd.HyperData(original)

        with patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank):
            result = data.rank_scree(
                'parafac', [1, 3], plot=False, progress=False,
                error_chunk_elements=5,
            )

        np.testing.assert_allclose(result['relative_error'], [0.5, 0.25])
        np.testing.assert_allclose(
            result['residual_norm'],
            [np.linalg.norm(original) / 2, np.linalg.norm(original) / 4],
        )
        self.assertNotIn('reconstructions', result)

    def test_single_unfolded_denoise_still_refolds(self):
        original = np.arange(1, 73, dtype=np.float32).reshape(3, 4, 2, 3)
        data = fd.HyperData(original)

        with patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank):
            denoised = data.denoise(
                'parafac', rank=2, unfold_domain='reciprocal',
                unfold_method='serpentine',
            )

        self.assertIsInstance(denoised, fd.HyperData)
        np.testing.assert_allclose(denoised.array, original * (2 / 3))
        np.testing.assert_array_equal(data.array, original)

    def test_full_traversals_unfold_once_without_refolding(self):
        original = np.arange(1, 73, dtype=np.float32).reshape(3, 4, 2, 3)
        for domain in ('real', 'reciprocal', 'both'):
            with self.subTest(domain=domain):
                data = fd.HyperData(original)
                with (
                    patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank),
                    patch.object(data, 'unfold', wraps=data.unfold) as unfold_spy,
                    patch.object(fd, '_unfold_array', wraps=fd._unfold_array) as array_spy,
                ):
                    result = data.rank_scree(
                        'parafac', [1, 2, 4], unfold_domain=domain,
                        unfold_method='row_major', plot=False, progress=False,
                        error_chunk_elements=7,
                    )

                self.assertEqual(unfold_spy.call_count, 1)
                self.assertEqual(array_spy.call_count, 1)
                np.testing.assert_allclose(
                    result['relative_error'], [0.5, 1 / 3, 0.2],
                )

    def test_preserved_center_crop_measures_only_changed_values(self):
        original = np.arange(1, 5 * 7 * 2 * 3 + 1, dtype=np.float64).reshape(
            5, 7, 2, 3,
        )
        data = fd.HyperData(original)

        with (
            patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank),
            patch.object(fd, '_unfold_array', wraps=fd._unfold_array) as array_spy,
        ):
            result = data.rank_scree(
                'parafac', [1, 3], unfold_domain='real',
                unfold_method='hilbert', plot=False, progress=False,
                error_chunk_elements=11,
            )

        self.assertEqual(array_spy.call_count, 1)
        np.testing.assert_allclose(result['relative_error'], [0.5, 0.25])
        self.assertEqual(result['comparison_shape'], (16, 2, 3))

    def test_requested_reconstructions_are_refolded_and_retained(self):
        original = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 2, 2)
        data = fd.HyperData(original)

        with (
            patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank),
            patch.object(fd, '_unfold_array', wraps=fd._unfold_array) as array_spy,
        ):
            result = data.rank_scree(
                'parafac', [1, 2], unfold_domain='real',
                plot=False, progress=False, return_reconstructions=True,
                error_chunk_elements=5,
            )

        self.assertEqual(array_spy.call_count, 3)
        self.assertEqual(len(result['reconstructions']), 2)
        np.testing.assert_allclose(result['reconstructions'][0], original / 2)
        np.testing.assert_allclose(
            result['reconstructions'][1], original * (2 / 3),
        )
        np.testing.assert_allclose(result['relative_error'], [0.5, 1 / 3])

    def test_processing_rejects_resize_with_preserved_original(self):
        original = np.arange(1, 5 * 7 * 2 * 2 + 1, dtype=np.float32).reshape(
            5, 7, 2, 2,
        )
        data = fd.HyperData(original)

        options = {
            'curve_shape_strategy': 'resize',
            'preserve_original': True,
        }
        with patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank):
            with self.assertRaisesRegex(ValueError, 'discard the processed tensor'):
                data.rank_scree(
                    'parafac', [1, 2], unfold_domain='real',
                    unfold_method='hilbert', unfold_kwargs=options,
                    plot=False, progress=False,
                )
            with self.assertRaisesRegex(ValueError, 'discard the processed tensor'):
                data.denoise(
                    'parafac', rank=2, unfold_domain='real',
                    unfold_method='z_order', unfold_kwargs=options,
                )

        unfolded = data.unfold(
            domain='real', method='hilbert', return_metadata=True, **options,
        )
        restored = unfolded[0].unfold(undo=True, metadata=unfolded[1])
        np.testing.assert_array_equal(restored.array, original)

    def test_resize_scree_compares_processed_tensor_and_returns_resized_shape(self):
        original = np.arange(1, 5 * 7 * 2 * 2 + 1, dtype=np.float32).reshape(
            5, 7, 2, 2,
        )
        data = fd.HyperData(original)
        with patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank):
            result = data.rank_scree(
                'parafac', [1, 3], unfold_domain='real',
                unfold_method='hilbert',
                unfold_kwargs={'curve_shape_strategy': 'resize'},
                plot=False, progress=False, return_reconstructions=True,
            )
        np.testing.assert_allclose(result['relative_error'], [0.5, 0.25])
        self.assertEqual(result['comparison_shape'], (16, 2, 2))
        self.assertEqual(result['reconstructions'][0].shape, (4, 4, 2, 2))

    def test_error_chunk_size_must_be_positive_integer(self):
        data = fd.HyperData(np.ones((2, 3, 4)))
        with patch.object(fd._DenoisingMethods, 'parafac', new=_scaled_rank):
            for value in (0, -1, 1.5, True):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        data.rank_scree(
                            'parafac', [1], plot=False,
                            error_chunk_elements=value,
                        )


if __name__ == '__main__':
    unittest.main()
