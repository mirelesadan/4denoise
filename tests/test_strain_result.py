"""Regression tests for structured HyperData strain results."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class StrainResultTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([
            [-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0],
            [2.0, 1.0], [3.0, 2.0],
        ])
        self.data = fd.HyperData(
            np.zeros((2, 2, 8, 8)),
            real_units='nm', real_conv_factor=(0.5, 0.75),
            real_origin=(10.0, 20.0),
        )
        self.measured = np.broadcast_to(
            self.reference, (2, 2) + self.reference.shape,
        ).copy()

    def test_result_names_quality_and_calibrated_maps(self):
        result = self.data.get_strains(
            centers=self.measured, ref_centers=self.reference,
            center=(0, 0), match_peaks='ordered',
        )

        self.assertIsInstance(result, fd.StrainResult)
        self.assertEqual(result.exx.shape, (2, 2))
        np.testing.assert_allclose(result.exx, 0, atol=1e-12)
        np.testing.assert_allclose(result.eyy, 0, atol=1e-12)
        np.testing.assert_allclose(result.exy, 0, atol=1e-12)
        np.testing.assert_allclose(result.erot, 0, atol=1e-12)
        np.testing.assert_allclose(result.fit_rmse, 0, atol=1e-12)
        np.testing.assert_allclose(result.relative_fit_rmse, 0, atol=1e-12)
        np.testing.assert_array_equal(result.match_counts, 5)
        self.assertTrue(result.valid_mask.all())
        self.assertEqual(result.peak_origin, (0.0, 0.0))
        self.assertEqual(result.peak_origin_source, 'user')
        self.assertEqual(result.real_units, 'nm')
        self.assertEqual(result.real_conv_factor, (0.5, 0.75))
        self.assertEqual(result.real_origin, (10.0, 20.0))

        image = result.as_real_space('exx')
        self.assertIsInstance(image, fd.RealSpace)
        self.assertIs(image.array, result.exx)
        self.assertEqual(image.units, 'nm')
        self.assertEqual(image.conv_factor, (0.5, 0.75))
        self.assertEqual(image.origin, (10.0, 20.0))
        self.assertEqual(result.as_real_space('erot').value_units, 'rad')
        self.assertEqual(result.as_real_space('match_counts').value_units, 'peaks')

        exx, eyy, exy, erot = result
        self.assertIs(exx, result.exx)
        self.assertIs(eyy, result.eyy)
        self.assertIs(exy, result.exy)
        self.assertIs(erot, result.erot)
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], result.exx)

    def test_final_weighted_residual_matches_reported_error(self):
        measured = self.measured.copy()
        measured[0, 0, 0] += [0.35, -0.2]
        weights = np.broadcast_to(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (2, 2, 5),
        )
        result = self.data.get_strains(
            centers=measured, ref_centers=self.reference,
            intensity_array=weights, center=(0, 0),
            match_peaks='ordered', fit_translation=True,
            return_transform=True,
        )

        self.assertEqual(len(result), 5)
        self.assertIs(result[4], result.diagnostics)
        exx, eyy, exy, erot, diagnostics = result
        self.assertIs(exx, result.exx)
        self.assertIs(eyy, result.eyy)
        self.assertIs(exy, result.exy)
        self.assertIs(erot, result.erot)
        transform = diagnostics['transforms'][0, 0]
        translation = diagnostics['translations'][0, 0]
        residual = measured[0, 0] - (
            self.reference @ transform.T + translation
        )
        expected_rmse = np.sqrt(np.average(
            np.sum(residual ** 2, axis=1), weights=weights[0, 0],
        ))
        reference_rms = np.sqrt(np.average(
            np.sum(self.reference ** 2, axis=1), weights=weights[0, 0],
        ))
        self.assertAlmostEqual(result.fit_rmse[0, 0], expected_rmse)
        self.assertAlmostEqual(
            result.relative_fit_rmse[0, 0], expected_rmse / reference_rms,
        )
        self.assertGreater(result.fit_rmse[0, 0], 0)
        self.assertIs(diagnostics['fit_rmse'], result.fit_rmse)

    def test_auto_matching_is_invariant_to_equal_count_peak_order(self):
        permutation = [2, 0, 4, 1, 3]
        shuffled = self.measured[:, :, permutation, :]
        automatic = self.data.get_strains(
            centers=shuffled, ref_centers=self.reference, center=(0, 0),
        )
        for component in ('exx', 'eyy', 'exy', 'erot', 'fit_rmse'):
            np.testing.assert_allclose(
                getattr(automatic, component), 0, atol=1e-12,
            )
        np.testing.assert_array_equal(automatic.match_counts, 5)

        ordered = self.data.get_strains(
            centers=shuffled, ref_centers=self.reference, center=(0, 0),
            match_peaks='ordered',
        )
        self.assertGreater(float(ordered.fit_rmse[0, 0]), 0.1)

    def test_auto_matching_keeps_intensity_weights_with_measured_peaks(self):
        measured = self.measured.copy()
        measured[:, :, 0] += [0.35, -0.2]
        weights = np.broadcast_to(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (2, 2, 5),
        )
        permutation = [2, 0, 4, 1, 3]
        automatic = self.data.get_strains(
            centers=measured[:, :, permutation, :],
            ref_centers=self.reference,
            intensity_array=weights[:, :, permutation],
            center=(0, 0), fit_translation=True,
        )
        ordered = self.data.get_strains(
            centers=measured, ref_centers=self.reference,
            intensity_array=weights, center=(0, 0),
            fit_translation=True, match_peaks='ordered',
        )
        for component in ('exx', 'eyy', 'exy', 'erot', 'fit_rmse'):
            np.testing.assert_allclose(
                getattr(automatic, component),
                getattr(ordered, component), atol=1e-12,
            )

    def test_masked_locations_have_no_fit_quality(self):
        mask = np.array([[True, False], [True, True]])
        result = self.data.get_strains(
            centers=self.measured, ref_centers=self.reference,
            center=(0, 0), real_mask=mask,
        )

        np.testing.assert_array_equal(result.valid_mask, mask)
        self.assertTrue(np.isnan(result.fit_rmse[0, 1]))
        self.assertTrue(np.isnan(result.relative_fit_rmse[0, 1]))
        self.assertEqual(result.match_counts[0, 1], 0)

    def test_unrelated_scan_geometry_does_not_claim_calibration(self):
        result = self.data.get_strains(
            centers=self.measured[:1], ref_centers=self.reference,
            center=(0, 0),
        )

        self.assertIsNone(result.real_units)
        self.assertIsNone(result.real_conv_factor)
        self.assertEqual(result.real_origin, (0.0, 0.0))
        self.assertIsNone(result.as_real_space('exx').units)

    def test_stack_result_rejects_realspace_conversion(self):
        stack = fd.HyperData(np.zeros((3, 8, 8)))
        measured = np.broadcast_to(
            self.reference, (3,) + self.reference.shape,
        )
        result = stack.get_strains(
            centers=measured, ref_centers=self.reference, center=(0, 0),
        )

        self.assertEqual(result.exx.shape, (3,))
        self.assertIsNone(result.real_units)
        with self.assertRaisesRegex(ValueError, 'requires a 2D scan map'):
            result.as_real_space('exx')
        with self.assertRaisesRegex(ValueError, 'Unknown strain component'):
            result.as_real_space('something_else')

    def test_mismatched_intensity_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'peak count'):
            self.data.get_strains(
                centers=self.measured, ref_centers=self.reference,
                intensity_array=np.ones((2, 2, 4)), center=(0, 0),
            )

    def test_collinear_or_ill_conditioned_peaks_have_no_valid_fit(self):
        for reference in (
            np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
            np.array([[1.0, 0.0], [2.0, 1e-11], [3.0, 0.0]]),
        ):
            with self.subTest(reference=reference):
                measured = np.broadcast_to(reference, (2, 2) + reference.shape)
                result = self.data.get_strains(
                    centers=measured, ref_centers=reference,
                    center=(0, 0), match_peaks='ordered',
                )
                self.assertFalse(result.valid_mask.any())
                self.assertTrue(np.isnan(result.fit_rmse).all())

    def test_translation_fit_needs_two_dimensional_peak_geometry(self):
        reference = np.array([[0.0, 0.0], [1.0, 1.0]])
        measured = np.broadcast_to(reference, (2, 2, 2, 2))
        result = self.data.get_strains(
            centers=measured, ref_centers=reference,
            center=(0, 0), fit_translation=True,
        )
        self.assertFalse(result.valid_mask.any())

    def test_calibrated_peak_units_convert_stored_pixel_center(self):
        data = fd.HyperData(
            np.zeros((2, 2, 8, 8)),
            reciprocal_units='Å^-1', reciprocal_conv_factor=2.0,
            center_beam_metadata={'mean_fit_center_px': (4.0, 3.0)},
        )
        measured = np.broadcast_to(
            self.reference, (2, 2) + self.reference.shape,
        )
        result = data.get_strains(
            centers=measured, ref_centers=self.reference,
            peak_units='calibrated', match_peaks='ordered',
            return_transform=True,
        )
        self.assertEqual(result.peak_origin, (-1.0, -1.0))
        self.assertEqual(result.peak_units, 'Å^-1')
        self.assertEqual(result.as_real_space('fit_rmse').value_units, 'Å^-1')
        self.assertEqual(result.diagnostics['peak_units'], 'calibrated')
        self.assertTrue(result.valid_mask.all())

        with self.assertRaisesRegex(ValueError, 'requires reciprocal_units'):
            self.data.get_strains(
                centers=self.measured, ref_centers=self.reference,
                peak_units='calibrated',
            )


if __name__ == '__main__':
    unittest.main()
