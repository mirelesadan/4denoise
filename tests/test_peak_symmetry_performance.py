"""Regression checks for peak-orbit assignment and stack search reuse."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

import fourdenoise as fd


def disk_pattern(shape, positions, radius=2):
    image = np.zeros(shape, dtype=float)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for y, x in positions:
        image[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = 100
    return image


class PeakSymmetryAndSearchTests(TestCase):
    def test_repair_labels_only_missing_orbit_slot_as_synthetic(self):
        pattern = disk_pattern((65, 65), [(20, 32), (32, 44), (44, 32)])
        result = fd.ReciprocalSpace(pattern).get_peaks(
            radius=2, min_distance=4, threshold_abs=1,
            n_fold=4, sym_mode='repair', center=(32, 32),
            orbit_min_fraction=0.75, return_details=True, reorder=True,
        )
        self.assertIsInstance(result, fd.PeakDetectionResult)
        self.assertEqual(len(result.coords), 4)
        self.assertEqual(int(np.count_nonzero(result.synthetic_mask)), 1)
        np.testing.assert_array_equal(
            result.coords[result.synthetic_mask], [[32, 20]],
        )
        self.assertTrue(np.all(np.isfinite(result.scores[result.observed_mask])))
        self.assertTrue(np.all(np.isnan(result.scores[result.synthetic_mask])))
        self.assertEqual(len(set(result.orbit_ids)), 1)

    def test_repair_does_not_create_out_of_bounds_or_outside_annulus(self):
        pattern = disk_pattern((41, 41), [(2, 13), (13, 24)])
        result = fd.ReciprocalSpace(pattern).get_peaks(
            radius=2, min_distance=4, threshold_abs=1,
            n_fold=4, sym_mode='repair', center=(2, 2),
            r_range=(9, 12), orbit_min_fraction=0.5,
            return_details=True,
        )
        self.assertTrue(np.all(result.coords >= 0))
        self.assertTrue(np.all(result.coords < 41))
        radial = np.linalg.norm(result.coords - [2, 2], axis=1)
        self.assertTrue(np.all((radial >= 9) & (radial <= 12)))

    def test_ambiguous_orbits_cannot_reuse_a_measured_peak(self):
        pattern = disk_pattern(
            (65, 65), [(22, 32), (25, 32), (32, 41)], radius=1,
        )
        result = fd.ReciprocalSpace(pattern).get_peaks(
            radius=1, min_distance=2, threshold_abs=1,
            n_fold=4, sym_mode='repair', center=(32, 32),
            sym_tolerance_px=2, orbit_min_fraction=0.5,
            return_details=True,
        )
        self.assertEqual(int(np.count_nonzero(result.synthetic_mask)), 2)
        self.assertEqual(
            int(np.count_nonzero(result.orbit_ids[result.observed_mask] >= 0)),
            2,
        )
        self.assertEqual(
            int(np.count_nonzero(result.orbit_ids[result.observed_mask] == -1)),
            1,
        )

    def test_cropped_search_matches_full_image_search(self):
        pattern = disk_pattern(
            (128, 128), [(64, 78), (64, 49), (77, 64), (10, 10)],
        )
        pattern[66, 66] = np.nan
        detector = fd.ReciprocalSpace(pattern)
        params = dict(
            radius=2, min_distance=4, threshold_abs=1,
            center=(64, 64), return_details=True,
        )
        full = detector.get_peaks(**params)
        cropped = detector.get_peaks(**params, r_range=(10, 17))
        radial = np.linalg.norm(full.coords - [64, 64], axis=1)
        expected = full.coords[(radial >= 10) & (radial <= 17)]
        self.assertEqual(set(map(tuple, cropped.coords)), set(map(tuple, expected)))
        full_scores = dict(zip(map(tuple, full.coords), full.scores))
        for coord, score in zip(cropped.coords, cropped.scores):
            self.assertAlmostEqual(score, full_scores[tuple(coord)], places=8)

    def test_stack_constructs_detector_once_and_keeps_ragged_result(self):
        pattern = disk_pattern((65, 65), [(20, 32), (32, 44), (44, 32)])
        stack = fd.HyperData(np.stack([pattern, pattern, pattern]))
        with patch.object(
            stack, '_spawn_reciprocal', wraps=stack._spawn_reciprocal,
        ) as spawn:
            results = stack.get_peaks(
                radius=2, min_distance=4, threshold_abs=1,
                n_fold=4, sym_mode='repair', center=(32, 32),
                return_details=True,
            )
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, fd.PeakDetectionResult)
            self.assertEqual(len(result.coords), 4)

        scan = fd.HyperData(np.stack([pattern, pattern])[None, :, :, :])
        mapped = scan.get_peaks(
            radius=2, min_distance=4, threshold_abs=1,
            real_mask=[[True, False]], return_details=True,
        )
        self.assertEqual(mapped[0][1].coords.shape, (0, 2))
        self.assertEqual(mapped[0][1].scores.shape, (0,))
