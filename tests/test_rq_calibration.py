"""Orientation, GUI-pivot, and persistence regressions for R/Q comparison."""

from pathlib import Path
import gc
import sys
import tempfile
import unittest
import weakref

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':
    raise ImportError(f"Imported the wrong module: {fd.__file__}")


class CalibrationTests(unittest.TestCase):
    def test_rotation_reflection_and_inverse_roundtrip(self):
        for mirror in (None, 'x', 'y'):
            calibration = fd.RQCalibration(37.25, mirror)
            vectors = np.array([[1., 0.], [0., 1.], [4., -2.]])
            rotated = calibration.transform_vectors(vectors)
            np.testing.assert_allclose(calibration.transform_vectors(rotated, 'reciprocal_to_real'), vectors, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(vectors, axis=1))
            reconstructed = fd.RQCalibration.from_matrix(calibration.matrix, mirror_axis=mirror)
            np.testing.assert_allclose(reconstructed.matrix, calibration.matrix, atol=1e-12)
        np.testing.assert_allclose(fd.RQCalibration(90).transform_vectors([1, 0]), [0, 1], atol=1e-12)
        np.testing.assert_array_equal(fd.RQCalibration(0, 'x').transform_vectors([2, 3]), [2, -3])

    def test_invalid_calibrations(self):
        for angle in (np.nan, np.inf, True, [1, 2]):
            with self.assertRaises(ValueError):
                fd.RQCalibration(angle)
        with self.assertRaises(ValueError):
            fd.RQCalibration(0, 'z')
        with self.assertRaises(ValueError):
            fd.RQCalibration.from_matrix([[2, 0], [0, 1]])
        with self.assertRaises(ValueError):
            fd.HyperData(np.ones((2, 2)), rq_calibration={'version': 99})

    def test_saved_copy_chunk_and_value_operations_preserve_orientation(self):
        data = fd.HyperData(np.ones((2, 3, 6, 8))).set_rq_calibration(33, 'y')
        self.assertIsNot(data.copy().array, data.array)
        np.testing.assert_allclose(data.copy().rq_calibration.matrix, data.rq_calibration.matrix)
        np.testing.assert_allclose(data.clip().rq_calibration.matrix, data.rq_calibration.matrix)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibrated.4denoise'
            data.save(path)
            restored = fd.HyperData(path)
            self.assertEqual(restored.rq_calibration, data.rq_calibration)
            with fd.HyperData.open_hdf5(path) as reader:
                _, chunk = next(reader.iter_chunks(1))
                self.assertEqual(chunk.rq_calibration, data.rq_calibration)
            plain = fd.HyperData(np.ones((2, 2)))
            plain.save(Path(directory) / 'plain.4denoise')
            self.assertIsNone(fd.HyperData(Path(directory) / 'plain.4denoise').rq_calibration)

    def test_flips_rotate_and_domain_swap_update_orientation(self):
        original = fd.RQCalibration(22., 'x')
        source = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        flipped = fd.HyperData(source, rq_calibration=original, flip_axis=(0, 3))
        np.testing.assert_allclose(flipped.rq_calibration.matrix,
                                   np.diag([-1, 1]) @ original.matrix @ np.diag([1, -1]), atol=1e-12)
        data = fd.HyperData(source, rq_calibration=original)
        rotated = data.rotate_dps(90, order=0)
        np.testing.assert_allclose(rotated.rq_calibration.matrix,
                                   np.array([[0, -1], [1, 0]]) @ original.matrix, atol=1e-12)
        np.testing.assert_allclose(data.swap_domains().rq_calibration.matrix, original.inverse_matrix, atol=1e-12)
        self.assertIsNone(data.reshape(2, 3, 2, 8).rq_calibration)
        self.assertEqual(data.unfold(method='serpentine').unfold(undo=True).rq_calibration, original)
        data.array = source + 1
        self.assertEqual(data.rq_calibration, original)
        data.array = np.ones((2, 3, 2, 8))
        self.assertIsNone(data.rq_calibration)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.data = fd.HyperData(np.ones((6, 8, 6, 8)))
        self.viewers = []

    def tearDown(self):
        for viewer in self.viewers:
            viewer.close()

    def viewer(self, real=None, reciprocal=None, **kwargs):
        viewer = self.data.compare_rq(
            np.ones((6, 8)) if real is None else real,
            np.ones((6, 8)) if reciprocal is None else reciprocal,
            scale=1, show=False, **kwargs,
        )
        self.viewers.append(viewer)
        return viewer

    def test_offcenter_fractional_pivot_is_stationary_and_ccw(self):
        self.data.center_beam_metadata = {'center_px': (1.25, 4.75), 'shape': (6, 8)}
        viewer = self.viewer(interactive=False)
        self.assertEqual(viewer.center, (1.25, 4.75))
        original = self.data.array.copy()
        base_transform = viewer.real_artist.get_transform().get_matrix().copy()
        target = (2.5, 3.5)
        for angle in (0, 37.5, 90, -123):
            viewer.set_parameters(rotation_deg=angle)
            np.testing.assert_allclose(viewer.transform_points(viewer.center), target, atol=1e-12)
        viewer.set_parameters(rotation_deg=90)
        # One source pixel right of the pivot moves one reference pixel upward.
        np.testing.assert_allclose(viewer.transform_points((1.25, 5.75)), (1.5, 3.5), atol=1e-12)
        rendered = viewer.reciprocal_artist.get_transform().transform((5.75, 1.25))
        np.testing.assert_allclose(rendered, viewer.axes[-1].transData.transform((3.5, 1.5)), atol=1e-10)
        np.testing.assert_array_equal(viewer.real_artist.get_transform().get_matrix(), base_transform)
        np.testing.assert_array_equal(self.data.array, original)
        self.assertIsNone(self.data.rq_calibration)

    def test_midpoint_own_metadata_and_calibrated_only_center(self):
        viewer = self.viewer(interactive=False)
        self.assertEqual(viewer.center, (2.5, 3.5))
        self.data.center_beam_metadata = {'center_px': (1, 1)}
        external = fd.ReciprocalSpace(np.ones((6, 8)), center_beam_metadata={'center_px': (2.25, 3.75)})
        viewer = self.viewer(reciprocal=external, interactive=False)
        self.assertEqual(viewer.center, (2.25, 3.75))
        external.center_beam_metadata = dict(center_calibrated=(0.3, -0.2), conv_factor=0.2)
        viewer = self.viewer(reciprocal=external, interactive=False)
        np.testing.assert_allclose(viewer.center, (1, 2.5))
        self.assertEqual(viewer.center_source, 'center_calibrated')

    def test_invalid_input_does_not_create_figures(self):
        existing = set(plt.get_fignums())
        self.data.center_beam_metadata = {'center_px': (1, 2), 'shape': (10, 10)}
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.viewer()
        self.data.center_beam_metadata = None
        for kwargs in ({'real_alpha': -1}, {'reciprocal_alpha': np.nan}, {'layout': 'wrong'}):
            with self.assertRaises(ValueError):
                self.viewer(**kwargs)
        polar = fd.ReciprocalSpace(np.ones((6, 8)), polar_metadata={'kind': 'polar'})
        with self.assertRaisesRegex(ValueError, 'Cartesian'):
            self.viewer(reciprocal=polar)
        self.assertEqual(set(plt.get_fignums()), existing)

    def test_gui_callbacks_opacities_and_angle_entry(self):
        viewer = self.viewer()
        viewer.controls['real_alpha'].set_val(.23)
        viewer.controls['reciprocal_alpha'].set_val(.81)
        viewer.controls['angle_entry'].set_val('12.375')
        self.assertAlmostEqual(viewer.real_artist.get_alpha(), .23)
        self.assertAlmostEqual(viewer.reciprocal_artist.get_alpha(), .81)
        self.assertAlmostEqual(viewer.parameters['rotation_deg'], 12.375)
        state = viewer.parameters
        viewer.controls['angle_entry'].set_val('invalid')
        self.assertEqual(viewer.parameters, state)
        viewer.controls['mirror_axis'].set_active(2)
        self.assertEqual(viewer.parameters['mirror_axis'], 'y')
        viewer.reset()
        self.assertAlmostEqual(viewer.parameters['rotation_deg'], 0)
        self.assertIsNone(viewer.parameters['mirror_axis'])

    def test_apply_and_reopen_preserve_inverse_for_both_mirrors(self):
        for axis in (None, 'x', 'y'):
            viewer = self.viewer(interactive=False)
            viewer.set_parameters(rotation_deg=31.5, mirror_axis=axis, translation=(1, -2), scale=2.)
            applied = viewer.apply()
            self.assertEqual(self.data.rq_calibration, applied)
            expected = np.array([[np.cos(np.deg2rad(31.5)), -np.sin(np.deg2rad(31.5))],
                                 [np.sin(np.deg2rad(31.5)), np.cos(np.deg2rad(31.5))]])
            if axis == 'x':
                expected = expected @ np.diag([1, -1])
            elif axis == 'y':
                expected = expected @ np.diag([-1, 1])
            np.testing.assert_allclose(applied.inverse_matrix, expected, atol=1e-12)
            reopened = self.viewer(interactive=False)
            self.assertAlmostEqual(reopened.parameters['rotation_deg'], 31.5)
            self.assertEqual(reopened.parameters['mirror_axis'], axis)

    def test_anisotropic_scan_preserves_physical_angles_in_either_tick_mode(self):
        real = fd.RealSpace(np.ones((6, 8)), units='nm', conv_factor=(2., 1.))
        for units in ('pixels', 'calibrated'):
            viewer = self.viewer(real=real, real_show_kwargs={'axis_units': units}, interactive=False)
            viewer.set_parameters(rotation_deg=90)
            np.testing.assert_allclose(viewer.transform_points((2.5, 4.5)), (2., 3.5), atol=1e-12)

    def test_side_by_side_export_and_dataset_lifetime(self):
        viewer = self.viewer(layout='side_by_side', interactive=False)
        self.assertEqual(len(viewer.axes), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comparison.png'
            viewer.export(path)
            self.assertGreater(path.stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                viewer.export(path)
        owner = weakref.ref(self.data)
        self.data = None
        gc.collect()
        self.assertIsNone(owner())
        with self.assertRaisesRegex(RuntimeError, 'no longer exists'):
            viewer.apply()


if __name__ == '__main__':
    unittest.main()
