"""Regression tests for detector-count fidelity during spatial resampling."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / "fourdenoise.py":
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class ResamplingFidelityTests(unittest.TestCase):
    def test_hyper_crop_preserves_integer_count_range_in_reciprocal_space(self):
        original = np.full((2, 2, 4, 4), 100, dtype=np.uint16)
        data = fd.HyperData(original)

        resized = data.crop(kshape=(2, 2))
        subpixel = data.crop(
            kylim=(0.5, 3.5), kxlim=(0.5, 3.5),
            reciprocal_limit_units="pixels",
        )

        self.assertEqual(resized.array.dtype.kind, "f")
        self.assertEqual(subpixel.array.dtype.kind, "f")
        np.testing.assert_allclose(resized.array, 100)
        np.testing.assert_allclose(subpixel.array, 100)
        np.testing.assert_array_equal(original, 100)

    def test_hyper_crop_keeps_fractional_block_means_and_interpolation(self):
        scan = np.array([[100, 101], [102, 103]], dtype=np.uint16)
        original = np.broadcast_to(scan[:, :, None, None], (2, 2, 2, 2)).copy()
        binned = fd.HyperData(original).crop(rshape=(1, 1))

        self.assertEqual(binned.array.dtype.kind, "f")
        np.testing.assert_allclose(binned.array, 101.5)

        constant = fd.HyperData(np.full((3, 3, 2, 2), 100, dtype=np.uint16))
        interpolated = constant.crop(rshape=(2, 2))
        self.assertEqual(interpolated.array.dtype.kind, "f")
        np.testing.assert_allclose(interpolated.array, 100)

    def test_reciprocal_crop_avoids_unneeded_resampling(self):
        original = np.full((4, 4), 100, dtype=np.uint16)
        pattern = fd.ReciprocalSpace(original)

        untouched = pattern.crop()
        resized = pattern.crop(kshape=(2, 2))
        subpixel = pattern.crop(kylim=(0.5, 3.5), kxlim=(0.5, 3.5))

        self.assertEqual(untouched.array.dtype, original.dtype)
        np.testing.assert_array_equal(untouched.array, original)
        for result in (resized, subpixel):
            self.assertEqual(result.array.dtype.kind, "f")
            np.testing.assert_allclose(result.array, 100)

    def test_polar_interpolation_keeps_subpixel_values_for_3d_and_4d(self):
        image = np.broadcast_to(np.arange(8, dtype=np.uint16), (8, 8))
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                data = fd.HyperData(image.reshape(stack_shape + (8, 8)))
                polar = data.to_polar(
                    output_shape=(4, 16), order=1, clip=False, progress=False
                )
                nearest = data.to_polar(
                    output_shape=(4, 16), order=0, clip=False, progress=False
                )

                self.assertEqual(polar.array.dtype.kind, "f")
                self.assertEqual(nearest.array.dtype, image.dtype)
                np.testing.assert_allclose(
                    polar.array[(0,) * len(stack_shape) + (0, 0)], 3.5
                )

    def test_cartesian_interpolation_keeps_subpixel_values_for_3d_and_4d(self):
        image = np.broadcast_to(
            np.arange(5, dtype=np.uint16)[:, None], (5, 16)
        )
        metadata = {
            "r_max": 5.0,
            "radius_display_range": (0.0, 5.0),
            "radius_units": "pixels",
        }
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                data = fd.HyperData(
                    image.reshape(stack_shape + (5, 16)),
                    polar_metadata=metadata,
                )
                cartesian = data.to_cartesian(
                    output_shape=(10, 10), order=1,
                    clip=False, progress=False,
                )
                nearest = data.to_cartesian(
                    output_shape=(10, 10), order=0,
                    clip=False, progress=False,
                )

                self.assertEqual(cartesian.array.dtype.kind, "f")
                self.assertEqual(nearest.array.dtype, image.dtype)
                value = cartesian.array[(0,) * len(stack_shape) + (4, 4)]
                self.assertGreater(value, 0)
                self.assertLess(value, 1)

    def test_polar_and_cartesian_do_not_clip_intensities_by_default(self):
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                source = np.full(stack_shape + (8, 8), -2.0, dtype=np.float32)
                data = fd.HyperData(source)
                polar = data.to_polar(
                    output_shape=(4, 16), progress=False,
                )
                polar_clipped = data.to_polar(
                    output_shape=(4, 16), clip=True, progress=False,
                )
                polar_index = (0,) * len(stack_shape) + (0, 0)
                self.assertEqual(polar.array[polar_index], -2.0)
                self.assertEqual(polar_clipped.array[polar_index], 1.0)

                cartesian = polar.to_cartesian(
                    output_shape='original', progress=False,
                )
                cartesian_clipped = polar.to_cartesian(
                    output_shape='original', clip=True, progress=False,
                )
                cartesian_index = (0,) * len(stack_shape) + (3, 3)
                self.assertAlmostEqual(cartesian.array[cartesian_index], -2.0)
                self.assertAlmostEqual(
                    cartesian_clipped.array[cartesian_index], 1.0
                )
                np.testing.assert_array_equal(data.array, source)

                zeros = fd.HyperData(np.zeros_like(source)).to_polar(
                    output_shape=(4, 16), progress=False,
                )
                self.assertEqual(np.count_nonzero(zeros.array), 0)

    def test_interpolated_rotation_retains_fractional_intensities(self):
        image = np.broadcast_to(
            np.arange(7, dtype=np.uint16), (7, 7)
        ).copy()
        expected = scipy_rotate(image.astype(np.float32), 25, order=1)
        for stack_shape in ((1,), (1, 1)):
            with self.subTest(stack_shape=stack_shape):
                data = fd.HyperData(image.reshape(stack_shape + image.shape))
                rotated = data.rotate_dps(25, order=1)
                nearest = data.rotate_dps(25, order=0)
                pattern = rotated.array[(0,) * len(stack_shape)]

                self.assertEqual(rotated.array.dtype.kind, 'f')
                self.assertEqual(nearest.array.dtype, image.dtype)
                np.testing.assert_allclose(pattern, expected)
                self.assertTrue(
                    np.any(np.abs(pattern - np.rint(pattern)) > 1e-3)
                )
                np.testing.assert_array_equal(
                    data.array[(0,) * len(stack_shape)], image
                )

    def test_polar_rotation_keeps_exact_roll_and_promotes_subpixel_shift(self):
        image = np.broadcast_to(
            np.arange(8, dtype=np.uint16), (3, 8)
        ).copy()
        data = fd.HyperData(
            image[None],
            polar_metadata={
                'is_polar': True,
                'axis_order': ('radius', 'theta'),
                'theta_range': (0.0, 360.0),
                'theta_step': 45.0,
            },
        )

        exact = data.rotate_dps(45.0)
        fractional = data.rotate_dps(22.5, order=1)

        self.assertEqual(exact.array.dtype, image.dtype)
        np.testing.assert_array_equal(exact.array[0], np.roll(image, -1, axis=-1))
        self.assertEqual(fractional.array.dtype.kind, 'f')
        self.assertTrue(
            np.any(np.abs(fractional.array - np.rint(fractional.array)) > 1e-3)
        )


if __name__ == "__main__":
    unittest.main()
