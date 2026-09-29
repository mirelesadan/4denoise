"""Regression tests for replacing the public HyperData.array."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class ArrayReplacementTests(unittest.TestCase):
    def test_same_shape_replacement_updates_dtype_and_denoising_engine(self):
        original = np.ones((3, 3), dtype=np.uint16)
        data = fd.HyperData(original)
        replacement = np.arange(9, dtype=np.float32).reshape(3, 3)

        data.array = replacement

        self.assertIs(data.array, replacement)
        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(data.dtype, np.dtype('float32'))
        self.assertIs(data._denoise_engine.array, replacement)
        self.assertEqual(data._denoise_engine.ndim, 2)
        np.testing.assert_array_equal(
            data.denoise(method='median', window_size=1).array,
            replacement,
        )

    def test_shape_replacement_refreshes_all_geometry_fields(self):
        data = fd.HyperData(np.zeros((2, 3, 4, 5), dtype=np.uint16))
        replacement = np.ones((6, 7, 8), dtype=np.float32)

        data.array = replacement

        self.assertEqual(data.ndim, 3)
        self.assertEqual(data.shape, (6, 7, 8))
        self.assertEqual(data.scan_shape, (6,))
        self.assertEqual(data.pattern_shape, (7, 8))
        self.assertIsNone(data.real_shape)
        self.assertEqual(data.k_shape, (7, 8))
        self.assertEqual(data.dtype, np.dtype('float32'))
        self.assertIs(data._denoise_engine.array, replacement)
        self.assertEqual(data._denoise_engine.ndim, 3)

    def test_invalid_replacement_keeps_previous_state(self):
        data = fd.HyperData(np.zeros((2, 3)))
        original = data.array
        engine = data._denoise_engine

        with self.assertRaisesRegex(ValueError, 'at least two spatial axes'):
            data.array = np.arange(6)

        self.assertIs(data.array, original)
        self.assertIs(data._denoise_engine, engine)
        self.assertEqual(data.shape, (2, 3))
        self.assertEqual(data.dtype, original.dtype)

    def test_unfold_metadata_survives_values_but_not_shape_changes(self):
        original = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        unfolded = fd.HyperData(original).unfold(
            domain='real', method='serpentine',
        )
        unfolded.array = unfolded.array + 10

        self.assertIsNotNone(unfolded.unfold_metadata)
        np.testing.assert_array_equal(
            unfolded.unfold(undo=True).array,
            original + 10,
        )

        unfolded.array = unfolded.array[:-1]
        self.assertIsNone(unfolded.unfold_metadata)

    def test_beam_and_polar_metadata_follow_pattern_shape(self):
        beam = fd._center_beam_metadata_from_pixels(
            1.0, (2.5, 2.5), (6, 6), source='alignment',
        )
        data = fd.HyperData(
            np.zeros((2, 3, 6, 6)),
            center_beam_metadata=beam,
            polar_metadata={'is_polar': True, 'output_shape': (6, 6)},
        )
        data.array = np.ones((4, 5, 6, 6))

        self.assertEqual(data.center_beam_metadata, beam)
        self.assertIsNotNone(data.polar_metadata)

        data.array = np.ones((4, 5, 4, 4))
        self.assertIsNone(data.center_beam_metadata)
        self.assertIsNone(data.polar_metadata)


if __name__ == '__main__':
    unittest.main()
