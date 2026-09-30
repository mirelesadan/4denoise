"""Pure simulation helper contracts without requiring an abTEM installation."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        '_abtem_helpers_contract_test', ROOT / 'abTEM_helpers.py',
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'abtem': types.ModuleType('abtem')}):
        spec.loader.exec_module(module)
    return module


class _Measurement:
    def __init__(self, array, is_lazy=False, computed=None):
        self.array = array
        self.is_lazy = is_lazy
        self._computed = computed

    def compute(self):
        return self._computed


class _RecordingRng:
    def __init__(self):
        self.expected = None

    def poisson(self, expected):
        self.expected = np.array(expected, copy=True)
        return np.zeros(expected.shape, dtype=np.int64)


class AbTEMHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_helpers()

    def test_conversion_without_resize_and_lazy_compute(self):
        expected = np.arange(16).reshape(4, 4)
        lazy = _Measurement(None, is_lazy=True, computed=_Measurement(expected))
        np.testing.assert_array_equal(self.helpers.abTEM2numpy(lazy), expected)
        np.testing.assert_array_equal(
            self.helpers.abTEM2numpy(_Measurement(expected)), expected,
        )

    def test_resize_preserves_stack_axes_and_fractional_counts(self):
        source = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
        resized = self.helpers.abTEM2numpy(
            _Measurement(source), resize_dims=(2, 3),
        )
        self.assertEqual(resized.shape, (2, 3, 3, 2))
        self.assertTrue(np.issubdtype(resized.dtype, np.floating))
        np.testing.assert_allclose(resized[0, 0], self.helpers.cv2.resize(
            source[0, 0].astype(np.float32), (2, 3),
            interpolation=self.helpers.cv2.INTER_LINEAR,
        ))

    def test_resize_rejects_bad_dimensions(self):
        for dims in ((0, 3), (2.5, 3), True):
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    self.helpers.abTEM2numpy(_Measurement(np.ones((4, 4))), dims)

    def test_poisson_helper_returns_array_with_total_count_scale(self):
        values = np.array([[-2.0, 1.0], [3.0, 0.0]])
        original = values.copy()
        rng = _RecordingRng()
        noisy = self.helpers.add_poisson_noise(values, 40, rng=rng)
        self.assertEqual(noisy.shape, values.shape)
        np.testing.assert_allclose(rng.expected, [[0, 10], [30, 0]])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(noisy, 0)
        np.testing.assert_array_equal(
            self.helpers.add_poisson_noise(np.zeros((2, 2)), 10),
            np.zeros((2, 2)),
        )

    def test_mask_geometry(self):
        mask = self.helpers.make_mask((2, 2), 1, mask_dim=(5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(mask.sum(), 5)
        self.assertEqual(mask[2, 2], 1)
        self.assertEqual(mask[0, 0], 0)


if __name__ == '__main__':
    unittest.main()
