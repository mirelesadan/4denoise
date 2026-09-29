"""Selection and error handling for standard and HDF5 MATLAB files."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourdenoise as fd


class MatLoadingTests(unittest.TestCase):
    def test_standard_mat_auto_selects_only_numeric_array(self):
        original = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'single.mat'
            io.savemat(path, {'comment': 'scan', 'data': original})
            loaded = fd.HyperData(path)
        np.testing.assert_array_equal(loaded.array, original)
        self.assertEqual(loaded.dtype, original.dtype)

    def test_standard_mat_requires_selection_when_ambiguous(self):
        first = np.zeros((2, 3), dtype=np.float64)
        second = np.ones((3, 4, 5), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'multiple.mat'
            io.savemat(path, {'first': first, 'second': second})
            with self.assertRaisesRegex(ValueError, "mat_variable='name'") as error:
                fd.HyperData(path)
            self.assertIn('first: shape=(2, 3)', str(error.exception))
            self.assertIn('second: shape=(3, 4, 5)', str(error.exception))
            loaded = fd.HyperData(path, mat_variable='second')
        np.testing.assert_array_equal(loaded.array, second)

    def test_standard_mat_reports_missing_and_non_numeric_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'types.mat'
            io.savemat(path, {'label': 'hello', 'data': np.ones((2, 2))})
            with self.assertRaisesRegex(KeyError, 'Available variables'):
                fd.HyperData(path, mat_variable='missing')
            with self.assertRaisesRegex(TypeError, 'non-numeric class'):
                fd.HyperData(path, mat_variable='label')

    def test_standard_mat_without_numeric_arrays_has_informative_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.mat'
            io.savemat(path, {'label': 'hello'})
            with self.assertRaisesRegex(ValueError, 'No numeric array') as error:
                fd.HyperData(path)
            self.assertIn('label', str(error.exception))

    def test_hdf5_mat_ignores_internal_references(self):
        original = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v73.mat'
            with h5py.File(path, 'w') as file:
                file.create_dataset('data', data=original)
                file.create_group('#refs#').create_dataset(
                    'internal', data=np.ones((2, 2))
                )
            loaded = fd.HyperData(path)
        np.testing.assert_array_equal(loaded.array, original)

    def test_hdf5_mat_lists_nested_candidates_and_selects_by_path(self):
        original = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested.mat'
            with h5py.File(path, 'w') as file:
                group = file.create_group('scan')
                group.create_dataset('first', data=np.ones((2, 2)))
                group.create_dataset('second', data=original)
            with self.assertRaisesRegex(ValueError, "mat_variable='/path/to/dataset'") as error:
                fd.HyperData(path)
            self.assertIn('/scan/first', str(error.exception))
            self.assertIn('/scan/second', str(error.exception))
            loaded = fd.HyperData(path, mat_variable='scan/second')
            np.testing.assert_array_equal(loaded.array, original)
            with self.assertRaisesRegex(ValueError, 'group, not a dataset'):
                fd.HyperData(path, mat_variable='scan')

    def test_mat_selection_argument_is_mat_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.npy'
            np.save(path, np.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, 'only to .mat'):
                fd.HyperData(path, mat_variable='data')

    def test_invalid_mat_file_does_not_return_none_or_file_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.mat'
            path.write_bytes(b'not a MAT file')
            with self.assertRaisesRegex(ValueError, 'MATLAB MAT file'):
                fd.HyperData(path)


if __name__ == '__main__':
    unittest.main()
