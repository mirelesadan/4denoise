"""Bounded-memory HDF5-to-polar conversion regression tests."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fourdenoise as fd


class HDF5PolarStreamTests(unittest.TestCase):
    def test_4d_chunks_match_in_memory_transform_and_keep_metadata(self):
        values = np.arange(5 * 6 * 8 * 8, dtype=np.uint16).reshape(5, 6, 8, 8)
        data = fd.HyperData(
            values, real_units='nm', real_conv_factor=(2.0, 3.0),
            real_origin=(10.0, 20.0),
            reciprocal_units='Å^-1', reciprocal_conv_factor=0.1,
        )
        expected = data.to_polar(output_shape=(4, 12), progress=False)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.4denoise'
            target = Path(directory) / 'polar.4denoise'
            data.save(source, atomic=False)
            original_transform = fd.HyperData.to_polar
            observed_chunks = []

            def tracked_transform(block, **kwargs):
                observed_chunks.append(block.scan_shape)
                return original_transform(block, **kwargs)

            with patch.object(fd.HyperData, 'to_polar', tracked_transform):
                result = fd.HyperData.to_polar_hdf5(
                    source, target, chunk_shape=(2, 3),
                    output_shape=(4, 12), progress=False,
                )

            self.assertEqual(result, str(target))
            self.assertGreater(len(observed_chunks), 1)
            self.assertTrue(all(
                chunk[0] <= 2 and chunk[1] <= 3
                for chunk in observed_chunks
            ))
            with h5py.File(target, 'r') as file:
                self.assertEqual(file['array'].shape, (5, 6, 4, 12))
                self.assertEqual(file['array'].chunks, (1, 1, 4, 12))

            loaded = fd.HyperData(target)
            np.testing.assert_allclose(loaded.array, expected.array)
            self.assertTrue(loaded.is_polar)
            self.assertEqual(loaded.real_origin, (10.0, 20.0))
            self.assertEqual(loaded.real_conv_factor, (2.0, 3.0))
            self.assertEqual(loaded.polar_metadata, expected.polar_metadata)
            with fd.HyperData.open_hdf5(target) as reader:
                np.testing.assert_allclose(
                    reader.get_dp(4, 5).array, expected.array[4, 5],
                )

    def test_3d_generic_hdf5_selection(self):
        values = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'generic.h5'
            target = Path(directory) / 'polar.h5'
            with h5py.File(source, 'w') as file:
                file.create_dataset('images', data=values)
                file.create_dataset('other', data=np.ones((2, 3, 4)))

            fd.HyperData.to_polar_hdf5(
                source, target, hdf5_dataset='images',
                chunk_shape=2, output_shape=(4, 8), progress=False,
            )
            expected = fd.HyperData(values).to_polar(
                output_shape=(4, 8), progress=False,
            )
            loaded = fd.HyperData(target)
            self.assertEqual(loaded.shape, (3, 4, 8))
            np.testing.assert_allclose(loaded.array, expected.array)

    def test_atomic_failure_preserves_existing_output(self):
        values = np.ones((2, 2, 8, 8), dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.h5'
            target = Path(directory) / 'target.h5'
            fd.HyperData(values).save(source, atomic=False)
            fd.HyperData(np.full((2, 2, 8, 8), 7)).save(target, atomic=False)

            with self.assertRaisesRegex(FileExistsError, 'exists'):
                fd.HyperData.to_polar_hdf5(
                    source, target, progress=False,
                )
            with self.assertRaisesRegex(ValueError, 'center must lie'):
                fd.HyperData.to_polar_hdf5(
                    source, target, center=(100, 100),
                    overwrite=True, progress=False,
                )
            np.testing.assert_array_equal(fd.HyperData(target).array, 7)
            with self.assertRaisesRegex(ValueError, 'different files'):
                fd.HyperData.to_polar_hdf5(
                    source, source, overwrite=True, progress=False,
                )


if __name__ == '__main__':
    unittest.main()
