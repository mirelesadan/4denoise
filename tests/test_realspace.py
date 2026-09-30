"""Regression tests for calibrated real-space images and HyperData integration."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd

if Path(fd.__file__).resolve() != REPOSITORY_ROOT / 'fourdenoise.py':
    raise ImportError(f"Regression test imported the wrong fourdenoise.py: {fd.__file__}")


class RealSpaceTests(unittest.TestCase):
    def setUp(self):
        self.image = fd.RealSpace(
            np.arange(24).reshape(4, 6),
            units='nm', conv_factor=(2.0, 3.0), origin=(10.0, 20.0),
            quantity='Strain', value_units='%',
        )
        self.addCleanup(plt.close, 'all')

    def test_constructor_scale_validation_and_copy(self):
        self.assertEqual(self.image.pixel_size, (2.0, 3.0))
        with self.assertRaisesRegex(ValueError, '2D'):
            fd.RealSpace(np.ones((2, 3, 4)))
        for factor in (float('nan'), float('inf'), 0.0, (1.0, 0.0)):
            with self.assertRaises(ValueError):
                fd.RealSpace(np.ones((2, 2)), units='nm', conv_factor=factor)
        with self.assertRaisesRegex(TypeError, 'real-valued'):
            fd.RealSpace(np.ones((2, 2), dtype=complex))

        copied = self.image.copy()
        self.assertEqual(copied.origin, (10.0, 20.0))
        self.assertEqual(copied.quantity, 'Strain')
        copied.array[0, 0] = -1
        self.assertEqual(self.image.array[0, 0], 0)
        before = (self.image.units, self.image.conv_factor)
        with self.assertRaises(ValueError):
            self.image.set_scale('um', float('nan'))
        self.assertEqual((self.image.units, self.image.conv_factor), before)

    def test_flip_preserves_physical_scan_coordinates(self):
        source = np.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
        flipped = fd.HyperData(
            source, real_units='nm', real_conv_factor=(2.0, 3.0),
            real_origin=(10.0, 20.0), flip_axis=(0, 1),
        )
        self.assertEqual(flipped.real_origin, (14.0, 29.0))
        self.assertEqual(flipped.real_conv_factor, (-2.0, -3.0))
        np.testing.assert_array_equal(
            flipped.get_dp(14, 29, selection_units='calibrated').array,
            source[-1, -1],
        )
        cropped = flipped.crop(
            ylim=(10.0, 14.0), xlim=(23.0, 29.0),
            real_limit_units='calibrated',
        )
        np.testing.assert_array_equal(cropped.array, flipped.array[1:3, 1:3])
        self.assertEqual(cropped.real_origin, (12.0, 26.0))
        restored = fd.HyperData(
            flipped.array, real_units=flipped.real_units,
            real_conv_factor=flipped.real_conv_factor,
            real_origin=flipped.real_origin, flip_axis=(0, 1),
        )
        np.testing.assert_array_equal(restored.array, source)
        self.assertEqual(restored.real_origin, (10.0, 20.0))
        self.assertEqual(restored.real_conv_factor, (2.0, 3.0))

    def test_signed_realspace_crop_and_scale_bar(self):
        image = fd.RealSpace(
            np.arange(12).reshape(3, 4), units='nm',
            conv_factor=(-2.0, -3.0), origin=(14.0, 29.0),
        )
        cropped = image.crop(
            (10.0, 14.0), (23.0, 29.0),
            selection_units='calibrated',
        )
        np.testing.assert_array_equal(cropped.array, image.array[1:3, 1:3])
        self.assertEqual(cropped.origin, (12.0, 26.0))
        _, ax = image.show(show=False, scale_bar=3.0)
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(tuple(ax.images[0].get_extent()), (30.5, 18.5, 9.0, 15.0))
        self.assertGreater(ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[1])

    def test_show_reuses_axes_and_maps_pixel_overlay_to_calibrated_axes(self):
        fig, ax = plt.subplots()
        with patch.object(fd.plt, 'show') as shown:
            returned_fig, returned_ax = self.image.show(
                ax=ax, show=False, coords=(1, 2), c='red',
                axis_units='calibrated', grid_ticks=(3, 4),
                scale_bar=6, scale_bar_color='white',
            )
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        shown.assert_not_called()
        np.testing.assert_allclose(
            ax.collections[0].get_offsets(), [[26.0, 12.0]],
        )
        np.testing.assert_allclose(
            ax.images[0].get_extent(), (18.5, 36.5, 17.0, 9.0),
        )
        self.assertEqual(len(ax.get_xticks()), 4)
        self.assertEqual(len(ax.get_yticks()), 3)
        self.assertEqual(fig.axes[1].get_ylabel(), 'Strain (%)')
        self.assertTrue(any(text.get_text() == '6 nm' for text in ax.texts))

    def test_calibrated_overlay_on_pixel_axes_and_axis_free_options(self):
        fig, ax = self.image.show(
            show=False, axes=False, colorbar=False, grid=True,
            coords_units='calibrated', coords=[[12, 26]], axis_units='pixels',
            scale_bar=2, scale_bar_label=False, title=None,
        )
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[2.0, 1.0]])
        self.assertEqual(len(fig.axes), 1)
        self.assertFalse(ax.axison)
        self.assertGreater(len(ax.lines), 1)
        self.assertEqual(ax.get_title(), '')

    def test_percentiles_symmetric_limits_and_coordinate_validation(self):
        image = fd.RealSpace(np.array([[-10, -1], [1, 20]], dtype=float))
        fig, ax = image.show(
            show=False, percentiles=(25, 75), symmetric=True,
        )
        low, high = ax.images[0].get_clim()
        self.assertEqual(low, -high)
        self.assertGreater(high, 0)
        with self.assertRaisesRegex(ValueError, 'shape'):
            image.show(show=False, coords=[[1, 2, 3]])
        with self.assertRaisesRegex(TypeError, 'requires coords'):
            image.show(show=False, c='red')
        with self.assertRaisesRegex(ValueError, 'requires image calibration'):
            image.show(show=False, coords=(1, 1), coords_units='calibrated')
        with self.assertRaisesRegex(ValueError, 'too long'):
            self.image.show(show=False, scale_bar=1000)
        self.image.show(show=False, num_div=None, colorbar=False)

    def test_crop_resize_and_metadata(self):
        cropped = self.image.crop((1, 4), (2, 6))
        self.assertEqual(cropped.shape, (3, 4))
        self.assertEqual(cropped.origin, (12.0, 26.0))
        np.testing.assert_array_equal(cropped.array, self.image.array[1:4, 2:6])

        selected = self.image.crop((12, 18), (26, 38), selection_units='calibrated')
        np.testing.assert_array_equal(selected.array, cropped.array)
        resized = cropped.resize((2, 2), method='nearest')
        self.assertEqual(resized.shape, (2, 2))
        self.assertEqual(resized.pixel_size, (3.0, 6.0))
        self.assertEqual(resized.origin, (12.5, 27.5))
        self.assertEqual(resized.quantity, 'Strain')
        area = cropped.resize((1, 2), method='area')
        self.assertEqual(area.shape, (1, 2))
        with self.assertRaisesRegex(ValueError, 'downsampling only'):
            self.image.resize((8, 12), method='area')

    def test_hyperdata_origin_anisotropy_and_virtual_image(self):
        source = np.arange(4 * 6 * 3 * 3).reshape(4, 6, 3, 3)
        data = fd.HyperData(
            source, real_units='nm', real_conv_factor=(2.0, 3.0),
            real_origin=(10.0, 20.0),
        )
        with self.assertWarnsRegex(RuntimeWarning, 'scalar-only'):
            swapped = data.swap_domains()
        self.assertIsNone(swapped.reciprocal_conv_factor)
        self.assertEqual(swapped.shape, (3, 3, 4, 6))
        cropped = data.crop(
            ylim=(1, 4), xlim=(2, 6), real_limit_units='pixels',
        )
        self.assertEqual(cropped.real_origin, (12.0, 26.0))
        selected = cropped.get_dp(
            12.0, 26.0, selection_units='calibrated',
        )
        np.testing.assert_array_equal(selected.array, source[1, 2])

        resized = cropped.resize((2, 2), domain='real', method='nearest')
        self.assertEqual(resized.real_conv_factor, (3.0, 6.0))
        self.assertEqual(resized.real_origin, (12.5, 27.5))
        with patch.object(fd.plt, 'show') as shown:
            image = resized.virtual_image(
                mask=np.ones((3, 3), dtype=bool), show=False,
            )
        shown.assert_not_called()
        self.assertIsInstance(image, fd.RealSpace)
        self.assertEqual(image.origin, resized.real_origin)
        self.assertEqual(image.pixel_size, (3.0, 6.0))
        self.assertEqual(image.quantity, 'Virtual detector signal')
        self.assertEqual(data.get_stdDev(domain='real').quantity, 'Standard deviation')
        self.assertEqual(data.copy().real_origin, (10.0, 20.0))

        crop_and_resize = data.crop(
            ylim=(1, 4), xlim=(2, 6), rshape=(2, 2),
            real_limit_units='pixels',
        )
        self.assertEqual(crop_and_resize.real_conv_factor, (3.0, 6.0))
        self.assertEqual(crop_and_resize.real_origin, (12.5, 27.5))

        cropped_again = data.crop(
            ylim=(12, 18), xlim=(26, 38), real_limit_units='calibrated',
        )
        np.testing.assert_array_equal(cropped_again.array, source[1:4, 2:6])

    def test_hyperdata_save_load_preserves_real_origin_and_spacing(self):
        data = fd.HyperData(
            np.ones((2, 3, 2, 2)), real_units='nm',
            real_conv_factor=(2.0, 3.0), real_origin=(10.0, 20.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'image.4denoise'
            data.save(filename)
            loaded = fd.HyperData(filename)
        self.assertEqual(loaded.real_conv_factor, (2.0, 3.0))
        self.assertEqual(loaded.real_origin, (10.0, 20.0))


if __name__ == '__main__':
    unittest.main()
