import unittest

import fourdenoise
import fourdenoise_geometry as geometry


class GeometryModuleTests(unittest.TestCase):
    def test_helpers_remain_available_from_fourdenoise(self):
        for name in (
            "_normalize_real_spacing",
            "_real_spacing_pair",
            "_scaled_real_spacing",
            "_normalize_real_origin",
            "_parse_real_selection",
            "_normalize_unit_mode",
            "_resolve_unit_mode",
            "_center_to_calibrated",
            "_calibrated_center_to_pixels",
        ):
            self.assertIs(getattr(fourdenoise, name), getattr(geometry, name))

    def test_center_coordinate_round_trip(self):
        center = (2.25, 8.5)
        shape = (12, 14)
        calibrated = geometry._center_to_calibrated(center, shape, 0.2)
        pixels = geometry._calibrated_center_to_pixels(calibrated, 0.2, shape)
        self.assertAlmostEqual(pixels[0], center[0])
        self.assertAlmostEqual(pixels[1], center[1])

    def test_unit_selection_preserves_calibration(self):
        self.assertEqual(
            geometry._resolve_unit_mode("auto", "nm", (0.5, -0.25)),
            ("nm", (0.5, -0.25), "calibrated"),
        )
        self.assertEqual(
            geometry._parse_real_selection((1, 3), 10, "ry", "calibrated", 0.5),
            (2, 6, "range"),
        )


if __name__ == "__main__":
    unittest.main()
