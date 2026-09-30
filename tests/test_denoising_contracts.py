"""Checks for the public denoising-method guide and its registry."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import fourdenoise as fd


class DenoisingContractTests(unittest.TestCase):
    def setUp(self):
        self.data = fd.HyperData(np.zeros((2, 3, 4, 5), dtype=np.float32))

    def test_every_available_method_has_a_contract(self):
        names = set(self.data.available_denoising_methods)
        self.assertEqual(names, set(fd._DENOISING_METHOD_CONTRACTS))

        overview = self.data.denoising_method_info(
            include_doc=False, print_info=False
        )
        self.assertEqual(names, set(overview['method_contracts']))
        for name, contract in overview['method_contracts'].items():
            with self.subTest(method=name):
                self.assertTrue(contract['supported_input_ndim'])
                self.assertTrue(contract['input_layout'])
                self.assertTrue(contract['four_dimensional_routes'])
                self.assertIn('HyperData', contract['default_output'])

    def test_three_dimensional_methods_require_unfolding_for_4d(self):
        for name in ('bm4d', 'parafac2'):
            with self.subTest(method=name):
                info = self.data.denoising_method_info(
                    name, include_doc=False, print_info=False
                )
                self.assertEqual(info['supported_input_ndim'], (3,))
                self.assertEqual(len(info['four_dimensional_routes']), 1)
                self.assertIn(
                    "unfold_domain='real'", info['four_dimensional_routes'][0]
                )

    def test_matrix_and_tensorized_matrix_routes_are_distinct(self):
        nmf = self.data.denoising_method_info(
            'nmf', include_doc=False, print_info=False
        )
        self.assertEqual(nmf['supported_input_ndim'], (2,))
        self.assertIn('nonnegative', nmf['input_layout'])
        self.assertTrue(any('both' in route for route in nmf['four_dimensional_routes']))

        tt_matrix = self.data.denoising_method_info(
            'tensor_train_matrix', include_doc=False, print_info=False
        )
        self.assertEqual(tt_matrix['supported_input_ndim'], (2, 4))
        self.assertIn('even', tt_matrix['constraints'])
        self.assertTrue(any('domain=None' in route for route in tt_matrix['four_dimensional_routes']))
        self.assertFalse(any('3D' in route for route in tt_matrix['four_dimensional_routes']))

    def test_median_special_routing_and_optional_result_flags(self):
        median = self.data.denoising_method_info(
            'median', include_doc=False, print_info=False
        )
        self.assertIn('axes=None filters every axis', median['constraints'])
        self.assertIn('selected axis pair', median['four_dimensional_routes'][0])

        parafac = self.data.denoising_method_info(
            'parafac', include_doc=False, print_info=False
        )
        self.assertEqual(
            parafac['optional_result_flags'],
            ('return_decomposition', 'return_errors'),
        )
        self.assertIn('method-specific', parafac['output_caveat'])
        self.assertTrue(any('domain=None' in route for route in parafac['four_dimensional_routes']))

    def test_printed_guide_includes_contract(self):
        output = StringIO()
        with redirect_stdout(output):
            self.data.denoising_method_info('bm4d', print_info=True)
        printed = output.getvalue()
        self.assertIn('Input: single 3D volume', printed)
        self.assertIn('For 4D data:', printed)
        self.assertIn('unfold_domain', printed)


if __name__ == '__main__':
    unittest.main()
