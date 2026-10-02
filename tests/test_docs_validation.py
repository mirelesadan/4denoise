"""Notebook validation contracts that do not require optional Jupyter packages."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('execute_demo', ROOT / 'scripts' / 'execute_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoValidationTests(unittest.TestCase):
    def test_cached_outputs_and_execution_metadata_are_cleared(self):
        notebook = {'cells': [
            {'cell_type': 'markdown', 'source': 'Instructions', 'metadata': {}},
            {'cell_type': 'code', 'source': 'print(1)', 'execution_count': 5,
             'outputs': [{'output_type': 'stream', 'text': 'old output'}],
             'metadata': {'execution': {'old': True}, 'tags': ['example']}},
        ]}
        demo.clear_execution_state(notebook)
        self.assertEqual(notebook['cells'][1]['outputs'], [])
        self.assertIsNone(notebook['cells'][1]['execution_count'])
        self.assertEqual(notebook['cells'][1]['metadata'], {'tags': ['example']})
        self.assertNotIn('outputs', notebook['cells'][0])

    def test_skipped_nonempty_cells_are_rejected(self):
        cell = {'cell_type': 'code', 'source': 'print(1)', 'execution_count': None, 'outputs': []}
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            demo.validate_execution({'cells': [cell]})

    def test_error_outputs_are_rejected(self):
        cell = {'cell_type': 'code', 'source': '1 / 0', 'execution_count': 1,
                'outputs': [{'output_type': 'error', 'ename': 'ZeroDivisionError'}]}
        with self.assertRaisesRegex(RuntimeError, 'contains an error'):
            demo.validate_execution({'cells': [cell]})

    def test_valid_execution_ignores_markdown_and_empty_cells(self):
        notebook = {'cells': [
            {'cell_type': 'markdown', 'source': 'Instructions'},
            {'cell_type': 'code', 'source': '  ', 'execution_count': None},
            {'cell_type': 'code', 'source': 'print(1)', 'execution_count': 1, 'outputs': []},
        ]}
        self.assertEqual(demo.validate_execution(notebook), 1)
        with self.assertRaisesRegex(RuntimeError, 'no executable'):
            demo.validate_execution({'cells': notebook['cells'][:2]})

    def test_output_is_separate_from_source(self):
        output = demo.resolve_output(ROOT)
        self.assertEqual(output, ROOT / 'docs' / '_build' / 'demo' / demo.DEMO)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            demo.resolve_output(ROOT, ROOT / demo.DEMO)
        with self.assertRaisesRegex(ValueError, 'extension'):
            demo.resolve_output(ROOT, ROOT / 'report.txt')


if __name__ == '__main__':
    unittest.main()
