"""Notebook validation contracts that do not require optional Jupyter packages."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('execute_demo', ROOT / 'scripts' / 'execute_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
conf_spec = importlib.util.spec_from_file_location('docs_conf', ROOT / 'docs' / 'conf.py')
docs_conf = importlib.util.module_from_spec(conf_spec)
conf_spec.loader.exec_module(docs_conf)


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


class LinkValidationTests(unittest.TestCase):
    def test_repo_file_links_check_the_exact_file_on_the_content_host(self):
        for repository in ('4Denoise', '4denoise'):
            uri = f'https://github.com/mirelesadan/{repository}/blob/main/missing_file.py'
            self.assertEqual(
                docs_conf._github_file_check_uri(None, uri),
                'https://raw.githubusercontent.com/mirelesadan/4Denoise/main/missing_file.py',
            )

    def test_other_destinations_queries_and_fragments_are_unchanged(self):
        base = 'https://github.com/mirelesadan/4Denoise/blob/main/gui.m'
        for uri in (base + '#L10', base + '?plain=1',
                    'https://github.com/another/repo/blob/main/gui.m',
                    'https://zenodo.org/records/17246822',
                    'https://github.com/mirelesadan/4Denoise/pull/9'):
            self.assertIsNone(docs_conf._github_file_check_uri(None, uri))

    def test_validation_hooks_are_registered(self):
        app = Mock()
        docs_conf.setup(app)
        app.connect.assert_any_call('linkcheck-process-uri', docs_conf._github_file_check_uri)
        app.connect.assert_any_call('build-finished', docs_conf._require_verified_external_links)


if __name__ == '__main__':
    unittest.main()
