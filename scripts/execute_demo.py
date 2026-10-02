"""Execute only the included mini-data DEMO, never the research notebooks."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEMO = 'DEMO_exp_4dstem_ripple_processing.ipynb'


def clear_execution_state(notebook):
    """Discard cached results so they cannot masquerade as a successful run."""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            cell.setdefault('metadata', {}).pop('execution', None)


def validate_execution(notebook):
    """Require every nonempty code cell to execute without an error output."""
    executed = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code' or not cell['source'].strip():
            continue
        if cell.get('execution_count') is None:
            raise RuntimeError(f'DEMO code cell {index} was not executed.')
        if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
            raise RuntimeError(f'DEMO code cell {index} contains an error.')
        executed += 1
    if executed == 0:
        raise RuntimeError('The DEMO has no executable code cells.')
    return executed


def resolve_output(root, output=None):
    """Resolve the report location and protect the source notebook from overwrite."""
    root = Path(root).resolve()
    path = Path(output) if output is not None else root / 'docs' / '_build' / 'demo' / DEMO
    path = path.resolve()
    if path == (root / DEMO).resolve():
        raise ValueError('The executed report must not overwrite the source DEMO.')
    if path.suffix != '.ipynb':
        raise ValueError('The executed report must use the .ipynb extension.')
    return path


def execute_demo(root=ROOT, *, output=None, timeout=120):
    """Run the full mini-data tutorial with this Python and save a separate report.

    Errors and per-cell timeouts are fatal. The source file is never written;
    the report retains partial output if execution fails. Kernel selection is
    independent of user-installed Jupyter kernel names or notebook metadata.
    """
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager, NoSuchKernel
    from nbclient import NotebookClient

    root = Path(root).resolve()
    source = root / DEMO
    output = resolve_output(root, output)
    if not source.is_file() or not (root / 'mini_dataset_binned.npy').is_file():
        raise FileNotFoundError('The repository must contain the DEMO and mini_dataset_binned.npy.')
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError('timeout must be a positive integer number of seconds.')
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    clear_execution_state(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)

    class InterpreterKernelSpecs(KernelSpecManager):
        def get_kernel_spec(self, name):
            if name != '4denoise-demo':
                raise NoSuchKernel(name)
            return KernelSpec(
                argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                display_name='4Denoise validation', language='python',
            )

    manager = KernelManager(
        kernel_name='4denoise-demo', kernel_spec_manager=InterpreterKernelSpecs(),
    )
    client = NotebookClient(
        notebook, km=manager, resources={'metadata': {'path': str(root)}},
        timeout=timeout, startup_timeout=60, allow_errors=False, force_raise_errors=True,
        record_timing=True,
    )
    print(f'Executing {source.name} with {sys.executable}', flush=True)
    try:
        client.execute(cleanup_kc=True)
        count = validate_execution(notebook)
    finally:
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        finally:
            nbformat.write(notebook, output)
            print(f'Executed notebook report: {output}', flush=True)
    print(f'Passed: all {count} nonempty code cells executed.', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Separate executed .ipynb report path.')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per cell in seconds (default: 120).')
    args = parser.parse_args()
    execute_demo(output=args.output, timeout=args.timeout)


if __name__ == '__main__':
    main()
