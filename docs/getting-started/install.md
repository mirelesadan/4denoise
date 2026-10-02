# Install 4Denoise

The documented Conda environment uses Python 3.11. An editable install is
recommended while the package is being developed; publishing to PyPI is not
required. Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate 4denoise-main
python -m pip install -e .
python -c "from fourdenoise import HyperData; print(HyperData.__name__)"
```

If the environment already exists, use
`conda env update -f environment.yml --prune` instead of creating it again.
The Python distribution is named `fourdenoise-main`, while the import is
`fourdenoise`.

## Optional Tools

In a pip-only Python 3.10+ environment, the package metadata provides
optional dependencies:

```bash
python -m pip install -e ".[notebook]"    # Jupyter and notebook analysis tools
python -m pip install -e ".[simulation]"  # abTEM/ASE simulation tools
python -m pip install -e ".[bm]"          # BM3D and BM4D denoisers
```

The Conda environment already includes Jupyter and most notebook tools, but
the complete simulation extra is separate. Install only the extras your
workflow needs. You can check the active interpreter inside Jupyter with:

```python
import sys
print(sys.executable)
```

If this path is not from the intended environment, select the correct
Jupyter kernel and restart it before importing `fourdenoise`.

## Build This Guide

Install the documentation extra, then build HTML locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The build imports the core
`fourdenoise` module to read docstrings; it does not run the large
experimental notebook. The public site is rebuilt from the validated `main`
branch, so it describes the development version rather than a versioned
PyPI release.

Continue with the [mini-data quickstart](quickstart.md).
