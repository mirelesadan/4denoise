# Troubleshooting

## A Different Package Version Imports

The active working directory does not choose the Python environment. In the
terminal and notebook, compare `sys.executable` and inspect the imported
module path:

```python
import sys
import fourdenoise
print(sys.executable)
print(fourdenoise.__file__)
```

Activate the intended environment and run `python -m pip install -e .`
from this repository. Select the matching Jupyter kernel, then restart it
to reload edited source code.

## A Raw File Has the Wrong Shape

Binary `.raw` data does not describe its own dimensions or byte order.
Pass `raw_shape`, `raw_dtype`, and `raw_order` explicitly when known. For
the study's EMPAD scan, use the dimensions shown in the
[case study](case-studies/3d-strain-mapping.md). Generic file loading does
not clip values below one or repair NaNs unless requested.

## An HDF5 File Has Multiple Datasets

Pass `hdf5_dataset="/path/to/array"` to select the intended numeric data.
For a multi-gigabyte file, use `HyperData.open_hdf5(...)` and
`iter_chunks(...)` rather than an eager `HyperData(path)` load. See
[I/O and calibration](guides/io-and-calibration.md).

## An Unfolded Result Changes Shape

Full-shape traversals retain all scan positions. Hilbert, Morton, Peano,
and block-meander methods may crop to a compatible centered domain or
resize first. Check the emitted warning and `unfold_metadata`. With
`preserve_excess=True`, excluded values are retained for exact undo but
were not included in denoising. With `preserve_excess=False`, undo returns
only the cropped geometry. See [denoising and unfolding](guides/denoising-and-unfolding.md).

## A Large Workflow Exhausts Memory

The mini-data DEMO is the routine smoke test. The experimental scan is
multi-gigabyte and eager loading requires substantially more RAM than the
raw file size once intermediates are allocated. Use bounded HDF5 scan
chunks where a method supports them, avoid keeping multiple full outputs,
and review export cells before running the experimental notebook. Saving
intermediates uses disk space and does not free live arrays from RAM. Not
every `HyperData` method is lazy.
