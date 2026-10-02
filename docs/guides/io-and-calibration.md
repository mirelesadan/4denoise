# I/O and Calibration

Load an array or a supported file path with `HyperData(...)`. Generic file
loading preserves stored values by default: clipping below one and repairing
NaNs are explicit `clip_on_load` and `repair_nans` choices.

## Raw Detector Data

For a binary file with known layout, provide the full stored shape and dtype:

```python
from fourdenoise import HyperData

data = HyperData(
    "scan.raw", raw_shape=(Ry, Rx, Ky, Kx),
    raw_dtype=">u2", raw_order="C",
)
```

Here `>u2` means big-endian unsigned 16-bit data; use the dtype of your
actual detector file. With an explicit `raw_shape`, detector rows are not
trimmed unless you set `raw_trim_meta=True` and `raw_trim_dims=(Ky, Kx)`.
The [case-study scan](../case-studies/3d-strain-mapping.md) uses float32
data with two stored metadata rows. Rectangular scans need an explicit
`raw_shape`; otherwise the legacy float32 EMPAD reader infers a square scan.

## HDF5 and Large Files

`HyperData(path)` loads a selected dataset eagerly. When a generic HDF5
file has multiple datasets, pass `hdf5_dataset="/entry/data"`. For 3D/4D
HDF5 files too large to load at once, read bounded scan blocks:

```python
with HyperData.open_hdf5("experiment.h5") as source:
    pattern = source.get_dp(0, 0)  # 4D scan position
    for scan_slices, block in source.iter_chunks((16, 16)):
        print(scan_slices, block.shape)
```

Each block is an in-memory `HyperData`; operations on one block do not
automatically process the rest of the source. A 3D stack uses one pattern
index and a scalar chunk size.

## Save and Reopen

```python
data.save("processed.4denoise")
reopened = HyperData("processed.4denoise")
```

The save format retains calibration and supported object metadata. Saving
is atomic by default: on overwrite, the old and new files coexist briefly,
so ensure there is enough free disk space. Use `atomic=False` only if that
extra space is unavailable and a partial destination on failure is
acceptable. `requirements.lock.txt` contains machine-specific `file:///`
paths; use `environment.yml` or the editable install on another computer.
