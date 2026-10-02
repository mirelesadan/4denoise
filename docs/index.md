# 4Denoise Guide

4Denoise provides loading, visualization, denoising, peak analysis, and strain
mapping for 4D-STEM data. Start with [installation](getting-started/install.md)
and the [quickstart](getting-started/quickstart.md), which uses the included
mini dataset. The full [DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb)
was executed in a clean kernel with the current package.

## Concepts and Tasks

- [Data model](concepts/data-model.md): axes, scan geometry, and calibration.
- [I/O and calibration](guides/io-and-calibration.md): raw and HDF5 data, bounded-memory reading, and saving.
- [Visualization](guides/visualization.md): diffraction patterns, virtual detectors, and calibrated axes.
- [Denoising and unfolding](guides/denoising-and-unfolding.md): what happens to excluded data during a traversal.
- [Peaks and strain](guides/peaks-and-strain.md): interpreting strain maps and fit quality.

The [API reference](reference/hyperdata.rst) lists selected public entry points
and draws their signatures and descriptions from the source docstrings.

## Research Workflow

The [3D strain-mapping case study](case-studies/3d-strain-mapping.md) links the
full experimental and simulation notebooks and describes the external data,
compute requirements, and MATLAB GUI. It is not a routine installation test.

The guide can be built locally with Sphinx. It is not published yet.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/install
getting-started/quickstart
concepts/data-model
```

```{toctree}
:maxdepth: 2
:caption: Tasks

guides/io-and-calibration
guides/visualization
guides/denoising-and-unfolding
guides/peaks-and-strain
```

```{toctree}
:maxdepth: 2
:caption: Reference and research

reference/hyperdata
reference/views-and-results
reference/utilities
case-studies/3d-strain-mapping
troubleshooting
```
