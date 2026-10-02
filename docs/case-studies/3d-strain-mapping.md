# 3D Strain-Mapping Case Study

This study-specific workflow accompanies *Strain Mapping of
Three-dimensionally Structured Two-dimensional Materials*. It is separate
from the package's [mini-data DEMO](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb).
The external experimental scan and simulated ripple dataset are described in
the [Zenodo data record](https://zenodo.org/records/17246822) (DOI:
[10.5281/zenodo.17246822](https://doi.org/10.5281/zenodo.17246822)).

## Data and Software

- Experimental data: `scan_x256_y256.raw`, approximately 4.4 GB. The notebook
  loads the stored float32 shape `(256, 256, 130, 128)` and trims two EMPAD
  metadata rows to obtain `(256, 256, 128, 128)` diffraction data. It uses a
  real-space calibration of 5.152 nm per scan pixel.
- Simulated data: `simulated_4d_dataset_highRes.npy` is external to the
  repository. Its generation and analysis are in the
  [simulation notebook](https://github.com/mirelesadan/4Denoise/blob/main/generateSimulatedRipple_4Ddata.ipynb).
- Python: [install the package](../getting-started/install.md).
  The full simulation workflow also requires the `simulation` optional extra;
  the notebooks require Jupyter.

The [experimental notebook](https://github.com/mirelesadan/4Denoise/blob/main/exp_MoS2_MoSe2_processing.ipynb) first
loads and crops the scan, aligns diffraction patterns, corrects elliptical
distortion, detects and refines Bragg peaks, and integrates their intensities.
It then generates or loads a kinematic tilt library, estimates tilt and
height, fits strain, and applies a tilt-aware correction. These are
experiment-specific choices, not defaults recommended for every 4D-STEM scan.

Edit the notebook's `filepath` cell to point to your local
`scan_x256_y256.raw`. The published experimental notebook contains
machine-specific paths; review its input and export paths before running it.
Unlike the mini-data DEMO, it has not yet been migrated and verified against
the current package end-to-end.

The full workflow is compute- and memory-intensive. `HyperData(filepath)`
loads this raw scan eagerly; peak refinement, library generation, and
reconstruction can take substantially longer than the DEMO. Review individual
save/export cells and their filenames before executing them; they can write
large intermediates. Use a separate output directory with sufficient disk
space, and do not overwrite earlier scientific results.

**Verification boundary:** loading, dose estimation, virtual images, and the
initial crop were smoke-tested on the real scan in a locally modernized
version of this notebook. That version is not part of this documentation
update. Historical outputs in the published notebook are not evidence of a
fresh run against the current API. The later alignment, simulation,
reconstruction, and strain stages have not been rerun end-to-end for this
documentation pass. CI executes only the separate mini-data DEMO.

## MATLAB Viewer

The optional [BRIGHT GUI](https://github.com/mirelesadan/4Denoise/blob/main/gui.m) visualizes height, tilt, strain, and
overlays. The historical workflow specifies MATLAB R2021a or newer; MATLAB is
not required to use the core Python package. The script loads ten `.mat` files
by relative filename from the current MATLAB working directory:

![BRIGHT MATLAB strain and topography viewer](https://github.com/user-attachments/assets/52c2c52f-20d8-429f-a906-590bdc09674a)

```text
height_map.mat
haadf.mat
tilts.mat
exx_rippleData.mat
eyy_rippleData.mat
exy_rippleData.mat
erot_rippleData.mat
exx_rippleData_corrected.mat
eyy_rippleData_corrected.mat
exy_rippleData_corrected.mat
```

To produce these files, complete and verify the experimental workflow and
review its MATLAB export cells. Put the ten files in one output directory,
make that the MATLAB working directory, and add the repository directory to
the MATLAB path before running `gui`. The viewer can switch strain components, compare corrected and
uncorrected maps, rotate the 3D surface, and select HAADF, height, or phase
overlays. It also provides a strain-basis rotation slider and an interpolation
toggle. The GUI is study-specific and is not exercised by Python CI.
