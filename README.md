# 1. About the Project
**4Denoise** is a Python package for advanced processing, visualizing, and denoising 4D-STEM datasets.

This package is used for processing of data in _"**Strain Mapping of Three-dimensionally Structured Two-dimensional Materials**"_
See the Zenodo (https://zenodo.org/records/17246822) record for the MoS<sub>2</sub>–MoSe<sub>2</sub> heterojunction 4D-STEM dataset (`scan_x256_y256.raw`) and the simulated ripple dataset (`simulated_4d_dataset_highRes.npy`)

# 2. Package Contents

## 2.1. 4Denoise — Core Module (`fourdenoise.py`)

The core Python module provides a modular toolkit for loading, cleaning, and analyzing nanobeam 4D-STEM datasets, and for exporting results to the companion GUI.

- **I/O & data model** – Reads common 4D-STEM formats and organizes them with minimal metadata for reproducible processing.
- **Preprocessing** – Basic detector/scan cleanup and normalization to get diffraction data into analysis-ready form.
- **Reciprocal-space features** – Bragg-peak localization, center-of-mass tracking, integrated-intensity extraction, and virtual-aperture imaging.
- **Tilt inference** – Converts intensity vectors at selected reflections into local tilt angles using a kinematic library, producing dense tilt maps.
- **Topography from tilt** – Translates tilt maps into gradients and integrates to a height field with light regularization.
- **Planar strain on non-planar surfaces** – Computes in-plane strain fields and applies tilt-aware corrections using the recovered topography.
- **Visualization** – Quick-look plots.
- **Extensibility** – Designed so denoising, inpainting, or custom models can be swapped in without changing the overall workflow.

## 2.1. Simulation of Ripple-like 4D-STEM Dataset (`generateSimulatedRipple_4Ddata.ipynb`) [Jupyter Notebook]

This Jupyter notebook shows how the simulation data in `simulated_4d_dataset_highRes.npy` was obtained, how Poisson shot noise was added to such data, and how the corresponding 3D reconstructions were obtained. The notebook includes the results shown in _"**Strain Mapping of Three-dimensionally Structured Two-dimensional Materials**"_.

## 2.2. Processing of Experimental Nanobeam 4D-STEM Data (`exp_MoS2_MoSe2_processing.ipynb`) [Jupyter Notebook]

In this notebook, we load the experimental data presented in _"**Strain Mapping of Three-dimensionally Structured Two-dimensional Materials**"_, show methods for visualizing and processing it, and enable 3D reconstruction followed by strain mapping and strain correction using tilt information. The notebook details each step carefully and contains multiple notes to guide the user. 

### 2.2.1. DEMO Notebook

We also add a Jupyter notebook that uses a small 4D dataset `mini_dataset_binned.npy` for testing the reconstructiong algorithm. This dataset is binned (each pixel is ~10.3 nm) and encompasses a whole ripple. All instructions are contained within the Jupyter notebook.

## 2.3. BRIGHT MATLAB GUI: 3D Strain & Topography Viewer (`gui.m`) [MATLAB Script]

<img width="1149" height="548" alt="image" src="https://github.com/user-attachments/assets/52c2c52f-20d8-429f-a906-590bdc09674a" />

### 2.3.1. Overview

`gui.mat` is an interactive MATLAB interface for exploring the outputs of the BRIGHT workflow—namely, 3D topography, tilt-aware planar strain maps, and virtual overlays (e.g., HAADF or tilt phase).
It allows toggling between corrected/uncorrected strain components, rotating the 3D view, and visualizing how tilt correction affects the apparent strain distribution.

### 2.3.2. Requirements

- MATLAB R2021a or newer
- Ten `.mat` input files generated from the Jupyter notebook ([`exp_MoS2_MoSe2_processing`]) in the **4Denoise** repository:
  ```
  height_map.mat
  exx_rippleData.mat
  eyy_rippleData.mat
  exy_rippleData.mat
  erot_rippleData.mat
  exx_rippleData_corrected.mat
  eyy_rippleData_corrected.mat
  exy_rippleData_corrected.mat
  haadf.mat
  tilts.mat
  ```
- The GUI will load the datasets, crop the region of interest, and render the interactive 3D surface.

Controls include:

- Switching between strain components (ε<sub>xx</sub>, ε<sub>yy</sub>, ε<sub>xy</sub>, ε<sub>rot</sub>).
- Toggling tilt-corrected vs uncorrected strain.
- Adjusting vertical stretch and rotation interactively.
- Strain Correction toggle: switches between corrected and uncorrected ε<sub>xx</sub>/ε<sub>yy</sub>/ε<sub>xy</sub>.
- Rotation Angle α slider: rotates the in-plane strain basis using the standard tensor rotation (see `rotate_strain` function).
- Interp On/Off: toggles interpolated vs. flat shading for the surface.
- Overlay group: HAADF / Height / Phase Map (mutually exclusive with the Strain group).

# 3. How to Run
#### 1. Make sure you install [**Anaconda**](https://www.anaconda.com/download) (create account and download for free).

#### 2. We begin by downloading the code.
<img width="1762" height="768" alt="image" src="https://github.com/user-attachments/assets/89124471-c322-4a93-97e3-fb45a315c1da" />

#### 3. Save the unzipped folder in a desired directory.

#### 4. Open Anaconda and **launch** `anaconda_prompt`.
<img width="1654" height="484" alt="image" src="https://github.com/user-attachments/assets/aa470bb6-0ce5-424e-b971-6ebdc5754179" />

#### 5. Go to the directory with the unzipped folder and **Copy Address as Text**:
<img width="1713" height="257" alt="image" src="https://github.com/user-attachments/assets/07d47c51-72d5-40ea-94ad-ab098f3b869d" />

#### 6. In the **Anaconda Prompt**, change directory to the one where the unzipped folder was saved using the following. Make sure that you type `cd` ("**c**hange **d**irectory") before the address you just copied. Paste it as is:
```
cd your\directory\here\4denoise-main
```

#### 7. We will then run the following (ensuring we are in the proper directory). This will download and install required Python packages to use **4Denoise**.
```
conda env create -f environment.yml
conda activate 4denoise-main
pip install -e .
python -m ipykernel install --user --name 4denoise-main --display-name "4denoise-main"
```

#### 8. We type `jupyter notebook` in the Anaconda Prompt (or select it directly in the Anaconda app) to open the Jupyter notebook application. We then open the file `DEMO_exp_4dstem_ripple_processing.ipynb`

The editable install now declares the packages needed to import and use
`fourdenoise`. Without the Conda environment, install the optional simulation
and notebook tools with `pip install -e ".[simulation,notebook]"`. The optional
BM3D/BM4D methods can be installed with `pip install -e ".[bm]"`.

To run the local regression suite from the repository directory:

```bash
python -m unittest discover -s tests
```

`HyperData` uses the last two axes as an image/pattern. Its `scan_shape` is
`()` for a 2D image, `(N,)` for a 3D stack, and `(Ry, Rx)` for a 4D scan.
`pattern_shape` (also available as `k_shape`) is the shape of the last two
axes. `real_shape` is `(Ry, Rx)` only for a 4D scan; it is `None` for a 3D
stack because the original 2D scan geometry is not known.

`HyperData.get_strains()` returns a `StrainResult` with named maps, fit
quality, and the source scan calibration when the maps match the 4D scan:

```python
result = data.get_strains(centers=measured, ref_centers=reference)
result.as_real_space('exx').show()
result.as_real_space('relative_fit_rmse').show()
valid = result.valid_mask
```

`exx`, `eyy`, and `exy` are dimensionless, while `erot` is in radians.
`fit_rmse` is the RMS peak-position residual in the units supplied for the
peaks; `relative_fit_rmse` divides it by the RMS reference-peak radius.
Neither is a confidence probability: compare error with `match_counts` and
`outlier_counts`, especially when only a few peaks were fitted. Existing
four-value unpacking and numeric indexing still work. With
`return_transform=True`, detailed transforms and peak-rejection information
are available through `result.diagnostics` and as item `result[4]`.

For a raw binary file with known layout, pass the full stored shape and dtype:

```python
data = HyperData(
    'scan.raw', raw_shape=(Ry, Rx, Ky, Kx), raw_dtype='>u2',
    raw_order='C',
)
```

The dtype can include byte order (`'>u2'` is big-endian unsigned 16-bit).
With an explicit `raw_shape`, no detector rows are trimmed unless
`raw_trim_meta=True` and `raw_trim_dims=(Ky, Kx)` are supplied. Without an
explicit shape, the legacy float32 EMPAD reader infers a square scan and
trims a 130x128 detector to 128x128 when those dimensions are available.
Rectangular scans require `raw_shape`.

Curve unfolding can retain excluded values for exact undo. The returned
metadata reports their size as `preserved_values_nbytes`; resizing with
`preserve_original=True` retains a complete original-tensor copy. Subsequent
shape-preserving operations share these read-only saved values, while
`HyperData.copy()` makes an independent metadata copy. Values outside a
center-cropped traversal remain unchanged by operations on its unfolded data.

`HyperData.save()` writes atomically by default: a failed write leaves an
existing output untouched. Overwriting temporarily needs disk space for both
the old and new files; use `atomic=False` only when that extra space is not
available and a partial file on failure is acceptable.
New saves use format version 1.1. Existing version 1.0 files remain readable;
unsupported or unversioned files raise a clear error instead of being
interpreted as generic HDF5 data.

For a large 3D/4D HDF5 dataset, read bounded scan blocks instead of loading
the entire array:

```python
from fourdenoise import HyperData

with HyperData.open_hdf5('experiment.h5') as source:
    pattern = source.get_dp(0, 0)  # 4D data: one scan position
    for scan_slices, block in source.iter_chunks((16, 16)):
        print(scan_slices, block.shape)  # each block is an in-memory HyperData
```

For 3D stacks, `get_dp` takes one index and `iter_chunks` accepts a scalar
chunk size. Use `hdf5_dataset='/entry/data'` when a file contains multiple
datasets. `HyperData(path)` remains eager, and methods called on a chunk do
not automatically process the rest of the HDF5 file.

`requirements.lock.txt` is a snapshot from a specific machine and contains
local `file:///` package paths. Use `environment.yml` or the editable install
above on another computer instead of treating that snapshot as a portable lock.
