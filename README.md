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
