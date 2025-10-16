⚠️ **Status (Oct 14, 2025)**: We are updating this package to ensure full reproducibility of the results in **_“Strain Mapping of Three-dimensionally Structured Two-dimensional Materials”_**, including Jupyter notebooks and detailed, step-by-step instructions.

# About the Project
**4Denoise** is a Python package for advanced processing, visualizing, and denoising 4D-STEM datasets.

This package is used for processing of data in _"**Strain Mapping of Three-dimensionally Structured Two-dimensional Materials**"_
See the Zenodo (https://zenodo.org/records/17246822) record for the MoS<sub>2</sub>–MoSe<sub>2</sub> heterojunction 4D-STEM dataset (`scan_x256_y256.raw`) and the simulated ripple dataset (`simulated_4d_dataset_highRes.npy`)

# BRIGHT MATLAB GUI (3D Strain & Topography Viewer)

<img width="1149" height="548" alt="image" src="https://github.com/user-attachments/assets/52c2c52f-20d8-429f-a906-590bdc09674a" />

### Overview

`gui.mat` is an interactive MATLAB interface for exploring the outputs of the BRIGHT workflow—namely, 3D topography, tilt-aware planar strain maps, and virtual overlays (e.g., HAADF or tilt phase).
It allows toggling between corrected/uncorrected strain components, rotating the 3D view, and visualizing how tilt correction affects the apparent strain distribution.

### Requirements

- MATLAB R2021a or newer
- Ten `.mat` input files generated from the Jupyter notebook ([`name of notebook`]) in the **4Denoise** repository:
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
