# Quickstart with the Included Dataset

This example uses `mini_dataset_binned.npy` from the repository root. It is
a small 4D scan with shape `(15, 13, 128, 128)`, so the steps can run in one
Jupyter kernel without the multi-gigabyte experimental scan. The detector
radii below are in pixels; the mini-data example does not require physical
calibration. The full runnable version is the
[DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb).

## Load and Inspect

Start Jupyter from the repository root. The scan axes are `(Ry, Rx)` and the
last two axes are the diffraction pattern `(Ky, Kx)`.

```python
from fourdenoise import HyperData

data = HyperData("mini_dataset_binned.npy")
assert data.shape == (15, 13, 128, 128)
print(data.real_shape, data.pattern_shape)
```

## Make a Virtual Image

An annular virtual detector integrates pixels between radii 37 and 50 in
every diffraction pattern, producing a real-space `(15, 13)` image.

```python
image = data.virtual_image(
    annulus=(37, 50), detector_units="pixels", show=False,
)
image.show(title="Annular virtual image")
```

Inspect an average pattern from the reference strip:

```python
reference_dp = data.get_dp(y=(2, 11), x=6, selection_units="pixels")
reference_dp.show(title="Reference diffraction pattern", axes=False)
```

## Denoise and Find Peaks

`domain="reciprocal"` applies this median filter along the diffraction
axes, not across neighboring scan positions. Denoising returns a new
`HyperData`; `data` is unchanged.

```python
denoised = data.denoise(
    method="median", domain="reciprocal", window_size=3,
)
denoised.get_dp(y=(2, 11), x=6).show(axes=False)
```

Detect peaks on the original reference pattern, then refine their centers
and integrate intensities throughout the scan. These thresholds are
specific to the included mini dataset.

```python
reference_peaks = reference_dp.get_peaks(
    radius=3, min_distance=7, trench_width=2,
    threshold_abs=None, threshold_rel=0.1, r_range=(10, 61),
)
print(len(reference_peaks))  # 18 in the verified DEMO

centers = data.get_centers(r=5, ref_coords=reference_peaks, method="CoM")
intensities = data.get_intensities(
    r=5.25, centers=centers, method="CoM",
)
print(centers.shape, intensities.shape)
```

## Fit a Relative Strain Map

The reference below is the mean of refined peak positions from a selected
strip. It gives strain *relative to that strip*, not an absolute unstrained
crystal standard.

```python
reference_positions = centers[2:11, 6].mean(axis=0)
strain = data.get_strains(
    centers=centers, ref_centers=reference_positions,
)
strain.as_real_space("exx").show(
    title="Relative exx strain", cmap="RdBu", symmetric=True,
)
print(int(strain.valid_mask.sum()), "valid scan positions")
```

See [peaks and strain](../guides/peaks-and-strain.md) for fit-quality
interpretation, or the [case study](../case-studies/3d-strain-mapping.md)
for the separate full-data workflow.
