# Peaks and Strain Results

The [DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb) shows
one verified path from a reference diffraction pattern through peak
detection, center refinement, intensity integration, and a relative strain
map. Peak-selection thresholds and reference regions are data-dependent;
the [experimental case study](../case-studies/3d-strain-mapping.md) uses
different choices.

For a shared reference, detect spots once on an averaged diffraction pattern
and pass those `(y, x)` pixel positions to `get_centers`. Alternatively,
`HyperData.get_peaks(...)` detects each pattern independently and returns
ragged peak lists when counts vary. Keep the peak coordinate order and units
consistent when supplying `ref_centers` or `intensity_array` to a strain fit.

```python
reference_dp = data.get_dp(operation="mean")
reference_peaks = reference_dp.get_peaks(
    radius=3, min_distance=7, threshold_rel=0.1, r_range=(10, 61),
)
measured = data.get_centers(r=5, ref_coords=reference_peaks)
```

These thresholds illustrate the included mini dataset, not general detector
defaults. Inspect an overlay before fitting strain and adjust the peak set
for your material and calibration.

`HyperData.get_strains(centers=measured, ref_centers=reference)` returns a
`StrainResult` with named maps and fit diagnostics:

```python
result = data.get_strains(centers=measured, ref_centers=reference)
result.as_real_space("exx").show()
result.as_real_space("relative_fit_rmse").show()
valid = result.valid_mask
```

`exx`, `eyy`, and `exy` are dimensionless. `erot` is in radians.
`fit_rmse` measures residual peak-position error in the supplied peak
coordinate units; `relative_fit_rmse` divides that error by the RMS
reference-peak radius. Neither number is a confidence probability. Check
`match_counts`, `outlier_counts`, and `valid_mask` as well, especially
when only a few peaks are fitted. A 2D map inherits scan calibration from
the source 4D data when its shape matches that scan.

Existing four-value unpacking and numeric indexing still work. With
`return_transform=True`, transform and peak-rejection diagnostics are
available in `result.diagnostics` and as `result[4]`.
