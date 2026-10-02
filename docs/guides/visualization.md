# Visualize a 4D Scan

`HyperData` holds the full scan. `get_dp(...)` returns a
`ReciprocalSpace` view of one or more diffraction patterns, while
`virtual_image(...)` returns a `RealSpace` image integrated through a
virtual detector. The final two array axes are always the pattern axes.

## Diffraction Patterns and Peak Overlays

```python
from fourdenoise import HyperData

data = HyperData("mini_dataset_binned.npy")
dp = data.get_dp(y=7, x=6)
dp.show(
    axis_units="pixels", grid=True, grid_ticks=(5, 5),
    scale_bar=20, scale_bar_color="white",
)
```

To show measured peak positions, pass `coords` in `(y, x)` pixel order:

```python
dp.show(coords=reference_peaks, c="white", s=25)
```

`mode="polar"` displays a circular radius/angle projection without changing
the stored data. For an actual transformed array, use
`HyperData.to_polar(...)`; the returned object's metadata records the polar
geometry, and `to_cartesian(...)` remaps it back by interpolation.

## Virtual Detectors

Choose exactly one detector specification: an annulus, one or more disks,
or a pixel mask. The detector's numeric inputs use `detector_units`.

```python
virtual = data.virtual_image(
    annulus=(37, 50), detector_units="pixels", show=False,
)
virtual.show(axes=False, cmap="magma")
```

Use `plot_mask=True` to preview the detector over a mean diffraction
pattern. `ring_color`, `ring_alpha`, `mask_cmap`, `mask_vmin`, and
`mask_vmax` control that preview. For an arbitrary detector, supply a
Boolean or weighted `mask` with shape `(Ky, Kx)`.

## Units and Scale Bars

When `real_units` and `real_conv_factor` are defined, `RealSpace.show`
accepts `axis_units="calibrated"` and `scale_bar` in those units. Likewise,
`ReciprocalSpace.show` uses reciprocal calibration for its axes and scale
bar. `axis_units="auto"` selects calibration when available and otherwise
uses pixels. Use `axis_units="pixels"` to request pixel labels explicitly.
Detector geometry and displayed scan axes are independent:
`virtual_image(detector_units=..., axis_units=...)` controls them separately.

See the [data model](../concepts/data-model.md) for the axis convention.

## Compare Real and Diffraction Orientations

Use `compare_rq` with a real-space image and a diffraction-space shadow image
showing the same specimen features. The real image stays fixed while the
diffraction image can rotate, translate, scale, or reflect. These are manual
preview controls; matching an ordinary Bragg pattern to a STEM image by
appearance does not establish an orientation calibration.

In Jupyter, install the optional backend with `%pip install ipympl` (or install
the package with `python -m pip install -e ".[interactive]"`). Restart the
kernel if this is the first installation, then run this before making the
viewer:

```python
%matplotlib widget
```

In Spyder, select an interactive Qt graphics backend. An inline notebook
image is static; the viewer warns if it detects that backend. Retain the
returned viewer to adjust it programmatically or export it later.

```python
# real_image is a RealSpace image or a 2D array.
# shadow_image is a Cartesian ReciprocalSpace image or a 2D array.
viewer = data.compare_rq(
    real_image=real_image,
    reciprocal_image=shadow_image,
    layout="overlay",              # also "side_by_side"
    real_alpha=0.7,
    reciprocal_alpha=0.7,
    real_show_kwargs={"cmap": "gray", "axis_units": "auto"},
    reciprocal_show_kwargs={"cmap": "magma", "logScale": False},
)
```

Each image has its own opacity slider. The diffraction rotation also has a
numeric entry box. Translation is `(dy, dx)` in reference-image pixels,
positive down/right. The preview scale is real horizontal pixels per
diffraction pixel; it does not overwrite the physical pixel calibration.
Known unequal scan pixel spacings are respected during rotation.

The pivot uses the diffraction image's `center_beam_metadata`: `center_px`,
then `mean_fit_center_px`, or a calibrated center converted to pixels. A raw
array matching the dataset's pattern shape inherits its beam metadata. Without
a stored center, the pivot is the exact fractional midpoint
`((Ky - 1) / 2, (Kx - 1) / 2)`. Stale metadata is rejected. The pivot marker
and its source are visible in the viewer. Initially it is placed at the
real-image midpoint; translation moves that anchor, and subsequent rotations
leave it fixed.

The displayed **diffraction correction** is positive counterclockwise. The
stored `RQCalibration` maps real directions to detector directions, so it
uses the inverse transform. Without reflection these two angles have opposite
signs. With reflection, the full matrix is inverted; the GUI mirrors the
diffraction image about its x or y axis before rotating it.

```python
viewer.set_parameters(rotation_deg=12.5, translation=(0.5, -2.0))
calibration = viewer.apply()        # same as Apply calibration in the GUI
print(calibration.rotation_deg, calibration.mirror_axis)
print(calibration.matrix)          # real -> reciprocal Cartesian (x, y)
viewer.export("rq_comparison.png") # image panels without GUI controls
data.save("calibrated.4denoise")
```

Apply stores only the orientation. The arrays are unchanged, and preview
scale, position and opacity do not become physical calibrations. Save/load,
copy, and value-only processing preserve the orientation; flips and
`rotate_dps` update its coordinate frame. If an angle is already known, use
`data.set_rq_calibration(rotation_deg=..., mirror_axis=...)` before opening
the viewer. This setter uses the real-to-detector convention.

The viewer accepts the intensity-related `.show()` options listed in the
[API reference](../reference/hyperdata.rst). Its axes and optional scale bar
use the reference image's units, including in side-by-side mode, where the
second panel shows the transformed diffraction image in that same frame.
Reset returns to the opening preview settings; it does not undo an already
applied calibration. Close the viewer with `viewer.close()` when finished.
