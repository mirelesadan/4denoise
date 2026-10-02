# Denoising and Unfolding

The [DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb) applies
a median filter to diffraction coordinates on a 4D scan. For denoisers that
need a 3D tensor, `HyperData.denoise(..., unfold_domain="real", ...)`
can unfold the scan before denoising and fold it back afterward. The
traversal method controls how scan positions become a 1D stack.

For a 4D scan, a 2D median window with `domain="reciprocal"` filters each
diffraction pattern in detector coordinates. A 3D median window can instead
follow scan positions in traversal order:

```python
denoised = data.denoise(
    method="median",
    unfold_domain="real",
    unfold_method="serpentine",
    window_size=(3, 3, 3),
)
```

The result is refolded to the scan geometry. Its first filter axis follows
the chosen traversal, so neighboring stack elements are not always physical
neighbors in a 2D scan. Use a full-shape traversal when every position must
be included in denoising.

`HyperData.unfold(domain="real", method="row_major")` maps a
`(Ry, Rx, Ky, Kx)` scan to `(Ry*Rx, Ky, Kx)`. Full-shape traversals such
as `serpentine` retain every scan position. Compatible-square traversals
such as `hilbert`, `morton`, and `peano` can instead center-crop or resize
their traversal domain.

Center-cropping excludes positions **from the unfolded tensor**, not from
the original object. The default `preserve_excess=True` stores excluded
values in unfolding metadata, so undo can reconstruct the full original
shape; denoising does not act on those excluded positions. Setting
`preserve_excess=False` means undo returns only the compatible cropped
tensor. In resize mode, undo normally returns the resized shape; exact
recovery of the original requires `preserve_original=True`, which stores a
full original-tensor copy. Metadata reports retained storage as
`preserved_values_nbytes`.

```python
unfolded = data.unfold(
    domain="real", method="hilbert", preserve_excess=True,
)
restored = unfolded.unfold(undo=True)
```

Reversibility depends on keeping the attached traversal metadata or
passing the metadata returned by `return_metadata=True`. Review the crop
warning and output shape before using a space-filling method on a large
dataset.

`HyperData.rank_scree(...)` performs a **separate fit per rank**, which can
be expensive. It measures the tensor actually denoised; for a cropped curve,
it does not include untouched excess positions. A convergence plot from a
single supported decomposition is different: it shows error over iterations
of that one fit. See the [HyperData API](../reference/hyperdata.rst) for the
exact arguments and available method information.
