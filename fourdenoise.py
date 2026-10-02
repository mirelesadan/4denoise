"""
The 4denoise data structures:
    - HyperData
    - ReciprocalSpace
    - RealSpace
    - StrainResult
    - _DenoisingMethods
    - _DenoiseEngine

Author: 
    Adan J Mireles
    Smalley Curl Insitute, Applied Physics
    Department of Materials Science and Nanoengineering
    Rice University; Houston, TX 

Date:
    April 2024
"""

import os
import tempfile
import warnings
import numpy as np
import h5py
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import hypot, isqrt, prod
from numbers import Integral


from scipy import ndimage
from scipy.stats import mode
import scipy.stats
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import label
from scipy.ndimage import rotate
from scipy.ndimage import grey_erosion, grey_dilation
from scipy import io
from scipy.linalg import polar
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.special import erfc
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy.ndimage import affine_transform
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import time

from skimage.measure import profile_line
from skimage.feature import peak_local_max
from skimage import transform
from skimage import feature
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, IncrementalPCA, MiniBatchNMF
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

import cv2
import inspect

import numba #new
from numba import jit, prange #new

import tensorly as tl
from tensorly.tt_matrix import tt_matrix_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tenalg import multi_mode_dot

from tensorly.decomposition import constrained_parafac
from tensorly.decomposition import parafac2 as par2
from tensorly.decomposition import tensor_ring_als as tr_als
from tensorly.decomposition import tensor_ring_als_sampled as tr_als_sampled
from tensorly.decomposition import tensor_train_matrix as tt_mat
from tensorly.decomposition import tensor_train as tt
from tensorly.decomposition import non_negative_tucker_hals as nnth
from tensorly.decomposition import non_negative_tucker as nnt
from tensorly.decomposition import partial_tucker as partial_tuck
from tensorly.decomposition import tucker as tuck
from tensorly.decomposition import randomised_parafac as rand_parafac
from tensorly.decomposition import non_negative_parafac_hals as nn_parafac_hals
from tensorly.decomposition import non_negative_parafac as nn_parafac
from tensorly.decomposition import parafac as par
from tensorly.decomposition import robust_pca as robust_tensor_pca

from tensorly.decomposition import parafac_power_iteration as parafac_power_iter
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac_power_iter

from pathlib import Path
from typing import Union, Sequence, Tuple
from fourdenoise_rq import (
    RQCalibration,
    RQComparison,
    _coerce_rq_calibration,
    _reframe_rq_calibration,
    _rotation_matrix as _rq_rotation_matrix,
)
from fourdenoise_geometry import (
    _calibrated_center_to_pixels,
    _center_to_calibrated,
    _normalize_real_origin,
    _normalize_real_spacing,
    _normalize_unit_mode,
    _parse_real_selection,
    _real_spacing_pair,
    _resolve_unit_mode,
    _scaled_real_spacing,
)

_SCALE_UNSET = object()
_HYPERDATA_HDF5_FORMAT = '4denoise.hyperdata'
_HYPERDATA_HDF5_VERSION = '1.1'
_HYPERDATA_HDF5_READABLE_VERSIONS = ('1.0', '1.1')


@dataclass(frozen=True)
class PeakDetectionResult:
    """Peak coordinates and their origin, in matching row order.

    ``scores`` are template-correlation values for measured peaks and NaN for
    symmetry-generated peaks. ``orbit_ids`` is -1 for peaks not assigned to a
    complete-enough rotational orbit (including central peaks).
    """

    coords: np.ndarray
    scores: np.ndarray
    synthetic_mask: np.ndarray
    orbit_ids: np.ndarray

    @property
    def observed_mask(self):
        """Boolean mask selecting peaks actually detected in the pattern."""
        return ~self.synthetic_mask


def _center_beam_metadata_from_pixels(radius_px, center_px, shape, *,
                                      units=None, conv_factor=None,
                                      **extra_metadata):
    """Build direct-beam metadata from pixel-space beam parameters."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2:
        raise ValueError("Direct-beam metadata requires a 2D reciprocal shape.")

    radius_px = float(radius_px)
    center_px = tuple(float(v) for v in center_px)
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Direct-beam radius must be positive and finite.")
    if len(center_px) != 2 or not np.all(np.isfinite(center_px)):
        raise ValueError("Direct-beam center must contain two finite values.")

    metadata = {
        'radius_px': radius_px,
        'center_px': center_px,
        'shape': shape,
        'coordinate_convention': (
            "center_px is an array-coordinate (ky, kx) pair. "
            "center_calibrated, when present, is relative to the diffraction "
            "origin with positive ky upward and positive kx to the right."
        ),
    }

    has_units = units is not None
    has_factor = conv_factor is not None
    if has_units != has_factor:
        raise ValueError(
            "Direct-beam metadata cannot use a partial reciprocal-space "
            "calibration. Define both units and conv_factor, or clear both."
        )
    if has_units:
        if not np.isscalar(conv_factor) or conv_factor <= 0:
            raise ValueError("conv_factor must be a positive scalar.")
        metadata.update({
            'calibrated_units': str(units).strip(),
            'conv_factor': float(conv_factor),
            'radius_calibrated': radius_px * float(conv_factor),
            'center_calibrated': _center_to_calibrated(
                center_px,
                shape,
                conv_factor,
            ),
        })

    for key, value in extra_metadata.items():
        if value is not None:
            metadata[key] = deepcopy(value)

    return metadata


def _resampled_center_beam_metadata(metadata, old_shape, new_shape, *,
                                    crop_origin=(0, 0), crop_shape=None,
                                    units=None, conv_factor=None):
    """Map a circular beam through an integer crop and isotropic resampling.

    Return None when the old geometry is stale, the beam center is cropped out,
    or unequal axis scales would turn the circle into an ellipse. Historical
    alignment coordinates are not copied because they describe the old image.
    """
    if metadata is None:
        return None
    old_shape = tuple(old_shape)
    new_shape = tuple(new_shape)
    crop_shape = tuple(crop_shape) if crop_shape is not None else old_shape
    if tuple(metadata.get('shape', ())) != old_shape:
        return None
    if 'center_px' not in metadata or 'radius_px' not in metadata:
        return None

    scales = np.asarray(new_shape, dtype=float) / np.asarray(crop_shape, dtype=float)
    if not np.isclose(scales[0], scales[1]):
        return None
    center = np.asarray(metadata['center_px'], dtype=float)
    origin = np.asarray(crop_origin, dtype=float)
    new_center = (center - origin + 0.5) * scales - 0.5
    if not np.all(np.isfinite(new_center)) or np.any(new_center < 0) or np.any(
        new_center > np.asarray(new_shape) - 1
    ):
        return None
    return _center_beam_metadata_from_pixels(
        float(metadata['radius_px']) * scales[0], new_center, new_shape,
        units=units, conv_factor=conv_factor,
        source='resampling', parent_source=metadata.get('source'),
    )


def _validate_center_pair(value, name):
    """Return a finite ``(ky, kx)`` pair as floats."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly two finite values.")
    return (float(arr[0]), float(arr[1]))


def _resolve_center_beam_parameters(shape, metadata=None, *, radius=None,
                                    center=None, units=None, conv_factor=None,
                                    beam_units='auto'):
    """
    Resolve direct-beam center and radius to pixel coordinates.

    Explicit ``radius`` and ``center`` values take precedence over metadata.
    Missing centers fall back to metadata and then the geometric midpoint.
    """
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2:
        raise ValueError("Direct-beam operations require 2D reciprocal axes.")
    metadata = deepcopy(metadata) if metadata is not None else {}
    if not isinstance(metadata, dict):
        raise ValueError("center_beam_metadata must be a dictionary or None.")

    _, factor, unit_mode = _resolve_unit_mode(
        beam_units,
        units,
        conv_factor,
        label='beam_units',
    )

    radius_source = 'explicit'
    if radius is None:
        radius_source = 'metadata'
        if 'radius_px' in metadata:
            radius_px = float(metadata['radius_px'])
        elif 'radius_calibrated' in metadata:
            metadata_factor = metadata.get('conv_factor', factor)
            if metadata_factor is None:
                raise ValueError(
                    "center_beam_metadata stores radius_calibrated but no "
                    "conv_factor, and this object has no reciprocal-space "
                    "calibration."
                )
            radius_px = float(metadata['radius_calibrated']) / float(metadata_factor)
        else:
            raise ValueError(
                "No direct-beam radius was provided and no "
                "center_beam_metadata['radius_px'] is available. Provide "
                "radius=... or run alignment(...) first."
            )
    else:
        radius_px = float(radius)
        if unit_mode == 'calibrated':
            radius_px /= float(factor)

    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Direct-beam radius must be positive and finite.")

    center_source = 'explicit'
    if center is None:
        center_source = 'metadata'
        if 'center_px' in metadata:
            center_px = _validate_center_pair(metadata['center_px'], 'center_px')
        elif 'center_calibrated' in metadata:
            metadata_factor = metadata.get('conv_factor', factor)
            if metadata_factor is None:
                raise ValueError(
                    "center_beam_metadata stores center_calibrated but no "
                    "conv_factor, and this object has no reciprocal-space "
                    "calibration."
                )
            center_px = _calibrated_center_to_pixels(
                metadata['center_calibrated'],
                metadata_factor,
                shape,
            )
        else:
            center_source = 'geometric_center'
            center_px = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    else:
        center_pair = _validate_center_pair(center, 'center')
        if unit_mode == 'calibrated':
            center_px = _calibrated_center_to_pixels(
                center_pair,
                factor,
                shape,
            )
        else:
            center_px = center_pair

    if not np.all(np.isfinite(center_px)):
        raise ValueError("Direct-beam center could not be resolved to pixels.")

    resolved_metadata = _center_beam_metadata_from_pixels(
        radius_px,
        center_px,
        shape,
        units=units,
        conv_factor=conv_factor,
        resolution={
            'radius_source': radius_source,
            'center_source': center_source,
            'beam_units': beam_units,
            'beam_unit_mode': unit_mode,
        },
    )
    return center_px, radius_px, resolved_metadata


def _decode_hdf5_value(value):
    """Convert HDF5 byte strings and NumPy scalars to Python values."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.dtype == object:
        flat_values = value.ravel()
        if all(isinstance(item, (bytes, str)) for item in flat_values):
            decoded = [
                item.decode('utf-8') if isinstance(item, bytes) else item
                for item in flat_values
            ]
            return np.array(decoded, dtype=str).reshape(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hdf5_dataset_kwargs(value, compression, compression_opts):
    """Return compression kwargs only when HDF5 supports them for the value."""
    array = np.asarray(value)
    if compression is None or array.shape == ():
        return {}
    kwargs = {'compression': compression}
    if compression_opts is not None:
        kwargs['compression_opts'] = compression_opts
    return kwargs


def _write_hdf5_value(parent, name, value, compression=None,
                      compression_opts=None):
    """Recursively write a Python metadata value into an HDF5 group."""
    if value is None:
        group = parent.create_group(name)
        group.attrs['kind'] = 'none'
        return

    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs['kind'] = 'dict'
        for key, item in value.items():
            key_name = str(key)
            if '/' in key_name:
                raise ValueError(
                    "Metadata dictionary keys cannot contain '/'. "
                    f"Got key {key_name!r}."
                )
            _write_hdf5_value(
                group,
                key_name,
                item,
                compression=compression,
                compression_opts=compression_opts,
            )
        return

    if isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs['kind'] = 'tuple' if isinstance(value, tuple) else 'list'
        group.attrs['length'] = len(value)
        for idx, item in enumerate(value):
            _write_hdf5_value(
                group,
                f'item_{idx:08d}',
                item,
                compression=compression,
                compression_opts=compression_opts,
            )
        return

    if isinstance(value, str):
        dtype = h5py.string_dtype(encoding='utf-8')
        dataset = parent.create_dataset(name, data=value, dtype=dtype)
        dataset.attrs['kind'] = 'str'
        return

    if isinstance(value, bytes):
        dtype = h5py.string_dtype(encoding='utf-8')
        dataset = parent.create_dataset(
            name,
            data=value.decode('utf-8'),
            dtype=dtype,
        )
        dataset.attrs['kind'] = 'str'
        return

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            group = parent.create_group(name)
            group.attrs['kind'] = 'ndarray-object'
            group.attrs['shape'] = value.shape
            flat_values = value.ravel()
            group.attrs['length'] = flat_values.size
            for idx, item in enumerate(flat_values):
                _write_hdf5_value(
                    group,
                    f'item_{idx:08d}',
                    item,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            return

        if value.dtype.kind in {'U', 'S'}:
            dtype = h5py.string_dtype(encoding='utf-8')
            dataset = parent.create_dataset(
                name,
                data=value.astype(str),
                dtype=dtype,
            )
        else:
            dataset = parent.create_dataset(
                name,
                data=value,
                **_hdf5_dataset_kwargs(value, compression, compression_opts),
            )
        dataset.attrs['kind'] = 'ndarray'
        return

    if np.isscalar(value):
        dataset = parent.create_dataset(name, data=value)
        dataset.attrs['kind'] = 'scalar'
        return

    raise TypeError(
        f"Cannot save metadata value {name!r} with unsupported type "
        f"{type(value).__name__}."
    )


def _read_hdf5_value(parent, name):
    """Recursively read a Python metadata value from an HDF5 group."""
    obj = parent[name]
    kind = _decode_hdf5_value(obj.attrs.get('kind', 'ndarray'))

    if kind == 'none':
        return None

    if isinstance(obj, h5py.Dataset):
        value = obj[()]
        return _decode_hdf5_value(value)

    if kind == 'dict':
        return {
            key: _read_hdf5_value(obj, key)
            for key in obj.keys()
        }

    if kind in {'list', 'tuple'}:
        length = int(obj.attrs.get('length', len(obj.keys())))
        values = [
            _read_hdf5_value(obj, f'item_{idx:08d}')
            for idx in range(length)
        ]
        return tuple(values) if kind == 'tuple' else values

    if kind == 'ndarray-object':
        shape = tuple(int(v) for v in obj.attrs['shape'])
        length = int(obj.attrs.get('length', np.prod(shape)))
        values = [
            _read_hdf5_value(obj, f'item_{idx:08d}')
            for idx in range(length)
        ]
        return np.array(values, dtype=object).reshape(shape)

    raise ValueError(f"Unsupported HDF5 metadata kind {kind!r}.")


def _is_hyperdata_hdf5_file(filename):
    """Return True if filename is a 4Denoise HyperData HDF5 file."""
    path = Path(filename).expanduser()
    if not path.exists() or not path.is_file():
        return False
    try:
        with h5py.File(path, 'r') as file:
            file_format = _decode_hdf5_value(
                file.attrs.get('fourdenoise_format', None)
            )
            return file_format == _HYPERDATA_HDF5_FORMAT
    except OSError:
        return False


def _checked_hyperdata_format_version(file, path):
    """Validate the on-disk schema before reading saved HyperData metadata."""
    version = _decode_hdf5_value(file.attrs.get('format_version'))
    if version is None:
        raise ValueError(
            f"'{path}' is missing its 4Denoise format_version; the saved "
            "metadata schema cannot be determined safely."
        )
    if version not in _HYPERDATA_HDF5_READABLE_VERSIONS:
        raise ValueError(
            f"'{path}' uses unsupported 4Denoise format version "
            f"{version!r}; this installation can read versions "
            f"{', '.join(_HYPERDATA_HDF5_READABLE_VERSIONS)}."
        )
    return version


def _migrate_hyperdata_metadata(metadata, version):
    """Normalize each supported file schema to the current in-memory form."""
    if version in _HYPERDATA_HDF5_READABLE_VERSIONS:
        migrated = dict(metadata)
        migrated.setdefault('real_origin', (0.0, 0.0))
        if version == '1.0' and isinstance(migrated.get('unfold_metadata'), dict):
            unfolded = dict(migrated['unfold_metadata'])
            unfolded.setdefault(
                'preserved_values_nbytes',
                sum(values.nbytes for values in _unfold_payload_arrays(unfolded)),
            )
            migrated['unfold_metadata'] = unfolded
        return migrated
    raise ValueError(f"No metadata migration is defined for version {version!r}.")


def _load_hyperdata_hdf5(filename):
    """Load array and metadata from a 4Denoise HyperData HDF5 file."""
    path = Path(filename).expanduser()
    with h5py.File(path, 'r') as file:
        file_format = _decode_hdf5_value(
            file.attrs.get('fourdenoise_format', None)
        )
        if file_format != _HYPERDATA_HDF5_FORMAT:
            raise ValueError(
                f"'{path}' is not a 4Denoise HyperData file."
            )
        if 'array' not in file:
            raise ValueError(f"'{path}' does not contain a saved data array.")

        version = _checked_hyperdata_format_version(file, path)
        array = file['array'][()]
        metadata = {}
        if 'metadata' in file:
            metadata_group = file['metadata']
            metadata = {
                key: _read_hdf5_value(metadata_group, key)
                for key in metadata_group.keys()
            }
    return array, _migrate_hyperdata_metadata(metadata, version)


def _format_hdf5_dataset_listing(dataset_info):
    """Format discovered HDF5 datasets for selection and error messages."""
    if not dataset_info:
        return '  (no datasets found)'

    return '\n'.join(
        f"  {info['path']}: shape={info['shape']}, "
        f"ndim={info['ndim']}, dtype={info['dtype']}"
        for info in dataset_info
    )


def _select_hdf5_dataset(file, dataset_path=None, *,
                         selector_name='hdf5_dataset', exclude_internal=False):
    """Select a numeric dataset without loading it into memory."""
    path = Path(file.filename)
    dataset_info = []

    def _collect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_info.append({
                'path': f'/{name}', 'shape': tuple(obj.shape),
                'ndim': obj.ndim, 'dtype': str(obj.dtype),
                'is_candidate': obj.ndim >= 2 and (
                    np.issubdtype(obj.dtype, np.number)
                    or np.issubdtype(obj.dtype, np.bool_)
                ),
            })

    file.visititems(_collect_dataset)
    if dataset_path is not None:
        if not isinstance(dataset_path, (str, Path)):
            raise TypeError(
                f"{selector_name} must be a string or pathlib.Path."
            )
        selected_path = str(dataset_path).replace('\\', '/')
        if not selected_path.strip('/'):
            raise ValueError(f"{selector_name} must name a dataset.")
        if not selected_path.startswith('/'):
            selected_path = f'/{selected_path}'
        try:
            selected = file[selected_path]
        except KeyError as exc:
            listing = _format_hdf5_dataset_listing(dataset_info)
            raise KeyError(
                f"HDF5 dataset '{selected_path}' was not found in "
                f"'{path}'.\nAvailable datasets:\n{listing}"
            ) from exc
        if not isinstance(selected, h5py.Dataset):
            raise ValueError(
                f"HDF5 path '{selected_path}' refers to a group, not a dataset."
            )
        if selected.ndim < 2:
            raise ValueError(
                f"HDF5 dataset '{selected_path}' has {selected.ndim} "
                "dimension(s); HyperData requires at least 2."
            )
        if not (
            np.issubdtype(selected.dtype, np.number)
            or np.issubdtype(selected.dtype, np.bool_)
        ):
            raise TypeError(
                f"HDF5 dataset '{selected_path}' has non-numeric dtype "
                f"{selected.dtype}."
            )
        return selected

    candidates = [
        info for info in dataset_info
        if info['is_candidate'] and (
            not exclude_internal
            or not any(
                part.startswith('#') for part in info['path'].split('/')
            )
        )
    ]
    if len(candidates) == 1:
        return file[candidates[0]['path']]
    if not candidates:
        listing = _format_hdf5_dataset_listing(dataset_info)
        raise ValueError(
            f"No numeric dataset with at least 2 dimensions was found "
            f"in '{path}'.\nAvailable datasets:\n{listing}"
        )
    listing = _format_hdf5_dataset_listing(candidates)
    raise ValueError(
        f"{len(candidates)} numeric multidimensional datasets were "
        f"found in '{path}'. Select one with "
        f"HyperData(filename, {selector_name}='/path/to/dataset').\n"
        f"Candidate datasets:\n{listing}"
    )


def _read_hdf5_file(filename, dataset_path=None):
    """Load a selected generic HDF5 dataset into a NumPy array."""
    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file.")
    try:
        with h5py.File(path, 'r') as file:
            return _select_hdf5_dataset(file, dataset_path)[()]
    except OSError as exc:
        raise ValueError(f"Could not open '{path}' as an HDF5 file.") from exc


class _HDF5ChunkReader:
    """Read-only, bounded-memory access to 3D/4D HDF5 diffraction data."""

    def __init__(self, dataset, hyperdata_type, metadata=None):
        if dataset.ndim not in (3, 4):
            raise ValueError(
                "Chunked HDF5 access requires a 3D stack or 4D-STEM dataset."
            )
        self._dataset = dataset
        self._hyperdata_type = hyperdata_type
        self._metadata = metadata or {}
        self._closed = False
        self.shape = tuple(dataset.shape)
        self.dtype = dataset.dtype
        self.ndim = dataset.ndim

    def _require_open(self):
        if self._closed:
            raise RuntimeError("The HDF5 reader is closed; use it inside the with block.")

    def get_dp(self, *scan_indices):
        """Load one diffraction pattern at a scan index without reading the stack."""
        self._require_open()
        expected = self.ndim - 2
        if len(scan_indices) != expected:
            raise ValueError(f"Expected {expected} scan index/indices.")
        for index, size in zip(scan_indices, self.shape[:expected]):
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (Integral, np.integer)
            ):
                raise TypeError("Scan indices must be integers.")
            if not 0 <= index < size:
                raise IndexError(f"Scan index {index} is outside [0, {size}).")
        pattern = self._dataset[scan_indices + (slice(None), slice(None))]
        return ReciprocalSpace(
            pattern,
            units=self._metadata.get('reciprocal_units'),
            conv_factor=self._metadata.get('reciprocal_conv_factor'),
            polar_metadata=self._metadata.get('polar_metadata'),
            center_beam_metadata=self._metadata.get('center_beam_metadata'),
        )

    def iter_chunks(self, chunk_shape=16):
        """Yield ``(scan_slices, HyperData)`` blocks bounded by chunk_shape.

        A scalar selects that many scan positions along each leading axis.
        Pass a two-tuple for 4D data or a one-tuple for a 3D stack. Returned
        chunks own their NumPy data and remain usable after the reader closes.
        """
        self._require_open()
        n_scan_axes = self.ndim - 2
        if isinstance(chunk_shape, (Integral, np.integer)) and not isinstance(
            chunk_shape, (bool, np.bool_)
        ):
            lengths = (int(chunk_shape),) * n_scan_axes
        else:
            try:
                lengths = tuple(chunk_shape)
            except TypeError as exc:
                raise ValueError("chunk_shape must contain positive integers.") from exc
        if len(lengths) != n_scan_axes or any(
            isinstance(length, (bool, np.bool_))
            or not isinstance(length, (Integral, np.integer)) or length < 1
            for length in lengths
        ):
            raise ValueError(
                f"chunk_shape must contain {n_scan_axes} positive integer(s)."
            )

        scan_shape = self.shape[:n_scan_axes]
        grid_shape = tuple(
            (size + length - 1) // length
            for size, length in zip(scan_shape, lengths)
        )
        for block_index in np.ndindex(*grid_shape):
            self._require_open()
            scan_slices = tuple(
                slice(index * length, min((index + 1) * length, size))
                for index, length, size in zip(block_index, lengths, scan_shape)
            )
            block = self._dataset[scan_slices + (slice(None), slice(None))]
            origin = self._metadata.get('real_origin', (0.0, 0.0))
            if self.ndim == 4:
                real_step = _real_spacing_pair(
                    self._metadata.get('real_conv_factor')
                )
                origin = tuple(
                    value + scan_slice.start * step
                    for value, scan_slice, step in zip(origin, scan_slices, real_step)
                )
            yield scan_slices, self._hyperdata_type(
                block,
                real_units=self._metadata.get('real_units'),
                real_conv_factor=self._metadata.get('real_conv_factor'),
                real_origin=origin,
                reciprocal_units=self._metadata.get('reciprocal_units'),
                reciprocal_conv_factor=self._metadata.get('reciprocal_conv_factor'),
                polar_metadata=self._metadata.get('polar_metadata'),
                center_beam_metadata=self._metadata.get('center_beam_metadata'),
                rq_calibration=self._metadata.get('rq_calibration'),
            )
    
#%%

# =============================================================================
# Useful Functions
# =============================================================================

#TODO: Allow for reading data that has more than (128,128) in k-space
#TODO: read EMD file data
#TODO: combine with RosettaSciIO

def _validate_raw_dimensions(value, name, allowed_lengths):
    """Normalize user-specified raw dimensions without silently rounding."""
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain {allowed_lengths} positive integer dimensions."
        ) from exc
    if (
        len(dimensions) not in allowed_lengths
        or any(
            not isinstance(size, Integral)
            or isinstance(size, (bool, np.bool_))
            or size <= 0
            for size in dimensions
        )
    ):
        raise ValueError(
            f"{name} must contain {allowed_lengths} positive integer dimensions."
        )
    return tuple(map(int, dimensions))


def read_4D(fname, dp_dims=(128, 130), trim_dims=(128, 128),
            trim_meta=None, clip=False, hdf5_dataset=None, repair_nans=False,
            raw_shape=None, raw_dtype=np.float32, raw_order='C',
            mat_variable=None):
    """
    Read array data from a .raw, .mat, .npy, .h5, .hdf5, or .hdf file.
    
    Function written by Chuqiao Shi (2022)
    See on GitHub: Chuqiao2333/Hierarchical_Clustering
    
    Modified by Adan J Mireles (April, 2024)
    - Addition of 'dp_dims', 'trim_dims', and 'trim_meta'
    - Modified 'print' statement
    
    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the input file.
    hdf5_dataset : str or pathlib.Path or None, optional
        HDF5 dataset path to load. Generic HDF5 files containing exactly one
        numeric multidimensional dataset are selected automatically. This
        argument is required when a file contains multiple candidates.
    mat_variable : str or pathlib.Path or None, optional
        Name of a numeric array in a MATLAB ``.mat`` file. For MATLAB v7.3
        (HDF5) files, a nested dataset path is also accepted. When omitted,
        the only numeric array with at least two dimensions is selected;
        multiple candidates require an explicit choice.
    raw_shape : tuple of 2 to 4 ints or None, optional
        Full shape stored in a raw file, in array axis order. For 4D-STEM,
        specify ``(Ry, Rx, Ky, Kx)``. If omitted, the legacy EMPAD layout
        uses ``dp_dims`` and requires a square scan grid.
    raw_dtype : numpy dtype, optional
        Stored raw scalar type, including byte order (for example ``'>u2'``).
        Defaults to float32. Used only for ``.raw`` files.
    raw_order : {'C', 'F'}, optional
        Storage order of raw values. Defaults to NumPy row-major order.
    trim_meta : bool or None, optional
        Crop the raw detector axes to ``trim_dims``. None keeps the historical
        crop for inferred EMPAD layouts and disables it for explicit
        ``raw_shape``, so an explicit shape is loaded without hidden trimming.
    clip : bool, optional
        Replace values below 1 with 1. Disabled by default to preserve raw
        detector counts, including zeros and negative values.
    repair_nans : bool, optional
        Replace patterns containing NaNs with neighbor averages (or replace
        NaNs with zero for 2D images). Disabled by default.

    Return:
        dp : numpy array
        
    Function modified by Adan Mireles to make '.mat' file reading more general (July 2024)
    """

    def _replace_nan_patterns(dp):
        """Replace corrupted 2D patterns in 3D/4D stacks with neighbor averages."""
        if not np.issubdtype(dp.dtype, np.number) or not np.isnan(dp).any():
            return dp

        if dp.ndim < 3:
            print('Found NaNs. Replacing with zeros...')
            return np.nan_to_num(dp, nan=0.0)

        scan_shape = dp.shape[:-2]
        nan_mask = np.isnan(dp).any(axis=(-2, -1))

        num_NaNs = int(nan_mask.sum())
        if num_NaNs == 0:
            return dp

        print(f'Found {num_NaNs} corrupted diffraction patterns. Replacing with local average...')

        bad_indices = np.argwhere(nan_mask)
        for bad_index_array in bad_indices:
            bad_index = tuple(bad_index_array)
            neighbors = []

            for offset_values in np.ndindex(*(3,) * len(scan_shape)):
                offsets = tuple(value - 1 for value in offset_values)
                if all(offset == 0 for offset in offsets):
                    continue

                neighbor_index = tuple(
                    index + offset
                    for index, offset in zip(bad_index, offsets)
                )
                in_bounds = all(
                    0 <= index < size
                    for index, size in zip(neighbor_index, scan_shape)
                )
                if in_bounds and not nan_mask[neighbor_index]:
                    neighbors.append(dp[neighbor_index])

            if neighbors:
                dp[bad_index] = np.stack(neighbors, axis=0).mean(axis=0)
            else:
                dp[bad_index] = 0.0

        print('...Done')
        return dp

    # Read 4D data from .raw file
    fname_end = Path(fname).suffix.lower()
    if fname_end != '.raw' and raw_shape is not None:
        raise ValueError("raw_shape applies only to .raw files.")
    if fname_end != '.mat' and mat_variable is not None:
        raise ValueError("mat_variable applies only to .mat files.")

    if fname_end == '.raw':
        try:
            dtype = np.dtype(raw_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid raw_dtype {raw_dtype!r}.") from exc
        if dtype.kind not in 'biufc':
            raise ValueError("raw_dtype must be a numeric or boolean dtype.")
        if raw_order not in ('C', 'F'):
            raise ValueError("raw_order must be 'C' or 'F'.")
        if trim_meta is not None and not isinstance(trim_meta, (bool, np.bool_)):
            raise ValueError("trim_meta must be a boolean or None.")

        explicit_raw_shape = raw_shape is not None
        if explicit_raw_shape:
            raw_shape = _validate_raw_dimensions(
                raw_shape, 'raw_shape', (2, 3, 4),
            )
        else:
            columns, rows = _validate_raw_dimensions(dp_dims, 'dp_dims', (2,))

        file_size = Path(fname).stat().st_size
        if file_size % dtype.itemsize:
            raise ValueError(
                f"Raw file has {file_size} bytes, not a multiple of "
                f"raw_dtype item size {dtype.itemsize}."
            )
        if raw_shape is None:
            frame_elements = rows * columns
            n_elements = file_size // dtype.itemsize
            if n_elements % frame_elements:
                raise ValueError(
                    "Raw file size is not divisible by dp_dims; specify the "
                    "correct dp_dims or raw_shape."
                )
            n_patterns = n_elements // frame_elements
            scan_side = isqrt(n_patterns)
            if scan_side == 0 or scan_side * scan_side != n_patterns:
                raise ValueError(
                    "Raw scan is not square; specify raw_shape=(Ry, Rx, Ky, Kx)."
                )
            raw_shape = (scan_side, scan_side, rows, columns)

        expected_size = prod(raw_shape) * dtype.itemsize
        if file_size != expected_size:
            raise ValueError(
                f"Raw file has {file_size} bytes but raw_shape={raw_shape} "
                f"and raw_dtype={dtype} require {expected_size} bytes."
            )

        dp = np.fromfile(fname, dtype=dtype).reshape(raw_shape, order=raw_order)

        if trim_meta is None:
            trim_meta = False
            if not explicit_raw_shape:
                trim_dims = _validate_raw_dimensions(trim_dims, 'trim_dims', (2,))
                trim_meta = all(
                    size <= limit for size, limit in zip(trim_dims, dp.shape[-2:])
                )
        if trim_meta:
            trim_dims = _validate_raw_dimensions(trim_dims, 'trim_dims', (2,))
            if any(size > limit for size, limit in zip(trim_dims, dp.shape[-2:])):
                raise ValueError(
                    "trim_dims must contain two positive sizes no larger than "
                    f"the raw detector shape {dp.shape[-2:]}."
                )
            dp = dp[..., :trim_dims[0], :trim_dims[1]]

    elif fname_end == '.mat':
        if hdf5_dataset is not None:
            raise ValueError(
                "Use mat_variable, not hdf5_dataset, to select .mat data."
            )
        dp = _read_mat_file(fname, mat_variable=mat_variable)
        
    elif fname_end == '.npy':
        dp = np.load(fname)

    elif fname_end in {'.h5', '.hdf5', '.hdf'}:
        dp = _read_hdf5_file(fname, dataset_path=hdf5_dataset)
        
    else:
        raise ValueError(
            "This function supports .mat, .raw, .npy, .h5, .hdf5, and .hdf "
            "files."
        )

    if dp is None:
        raise ValueError(f"Could not read data from '{fname}'.")

    dp = np.asarray(dp)

    if clip and np.issubdtype(dp.dtype, np.number):
        dp = np.array(dp, copy=True)
        # Replace negative and near-zero pixel values with 1
        low_vals_mask = dp < 1
        dp[low_vals_mask] = 1

    if repair_nans:
        dp = _replace_nan_patterns(np.array(dp, copy=True))

    return dp

_MAT_NUMERIC_CLASSES = frozenset({
    'double', 'single', 'int8', 'uint8', 'int16', 'uint16',
    'int32', 'uint32', 'int64', 'uint64', 'logical',
})


def _format_mat_variable_listing(variables):
    """List MATLAB variables without loading their array contents."""
    if not variables:
        return '  (no variables found)'
    return '\n'.join(
        f'  {name}: shape={shape}, class={matlab_class}'
        for name, shape, matlab_class in variables
    )


def _read_mat_file(filename, mat_variable=None):
    """Load one numeric 2D+ MATLAB variable, with explicit ambiguity handling.

    MATLAB v4-v7.2 files are inspected with ``whosmat`` and only the selected
    variable is loaded. MATLAB v7.3 files are read as HDF5 datasets.
    """
    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file.")
    if h5py.is_hdf5(path):
        return _read_mat_file_h5py(path, mat_variable=mat_variable)

    if mat_variable is not None:
        if not isinstance(mat_variable, (str, Path)):
            raise TypeError("mat_variable must be a string or pathlib.Path.")
        mat_variable = str(mat_variable)
        if not mat_variable:
            raise ValueError("mat_variable must name a MATLAB variable.")

    try:
        variables = io.whosmat(str(path))
    except NotImplementedError:
        return _read_mat_file_h5py(path, mat_variable=mat_variable)
    except Exception as exc:
        raise ValueError(
            f"Could not inspect '{path}' as a MATLAB MAT file: {exc}"
        ) from exc

    candidates = [
        info for info in variables
        if len(info[1]) >= 2 and info[2] in _MAT_NUMERIC_CLASSES
    ]
    if mat_variable is None:
        if not candidates:
            raise ValueError(
                f"No numeric array with at least 2 dimensions was found in "
                f"'{path}'.\nAvailable variables:\n"
                f"{_format_mat_variable_listing(variables)}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"{len(candidates)} numeric multidimensional variables were "
                f"found in '{path}'. Select one with "
                "HyperData(filename, mat_variable='name').\n"
                f"Candidate variables:\n{_format_mat_variable_listing(candidates)}"
            )
        mat_variable = candidates[0][0]
    else:
        selected = next(
            (info for info in variables if info[0] == mat_variable), None
        )
        if selected is None:
            raise KeyError(
                f"MATLAB variable '{mat_variable}' was not found in "
                f"'{path}'.\nAvailable variables:\n"
                f"{_format_mat_variable_listing(variables)}"
            )
        if selected[2] not in _MAT_NUMERIC_CLASSES:
            raise TypeError(
                f"MATLAB variable '{mat_variable}' has non-numeric class "
                f"{selected[2]!r}."
            )
        if len(selected[1]) < 2:
            raise ValueError(
                f"MATLAB variable '{mat_variable}' has fewer than 2 dimensions."
            )

    try:
        data = io.loadmat(str(path), variable_names=[mat_variable])[mat_variable]
    except Exception as exc:
        raise ValueError(
            f"Could not load MATLAB variable '{mat_variable}' from '{path}': "
            f"{exc}"
        ) from exc
    if data.ndim < 2 or not (
        np.issubdtype(data.dtype, np.number)
        or np.issubdtype(data.dtype, np.bool_)
    ):
        raise TypeError(
            f"MATLAB variable '{mat_variable}' did not load as a numeric "
            "array with at least 2 dimensions."
        )
    return data


def _read_mat_file_h5py(filename, mat_variable=None):
    """Load one MATLAB v7.3 dataset; ignore #refs# during auto-selection."""
    path = Path(filename).expanduser()
    try:
        with h5py.File(path, 'r') as file:
            dataset = _select_hdf5_dataset(
                file, mat_variable, selector_name='mat_variable',
                exclude_internal=True,
            )
            return dataset[()]
    except OSError as exc:
        raise ValueError(
            f"Could not open '{path}' as a MATLAB v7.3 HDF5 file: {exc}"
        ) from exc


def save_mat_data(data, fileName, varName):
    """
    Save a numpy array as .mat file.
    
    Inputs: 
    - data: numpy array
    - fileName: string with '.mat' termination
    - varName: string - name of the variable to appear on MATLAB
    """
    
    # Must verify that function has termination '.npy'
    io.savemat(fileName, {varName: data})


def circular_mask(center_y, center_x, radius):
    """
    Make a 2D boolean mask where 'True' corresponds to a circular region of 
    user-defined radius and position.
    
    Inputs:
        center_y: y-coordinate center of mask (int or float) 
        center_x: x-coordinate center of mask (int or float) 
        radius: radius of 'True'-valued values
    """ 
    
    y, x = np.ogrid[-center_y:radius*2-center_y, -center_x:radius*2-center_x]
    mask = x**2 + y**2 <= radius**2
    return mask


def make_mask(centers, r_mask, mask_dim=(128, 128), invert=False,):
    """
    Create a circular or annular mask around given center points.

    Generates a boolean mask of specified dimensions, marking pixels
    within a radius or between radii from the given center point(s).
    Optionally inverts the mask.

    Parameters
    ----------
    center : tuple or list of tuples
        A tuple (y, x) representing the center of the mask, or a list of such 
        tuples for multiple centers.
    r_mask : float or tuple of floats
        Radius of the mask. If a tuple (inner_radius, outer_radius) is provided, 
        an annular mask is created.
    mask_dim : tuple of ints, optional
        Dimensions of the mask (height, width). Default is (128, 128).
    invert : bool, optional
        If True, inverts the mask. Default is False.

    Returns
    -------
    mask : np.ndarray of bool
        The generated boolean mask.

    Notes
    -----
    The function assumes the origin (0, 0) is at the top-left corner.
    """

    mask = np.zeros(mask_dim, dtype=bool)
    centers = np.atleast_2d(np.array(centers))
    
    # Create vertical and horizontal vectors with indices
    y_grid, x_grid = np.ogrid[:mask_dim[0], :mask_dim[1]] 
    
    for center in centers:
        y, x = center
        
        # Get a 2D map of all combinations of distances
        dist_sq = (y_grid - y) ** 2 + (x_grid - x) ** 2
        
        if isinstance(r_mask, tuple):
            mask |= (r_mask[0]**2 <= dist_sq) & (dist_sq <= r_mask[1]**2)
        else:
            mask |= dist_sq <= r_mask**2

    if invert:
        mask = ~mask

    return mask

def anscombe_transform(array, inverse=False):
    """
    Function that will perform the forward Anscombe transform of input array
    to stabilize the variance of Poisson noise and make it a constant value.
    
    If 'inverse' is True, the inverse (unbiased) Anscombre transform is applied.
    
    We approximate the exact unbiased inverse transform using a closed-form
    expression. See Makitalo & Foi (2011); doi:10.1109/TIP.2011.2121085.
    """
    
    if not inverse:
        transformed_arr = 2*np.sqrt(array + 3/8)
        
    else:
        transformed_arr = (1/4)*array**2 + (1/4)*np.sqrt(3/2)*array**(-1) - \
                          (11/8)*array**(-2) + (5/8)*np.sqrt(3/2)*array**(-3) - 1/8
    
    return transformed_arr

def add_poisson_noise(array, counts):
    """Draw Poisson counts from the relative intensities of an array.

    Parameters
    ----------
    array : array_like
        Input intensities. Negative values are set to zero in a copy before
        normalizing the array; the original is not modified. The remaining
        intensities must have a positive sum.
    counts : float
        Expected total count across the array, distributed in proportion to
        its nonnegative intensities.

    Returns
    -------
    numpy.ndarray
        Random integer counts with the same shape as ``array``. The realized
        total generally differs from ``counts`` because it is Poisson sampled.
    """

    # Remove any negative numbers and normalize
    noisy_array = np.copy(array)
    noisy_array[noisy_array < 0] = 0
    noisy_array /= np.sum(noisy_array)
    
    # Apply Poisson noise
    noisy_array = np.random.default_rng().poisson(noisy_array * counts,) 
    
    return noisy_array

def get_surface_tilt_and_direction(surface, units='rad', show_results=True, figsize=(8, 5)):
    """
    Calculate the tilt magnitude and tilt axis (gradient direction) of a surface and optionally display the results.

    Parameters
    ----------
    surface : numpy.ndarray
        A 2D array of height values representing the surface.
    units : str, optional
        The unit of the tilt magnitude and direction. Can be 'rad' for radians or 'deg' for degrees.
        Default is 'rad'.
    show_results : bool, optional
        If True, plots the tilt magnitude and direction. Default is True.

    Returns
    -------
    tilt_magnitude : numpy.ndarray
        A 2D array where each value represents the angular magnitude of the gradient (tilt magnitude),
        in specified units (either radians or degrees).
    tilt_direction : numpy.ndarray
        A 2D array where each value represents the direction of the gradient, in specified units
        (either radians or degrees).


    Notes
    -----
    The tilt direction is calculated relative to the horizontal axis. Edge values for the tilt
    magnitude and direction are set based on the boundary conditions and may not accurately
    represent the actual tilt due to the lack of neighboring data.
    """
    
    # Compute gradients along x and y axes
    gy, gx = np.gradient(surface)
    
    # Calculate gradient magnitude and then calculate arctan of this magnitude
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    tilt_magnitude = np.arctan(gradient_magnitude)
    tilt_direction = np.arctan2(gy, gx) + np.pi
    
    # Convert tilt magnitude and direction to degrees if required
    if units == 'deg':
        tilt_magnitude = np.degrees(tilt_magnitude)
        tilt_direction = np.degrees(tilt_direction)
        unit_label = '(°)'
    else:
        unit_label = '(rad)'
    
    if show_results:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # First subplot
        im1 = axs[0].imshow(tilt_direction, cmap='hsv')
        axs[0].axis('off')
        axs[0].set_title('Azimuthal\nAngle', fontsize=8)
        cbar1 = fig.colorbar(im1, ax=axs[0],)
        cbar1.ax.set_title(r'$\phi$ '+unit_label, fontsize=8, pad=5)
        
        # Second subplot
        im2 = axs[1].imshow(tilt_magnitude, cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Elevation\nAngle', fontsize=8)
        cbar2 = fig.colorbar(im2, ax=axs[1],)
        cbar2.ax.set_title(r'$\theta$ '+unit_label, fontsize=8, pad=5)
        
        # Adjust layout to make the subplots fit well
        plt.tight_layout()
        plt.show()
    
    return tilt_direction, tilt_magnitude


def visualize_field_phase_amplitude(field, Amp='raw', scale=True, subplot=True):
    """
    Visualize the phase and amplitude of a field as an RGB image and optionally display a color wheel.

    Parameters
    ----------
    field : complex ndarray or tuple
        The input field. Can be a complex array or a tuple of (phase, magnitude).
    Amp : str, optional
        Control for amplitude visualization:
        'uniform' - use a uniform amplitude across the image,
        'log' - use logarithmic scaling of the amplitude,
        'raw' - use the raw amplitude values.
    scale : bool, optional
        If True, plots a color wheel with corresponding phase hues. Default is False.
    subplot : bool, optional
        If True, uses subplots to show the image and color wheel; otherwise, separate figures.

    Returns
    -------
    rgb_image : ndarray
        An array representing the RGB visualization of the input field.
    
    Notes
    -----
    This code is based on the MATLAB code written by [author] in [date].
    """
    
    if isinstance(field, tuple):
        phase, amplitude = field
    else:
        phase = np.angle(field)
        amplitude = np.abs(field)
    
    # Normalize amplitude
    amplitude = amplitude/np.max(amplitude)
    
    if Amp == 'uniform':
        amplitude = np.ones_like(amplitude)
    elif Amp == 'log':
        min_amplitude = np.min(amplitude[amplitude > 0])
        amplitude = np.log(amplitude / min_amplitude) / np.log(amplitude.max())
    elif Amp != 'raw':
        raise ValueError("Amp must be 'uniform', 'log', or 'raw'.")

    # Create RGB image
    rgb_image = np.zeros((*amplitude.shape, 3))
    rgb_image[..., 0] = 0.5 * (np.sin(phase) + 1) * amplitude  # Red
    rgb_image[..., 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amplitude  # Green
    rgb_image[..., 2] = 0.5 * (-np.sin(phase) + 1) * amplitude  # Blue

    if not subplot:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(rgb_image)
        plt.show()
    
    if scale:
        # Color wheel
        phase = np.linspace(0, 2 * np.pi, 256)
        r = 0.5 * (np.sin(phase) + 1)
        g = 0.5 * (np.sin(phase + np.pi / 2) + 1)
        b = 0.5 * (-np.sin(phase) + 1)
        colorwheel = np.stack([r, g, b], axis=1)
        warphase = ListedColormap(colorwheel)

        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        z = x + 1j * y
        mask = x**2 + y**2 > 1
        z[mask] = np.nan

        hue = np.angle(z)     
        idx = np.clip(((hue + np.pi) / (2 * np.pi) * 255).astype(int), 0, 255)
        color_wheel = colorwheel[idx]
        color_wheel = (color_wheel.T * np.abs(z)).T
        color_wheel = np.rot90(color_wheel, 2)
        color_wheel[np.isnan(color_wheel)] = 1.0
            
        if subplot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Unified figure creation
            axs[0].imshow(rgb_image)
            axs[0].axis('off')
            axs[1].imshow(color_wheel, origin='lower')
            axs[1].axis('off')
            sm = ScalarMappable(cmap=warphase)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[1], orientation='vertical', ticks=[0, 0.25, 0.5, 0.75, 1], fraction=0.046, pad=0.04)
            cbar.set_label('Hue', rotation=270, labelpad=15)
            cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(color_wheel, origin='lower')
            plt.axis('off')
            plt.show()


    return rgb_image

def _row_major_indices(shape):
    """Return row-major ``(y, x)`` traversal indices for a 2D grid."""
    y_size, x_size = shape
    return np.array(
        [(y_idx, x_idx) for y_idx in range(y_size) for x_idx in range(x_size)],
        dtype=int,
    )


def _serpentine_indices(shape):
    """
    Return left-to-right/right-to-left alternating row traversal indices.

    This keeps consecutive elements adjacent when crossing from one row to the
    next, which is useful for scan paths that physically snake through the grid.
    """
    y_size, x_size = shape
    indices = []
    for y_idx in range(y_size):
        if y_idx % 2 == 0:
            x_range = range(x_size)
        else:
            x_range = range(x_size - 1, -1, -1)
        indices.extend((y_idx, x_idx) for x_idx in x_range)
    return np.array(indices, dtype=int)


def _spiral_indices(shape):
    """
    Return top-left clockwise inward spiral traversal indices for a 2D grid.

    The path starts across the top row, moves down the right column, then left
    across the bottom row, up the left column, and repeats inward.
    """
    y_size, x_size = shape
    top = 0
    bottom = y_size - 1
    left = 0
    right = x_size - 1
    indices = []

    while top <= bottom and left <= right:
        for x_idx in range(left, right + 1):
            indices.append((top, x_idx))
        top += 1

        for y_idx in range(top, bottom + 1):
            indices.append((y_idx, right))
        right -= 1

        if top <= bottom:
            for x_idx in range(right, left - 1, -1):
                indices.append((bottom, x_idx))
            bottom -= 1

        if left <= right:
            for y_idx in range(bottom, top - 1, -1):
                indices.append((y_idx, left))
            left += 1

    return np.array(indices, dtype=int)


def _diagonal_zigzag_indices(shape):
    """
    Traverse anti-diagonals while alternating direction on each diagonal.

    This is not a fractal space-filling curve, but it is a simple
    locality-preserving baseline used in image compression and scanning.
    """
    y_size, x_size = shape
    indices = []
    for diag in range(y_size + x_size - 1):
        y_start = max(0, diag - x_size + 1)
        y_stop = min(y_size - 1, diag)
        diagonal = [(y_idx, diag - y_idx) for y_idx in range(y_start, y_stop + 1)]
        if diag % 2 == 0:
            diagonal.reverse()
        indices.extend(diagonal)
    return np.array(indices, dtype=int)


def _hilbert_d2yx(side, distance):
    """
    Convert Hilbert distance to ``(y, x)`` for a power-of-2 square.

    This is the standard iterative d2xy Hilbert algorithm. It is written in
    ``(x, y)`` internally and returned as ``(y, x)`` to match array indexing.
    """
    x_coord = 0
    y_coord = 0
    t_val = int(distance)
    scale = 1

    while scale < side:
        rx = 1 & (t_val // 2)
        ry = 1 & (t_val ^ rx)
        if ry == 0:
            if rx == 1:
                x_coord = scale - 1 - x_coord
                y_coord = scale - 1 - y_coord
            x_coord, y_coord = y_coord, x_coord

        x_coord += scale * rx
        y_coord += scale * ry
        t_val //= 4
        scale *= 2

    return y_coord, x_coord


def _hilbert_indices(shape):
    """Return a discrete Hilbert traversal for a power-of-2 square."""
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 2):
        raise ValueError("Hilbert traversal requires a power-of-2 square.")
    return np.array(
        [_hilbert_d2yx(side_y, distance) for distance in range(side_y * side_y)],
        dtype=int,
    )


def _morton_code(y_idx, x_idx):
    """Interleave the binary bits of ``y`` and ``x`` into a Morton code."""
    code = 0
    bit = 0
    max_val = max(int(y_idx), int(x_idx))
    while (1 << bit) <= max_val:
        code |= ((x_idx >> bit) & 1) << (2 * bit)
        code |= ((y_idx >> bit) & 1) << (2 * bit + 1)
        bit += 1
    return code


def _morton_indices(shape):
    """
    Return Morton/Z-order traversal for a power-of-2 square.

    Coordinates are sorted by interleaving the binary bits of row and column
    indices. This is less continuous than Hilbert, but useful as a hierarchical
    locality-preserving baseline.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 2):
        raise ValueError("Morton/Z-order traversal requires a power-of-2 square.")
    coords = [(y_idx, x_idx) for y_idx in range(side_y) for x_idx in range(side_x)]
    coords.sort(key=lambda coord: _morton_code(coord[0], coord[1]))
    return np.array(coords, dtype=int)


def _peano_indices(shape):
    """
    Return a recursive Peano traversal for a power-of-3 square.

    The first-order path traverses a 3x3 square in a ``2``-shaped motif. Higher
    orders recursively orient each child block so adjacent children connect by a
    nearest-neighbor step. This alternates ``2``- and ``S``-like motifs across
    the grid, avoiding the disconnected repeated-block pattern that a naive
    serpentine recursion would produce.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 3):
        raise ValueError("Peano traversal requires a power-of-3 square.")

    order = 0
    side = side_y
    while side > 1:
        side //= 3
        order += 1

    return np.array(_peano_path(order), dtype=int)


def _opposite_corner(corner):
    """Return the diagonally opposite corner name for a square block."""
    opposites = {
        'NW': 'SE',
        'SE': 'NW',
        'NE': 'SW',
        'SW': 'NE',
    }
    return opposites[corner]


def _peano_base_path(start_corner='NW', end_corner='SE'):
    """
    Return an oriented 3x3 Peano base path between opposite corners.

    The canonical orientation starts at ``NW`` and ends at ``SE``:
    right, right, down, left, left, down, right, right. Reflections provide the
    other opposite-corner orientations while preserving nearest-neighbor steps.
    """
    if _opposite_corner(start_corner) != end_corner:
        raise ValueError(
            "Peano child curves require opposite start and end corners."
        )

    path = np.array(
        [
            (0, 0), (0, 1), (0, 2),
            (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2),
        ],
        dtype=int,
    )

    if start_corner in {'SW', 'SE'}:
        path[:, 0] = 2 - path[:, 0]
    if start_corner in {'NE', 'SE'}:
        path[:, 1] = 2 - path[:, 1]

    return path


def _peano_next_entry_corner(direction, exit_corner):
    """Return the child entry corner compatible with a neighboring block step."""
    compatible_entries = {
        (0, 1): {'NE': 'NW', 'SE': 'SW'},
        (0, -1): {'NW': 'NE', 'SW': 'SE'},
        (1, 0): {'SW': 'NW', 'SE': 'NE'},
        (-1, 0): {'NW': 'SW', 'NE': 'SE'},
    }
    direction = tuple(int(v) for v in direction)
    if direction not in compatible_entries:
        raise ValueError(f"Invalid Peano block direction {direction}.")
    if exit_corner not in compatible_entries[direction]:
        raise ValueError(
            f"Peano child exit corner '{exit_corner}' is incompatible with "
            f"block direction {direction}."
        )
    return compatible_entries[direction][exit_corner]


def _peano_path(order, start_corner='NW', end_corner='SE'):
    """Return recursive Peano coordinates for a ``3**order`` square."""
    if order < 0:
        raise ValueError("order must be non-negative.")
    if order == 0:
        return ((0, 0),)

    macro_path = _peano_base_path(start_corner, end_corner)
    if order == 1:
        return tuple(map(tuple, macro_path))

    sub_side = 3 ** (order - 1)
    coords = []
    child_start = start_corner

    for block_idx, block in enumerate(macro_path):
        if block_idx < len(macro_path) - 1:
            direction = macro_path[block_idx + 1] - block
            child_end = _opposite_corner(child_start)
            next_child_start = _peano_next_entry_corner(direction, child_end)
        else:
            child_end = end_corner
            next_child_start = None
            if _opposite_corner(child_start) != child_end:
                raise RuntimeError(
                    "Could not orient the final Peano child block continuously."
                )

        child = _peano_path(order - 1, child_start, child_end)
        offset = block * sub_side
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in child
        )
        child_start = next_child_start

    return tuple(coords)


def _corner_coordinate(corner, side):
    """Return the local ``(y, x)`` coordinate of a named square corner."""
    corners = {
        'NW': (0, 0),
        'NE': (0, side - 1),
        'SW': (side - 1, 0),
        'SE': (side - 1, side - 1),
    }
    return corners[corner]


def _compatible_corner_pairs(direction):
    """Return adjacent exit/entry corner pairs for neighboring 3x3 blocks."""
    if direction == (0, 1):
        return [('NE', 'NW'), ('SE', 'SW')]
    if direction == (0, -1):
        return [('NW', 'NE'), ('SW', 'SE')]
    if direction == (1, 0):
        return [('SW', 'NW'), ('SE', 'NE')]
    if direction == (-1, 0):
        return [('NW', 'SW'), ('NE', 'SE')]
    raise ValueError(f"Invalid block direction {direction}.")


def _base_meander_path(start_corner, end_corner):
    """Find a 3x3 Hamiltonian path between two square corners."""
    start = _corner_coordinate(start_corner, 3)
    end = _corner_coordinate(end_corner, 3)
    if start == end:
        raise ValueError("start_corner and end_corner must differ.")

    path = [start]
    visited = {start}

    def neighbor_score(cell):
        y_idx, x_idx = cell
        onward = 0
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (y_idx + dy, x_idx + dx)
            if (
                0 <= neighbor[0] < 3
                and 0 <= neighbor[1] < 3
                and neighbor not in visited
            ):
                onward += 1
        distance_to_end = abs(y_idx - end[0]) + abs(x_idx - end[1])
        return onward, distance_to_end

    def dfs(cell):
        if len(path) == 9:
            return cell == end

        y_idx, x_idx = cell
        candidates = []
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (y_idx + dy, x_idx + dx)
            if 0 <= neighbor[0] < 3 and 0 <= neighbor[1] < 3:
                if neighbor in visited:
                    continue
                if neighbor == end and len(path) != 8:
                    continue
                candidates.append(neighbor)

        candidates.sort(key=neighbor_score)
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            if dfs(neighbor):
                return True
            path.pop()
            visited.remove(neighbor)
        return False

    if not dfs(start):
        raise RuntimeError(
            f"Could not construct 3x3 meander path from {start_corner} to {end_corner}."
        )
    return tuple(path)


def _meander_block_template(start_corner, end_corner):
    """
    Build a 3x3 block traversal with corner-aware child orientations.

    This alternates local 2-like and S-like motifs so neighboring child curves
    connect continuously across block boundaries.
    """
    corner_blocks = {
        'NW': (0, 0),
        'NE': (0, 2),
        'SW': (2, 0),
        'SE': (2, 2),
    }
    start_block = corner_blocks[start_corner]
    end_block = corner_blocks[end_corner]
    path = [(start_block, start_corner, None)]
    visited = {start_block}

    def ordered_neighbors(block):
        y_idx, x_idx = block
        if y_idx % 2 == 0:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        for direction in directions:
            neighbor = (y_idx + direction[0], x_idx + direction[1])
            if 0 <= neighbor[0] < 3 and 0 <= neighbor[1] < 3:
                yield neighbor, direction

    def dfs(current_block, current_start):
        if len(path) == 9:
            if current_block != end_block or current_start == end_corner:
                return False
            path[-1] = (current_block, current_start, end_corner)
            return True

        for next_block, direction in ordered_neighbors(current_block):
            if next_block in visited:
                continue
            remaining = 9 - len(path)
            if next_block == end_block and remaining != 1:
                continue

            for exit_corner, entry_corner in _compatible_corner_pairs(direction):
                if exit_corner == current_start:
                    continue
                path[-1] = (current_block, current_start, exit_corner)
                visited.add(next_block)
                path.append((next_block, entry_corner, None))
                if dfs(next_block, entry_corner):
                    return True
                path.pop()
                visited.remove(next_block)
                path[-1] = (current_block, current_start, None)

        return False

    if not dfs(start_block, start_corner):
        raise RuntimeError(
            f"Could not construct meander block from {start_corner} to {end_corner}."
        )
    return tuple(path)


def _peano_meander_path(order, start_corner='NW', end_corner='SE'):
    """
    Return a recursive Peano-meander path for ``3**order`` square grids.

    This is not a mathematically strict Peano curve. It is a Peano-like
    ternary meander that fills powers-of-3 grids with continuous nearest-neighbor
    steps while alternating child-curve orientation for better block-to-block
    continuity.
    """
    if order == 1:
        return _base_meander_path(start_corner, end_corner)

    sub_side = 3 ** (order - 1)
    coords = []
    for (block_y, block_x), child_start, child_end in _meander_block_template(
        start_corner,
        end_corner,
    ):
        child = _peano_meander_path(order - 1, child_start, child_end)
        coords.extend(
            (y_idx + block_y * sub_side, x_idx + block_x * sub_side)
            for y_idx, x_idx in child
        )
    return tuple(coords)


def _peano_meander_indices(shape):
    """
    Return a Peano-like ternary meander traversal for a power-of-3 square.

    This curve is useful when a continuous power-of-3 locality-preserving path is
    desired, while avoiding the repeated local motif in the simpler Peano-style
    implementation.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 3):
        raise ValueError("Peano-meander traversal requires a power-of-3 square.")

    order = 0
    side = side_y
    while side > 1:
        side //= 3
        order += 1

    return np.array(_peano_meander_path(order), dtype=int)


_MEANDER4_BLOCK_SIZE = 4
_MEANDER4_CORNERS = {
    'NW': (0, 0),
    'NE': (0, _MEANDER4_BLOCK_SIZE - 1),
    'SW': (_MEANDER4_BLOCK_SIZE - 1, 0),
    'SE': (_MEANDER4_BLOCK_SIZE - 1, _MEANDER4_BLOCK_SIZE - 1),
}
_MEANDER4_SIDE_EXIT_CORNERS = {
    'N': ('NW', 'NE'),
    'E': ('NE', 'SE'),
    'S': ('SW', 'SE'),
    'W': ('NW', 'SW'),
}
_MEANDER4_DIRECTION_TO_SIDE = {
    (0, 1): 'E',
    (0, -1): 'W',
    (1, 0): 'S',
    (-1, 0): 'N',
}
_MEANDER4_NEXT_ENTRY = {
    ((0, 1), 'NE'): 'NW',
    ((0, 1), 'SE'): 'SW',
    ((0, -1), 'NW'): 'NE',
    ((0, -1), 'SW'): 'SE',
    ((1, 0), 'SW'): 'NW',
    ((1, 0), 'SE'): 'NE',
    ((-1, 0), 'NW'): 'SW',
    ((-1, 0), 'NE'): 'SE',
}
_MEANDER4_BASE_NW_NE = np.array(
    [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (3, 1), (3, 2), (3, 3),
        (2, 3), (1, 3),
        (1, 2),
        (2, 2),
        (2, 1),
        (1, 1), (0, 1),
        (0, 2), (0, 3),
    ],
    dtype=int,
)


def _meander4_transform_path(path, transform_name):
    """Apply a square symmetry transform to a 4x4 meander tile path."""
    size = _MEANDER4_BLOCK_SIZE
    y_idx = path[:, 0]
    x_idx = path[:, 1]

    if transform_name == 'identity':
        transformed = np.column_stack([y_idx, x_idx])
    elif transform_name == 'rot90':
        transformed = np.column_stack([x_idx, size - 1 - y_idx])
    elif transform_name == 'rot180':
        transformed = np.column_stack([size - 1 - y_idx, size - 1 - x_idx])
    elif transform_name == 'rot270':
        transformed = np.column_stack([size - 1 - x_idx, y_idx])
    elif transform_name == 'flip_y':
        transformed = np.column_stack([size - 1 - y_idx, x_idx])
    elif transform_name == 'flip_x':
        transformed = np.column_stack([y_idx, size - 1 - x_idx])
    elif transform_name == 'diag':
        transformed = np.column_stack([x_idx, y_idx])
    elif transform_name == 'anti_diag':
        transformed = np.column_stack([size - 1 - x_idx, size - 1 - y_idx])
    else:
        raise ValueError(f"Unknown meander-4 transform '{transform_name}'.")

    return transformed.astype(int)


def _meander4_corner_name(coord):
    """Return the named 4x4 block corner at ``coord``."""
    coord = tuple(int(v) for v in coord)
    for name, corner in _MEANDER4_CORNERS.items():
        if coord == corner:
            return name
    return None


def _meander4_tile_paths():
    """
    Return oriented 4x4 meander tile paths between adjacent corners.

    The canonical tile is the user-specified path from ``NW`` to ``NE``:
    down x3, right x3, up x2, left, down, left, up x2, right x2. Direct square
    symmetries generate the remaining orientations while preserving that motif.
    """
    transforms = (
        'identity',
        'rot90',
        'rot180',
        'rot270',
        'flip_y',
        'flip_x',
        'diag',
        'anti_diag',
    )
    tile_paths = {}
    for transform_name in transforms:
        path = _meander4_transform_path(_MEANDER4_BASE_NW_NE, transform_name)
        start_corner = _meander4_corner_name(path[0])
        end_corner = _meander4_corner_name(path[-1])
        tile_paths[(start_corner, end_corner)] = path
    return tile_paths


_MEANDER4_TILE_PATHS = _meander4_tile_paths()


def _meander4_corners_are_adjacent(first_corner, second_corner):
    """Return True when two 4x4 block corners share a block side."""
    first_y, first_x = _MEANDER4_CORNERS[first_corner]
    second_y, second_x = _MEANDER4_CORNERS[second_corner]
    distance = abs(first_y - second_y) + abs(first_x - second_x)
    return distance == _MEANDER4_BLOCK_SIZE - 1


def _meander4_choose_exit_corner(entry_corner, direction):
    """Choose the exit corner for a block-to-block step."""
    side = _MEANDER4_DIRECTION_TO_SIDE[direction]
    for exit_corner in _MEANDER4_SIDE_EXIT_CORNERS[side]:
        if (
            exit_corner != entry_corner
            and _meander4_corners_are_adjacent(entry_corner, exit_corner)
        ):
            return exit_corner
    raise ValueError(
        f"No meander-4 exit corner for entry='{entry_corner}', direction={direction}."
    )


def _meander4_indices(shape):
    """
    Return a 4x4 Greek-key meander traversal for dimensions divisible by 4.

    The array is divided into 4x4 tiles. The first tile follows the exact local
    step pattern ``down x3, right x3, up x2, left, down, left, up x2, right x2``.
    Tiles are then visited along an inward spiral, with each tile orientation
    chosen so neighboring tiles connect by a one-pixel step. Rectangular
    ``4*Ny`` by ``4*Nx`` grids are supported.
    """
    height, width = tuple(int(v) for v in shape)
    if height <= 0 or width <= 0:
        raise ValueError("meander-4 traversal shape must be positive.")
    if height % _MEANDER4_BLOCK_SIZE != 0 or width % _MEANDER4_BLOCK_SIZE != 0:
        raise ValueError(
            "meander-4 traversal requires both dimensions to be multiples of 4."
        )

    block_rows = height // _MEANDER4_BLOCK_SIZE
    block_cols = width // _MEANDER4_BLOCK_SIZE
    block_order = [tuple(coord) for coord in _spiral_indices((block_rows, block_cols))]
    coords = []
    entry_corner = 'NW'

    for block_idx, (block_y, block_x) in enumerate(block_order):
        if block_idx < len(block_order) - 1:
            next_block = block_order[block_idx + 1]
            direction_to_next = (
                next_block[0] - block_y,
                next_block[1] - block_x,
            )
            exit_corner = _meander4_choose_exit_corner(
                entry_corner,
                direction_to_next,
            )
            next_entry_corner = _MEANDER4_NEXT_ENTRY[
                (direction_to_next, exit_corner)
            ]
        else:
            for candidate_corner in ('NE', 'SE', 'SW', 'NW'):
                if (
                    candidate_corner != entry_corner
                    and _meander4_corners_are_adjacent(
                        entry_corner,
                        candidate_corner,
                    )
                ):
                    exit_corner = candidate_corner
                    break
            next_entry_corner = None

        local_path = _MEANDER4_TILE_PATHS[(entry_corner, exit_corner)]
        offset = np.array(
            [
                block_y * _MEANDER4_BLOCK_SIZE,
                block_x * _MEANDER4_BLOCK_SIZE,
            ]
        )
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in local_path
        )
        entry_corner = next_entry_corner

    return np.array(coords, dtype=int)


_MEANDER5_BLOCK_SIZE = 5
_MEANDER5_SIDE_COORDS = {
    'N': (0, 2),
    'E': (2, 4),
    'S': (4, 2),
    'W': (2, 0),
    'C': (2, 2),
}
_MEANDER5_DIRECTION_TO_SIDE = {
    (0, 1): 'E',
    (0, -1): 'W',
    (1, 0): 'S',
    (-1, 0): 'N',
}
_MEANDER5_OPPOSITE_SIDE = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
}
_MEANDER5_STEP_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_MEANDER5_SPIRAL_RANK = {
    tuple(coord): idx
    for idx, coord in enumerate(_spiral_indices((_MEANDER5_BLOCK_SIZE, _MEANDER5_BLOCK_SIZE)))
}


@lru_cache(maxsize=None)
def _meander5_block_path(entry_side, exit_side):
    """
    Return a 5x5 Greek-key block traversal between two block endpoints.

    The endpoints lie on side midpoints, or on the center for the final block.
    A small depth-first Hamiltonian search is used once per entry/exit pair and
    cached; the scoring favors perimeter-to-interior squared turns, giving the
    Greek-key hook motif while preserving one-pixel steps.
    """
    if entry_side not in _MEANDER5_SIDE_COORDS:
        raise ValueError(f"Invalid meander-5 entry side '{entry_side}'.")
    if exit_side not in _MEANDER5_SIDE_COORDS:
        raise ValueError(f"Invalid meander-5 exit side '{exit_side}'.")

    block_size = _MEANDER5_BLOCK_SIZE
    start = _MEANDER5_SIDE_COORDS[entry_side]
    end = _MEANDER5_SIDE_COORDS[exit_side]
    if start == end:
        raise ValueError("meander-5 entry and exit endpoints must differ.")

    total = block_size * block_size
    visited = {start}
    path = [start]

    def connectivity_ok():
        remaining = {
            (y_idx, x_idx)
            for y_idx in range(block_size)
            for x_idx in range(block_size)
            if (y_idx, x_idx) not in visited
        }
        if not remaining:
            return True

        stack = [next(iter(remaining))]
        seen = {stack[0]}
        while stack:
            y_idx, x_idx = stack.pop()
            for dy, dx in _MEANDER5_STEP_DIRECTIONS:
                neighbor = (y_idx + dy, x_idx + dx)
                if neighbor in remaining and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(remaining)

    def onward_count(cell):
        y_idx, x_idx = cell
        count = 0
        for dy, dx in _MEANDER5_STEP_DIRECTIONS:
            neighbor = (y_idx + dy, x_idx + dx)
            if (
                0 <= neighbor[0] < block_size
                and 0 <= neighbor[1] < block_size
                and neighbor not in visited
            ):
                if neighbor == end and len(path) != total - 1:
                    continue
                count += 1
        return count

    def neighbor_score(cell, previous):
        py, px = previous
        cy, cx = cell
        straight_penalty = 0
        if len(path) >= 2:
            ay, ax = path[-2]
            old_direction = (py - ay, px - ax)
            new_direction = (cy - py, cx - px)
            straight_penalty = int(old_direction == new_direction)

        return (
            onward_count(cell),
            _MEANDER5_SPIRAL_RANK[cell],
            straight_penalty,
            abs(cy - end[0]) + abs(cx - end[1]),
        )

    def dfs(cell):
        if len(path) == total:
            return cell == end

        y_idx, x_idx = cell
        candidates = []
        for dy, dx in _MEANDER5_STEP_DIRECTIONS:
            neighbor = (y_idx + dy, x_idx + dx)
            if not (0 <= neighbor[0] < block_size and 0 <= neighbor[1] < block_size):
                continue
            if neighbor in visited:
                continue
            if neighbor == end and len(path) != total - 1:
                continue
            candidates.append(neighbor)

        candidates.sort(key=lambda neighbor: neighbor_score(neighbor, cell))
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            if connectivity_ok() and dfs(neighbor):
                return True
            path.pop()
            visited.remove(neighbor)
        return False

    if not dfs(start):
        raise RuntimeError(
            f"Could not construct meander-5 block path {entry_side}->{exit_side}."
        )
    return tuple(path)


def _meander5_indices(shape):
    """
    Return a Greek-key meander traversal for dimensions divisible by 5.

    The array is divided into 5x5 tiles. Tiles are visited along an inward
    spiral, and each tile is filled by a small squared Greek-key hook whose
    entry and exit sides are oriented to connect continuously to neighboring
    tiles. Rectangular ``5*Ny`` by ``5*Nx`` grids are supported.
    """
    height, width = tuple(int(v) for v in shape)
    if height <= 0 or width <= 0:
        raise ValueError("meander-5 traversal shape must be positive.")
    if height % _MEANDER5_BLOCK_SIZE != 0 or width % _MEANDER5_BLOCK_SIZE != 0:
        raise ValueError(
            "meander-5 traversal requires both dimensions to be multiples of 5."
        )

    block_rows = height // _MEANDER5_BLOCK_SIZE
    block_cols = width // _MEANDER5_BLOCK_SIZE
    block_order = [tuple(coord) for coord in _spiral_indices((block_rows, block_cols))]
    coords = []

    for block_idx, (block_y, block_x) in enumerate(block_order):
        if len(block_order) == 1:
            entry_side = 'W'
            exit_side = 'C'
        elif block_idx == 0:
            next_block = block_order[block_idx + 1]
            direction_to_next = (
                next_block[0] - block_y,
                next_block[1] - block_x,
            )
            exit_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_next]
            entry_side = _MEANDER5_OPPOSITE_SIDE[exit_side]
        else:
            previous_block = block_order[block_idx - 1]
            direction_to_previous = (
                previous_block[0] - block_y,
                previous_block[1] - block_x,
            )
            entry_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_previous]
            if block_idx < len(block_order) - 1:
                next_block = block_order[block_idx + 1]
                direction_to_next = (
                    next_block[0] - block_y,
                    next_block[1] - block_x,
                )
                exit_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_next]
            else:
                exit_side = 'C'

        local_path = _meander5_block_path(entry_side, exit_side)
        offset = np.array(
            [
                block_y * _MEANDER5_BLOCK_SIZE,
                block_x * _MEANDER5_BLOCK_SIZE,
            ]
        )
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in local_path
        )

    return np.array(coords, dtype=int)


def _moore_indices(shape):
    """
    Placeholder for a closed Hilbert-like Moore traversal.

    Moore curves are useful, but implementing a clear closed discrete variant is
    a separate step from the current unfolding refactor.
    """
    raise NotImplementedError(
        "method='moore' is registered but not implemented yet. "
        "Use method='hilbert' for an open Hilbert traversal."
    )


_FULL_SHAPE_TRAVERSAL_METHODS = {
    'row_major',
    'serpentine',
    'spiral',
    'diagonal_zigzag',
}

_CURVE_TRAVERSAL_METHODS = {
    'hilbert',
    'morton',
    'peano',
    'peano_meander',
    'moore',
}

_BLOCK_TRAVERSAL_METHODS = {
    'meander-4',
    'meander-5',
}

_TRAVERSAL_INDEX_GENERATORS = {
    'row_major': _row_major_indices,
    'serpentine': _serpentine_indices,
    'spiral': _spiral_indices,
    'diagonal_zigzag': _diagonal_zigzag_indices,
    'hilbert': _hilbert_indices,
    'morton': _morton_indices,
    'peano': _peano_indices,
    'peano_meander': _peano_meander_indices,
    'meander-4': _meander4_indices,
    'meander-5': _meander5_indices,
    'moore': _moore_indices,
}

_TRAVERSAL_METHOD_ALIASES = {
    'z_order': 'morton',
    'meander_4': 'meander-4',
    'meander_5': 'meander-5',
}


def _is_power_of(value, base):
    """Return True when ``value`` is an integer power of ``base``."""
    value = int(value)
    if value < 1:
        return False
    while value % base == 0:
        value //= base
    return value == 1


def _largest_power_leq(value, base):
    """Return the largest power of ``base`` less than or equal to ``value``."""
    value = int(value)
    if value < 1:
        raise ValueError("value must be positive.")
    power = 1
    while power * base <= value:
        power *= base
    return power


def _largest_multiple_leq(value, factor):
    """Return the largest positive multiple of ``factor`` not exceeding ``value``."""
    value = int(value)
    factor = int(factor)
    if value < factor:
        raise ValueError(
            f"value must be at least {factor} for this traversal method."
        )
    return (value // factor) * factor


def _smallest_power_geq(value, base):
    """Return the smallest power of ``base`` greater than or equal to ``value``."""
    value = int(value)
    if value < 1:
        raise ValueError("value must be positive.")
    power = 1
    while power < value:
        power *= base
    return power


def _nearest_power(value, base):
    """
    Return the nearest power of ``base`` to ``value``.

    Ties choose the smaller side to avoid introducing interpolated data unless
    the user explicitly requests upsampling.
    """
    lower = _largest_power_leq(value, base)
    upper = _smallest_power_geq(value, base)
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def _curve_base_for_method(method):
    """Return the natural integer base for a square-compatible traversal."""
    if method == 'peano':
        return 3
    if method == 'peano_meander':
        return 3
    return 2


def _block_size_for_method(method):
    """Return the required tile size for block-compatible traversals."""
    if method == 'meander-4':
        return _MEANDER4_BLOCK_SIZE
    if method == 'meander-5':
        return _MEANDER5_BLOCK_SIZE
    raise ValueError(f"method='{method}' is not a block traversal method.")


def _normalize_traversal_method(method):
    """Normalize method aliases such as ``z_order`` to their implementation."""
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    method = method.lower()
    return _TRAVERSAL_METHOD_ALIASES.get(method, method)


def _validate_unfold_shape(shape):
    """Validate and normalize a 4D-STEM tensor shape."""
    if shape is None:
        raise ValueError("original_shape is required to undo an unfolding.")
    if len(shape) != 4:
        raise ValueError(
            "Unfolding expects a 4D-STEM shape in the convention "
            "(Ry, Rx, Ky, Kx)."
        )
    shape = tuple(int(v) for v in shape)
    if any(v <= 0 for v in shape):
        raise ValueError("All original_shape dimensions must be positive.")
    return shape


def _normalize_unfold_request(domain='real', method='row_major'):
    """Validate the modern unfolding API: separate domain and method names."""
    if domain is None:
        domain = 'real'
    if method is None:
        method = 'row_major'
    if not isinstance(domain, str):
        raise ValueError("domain must be a string.")

    domain = domain.lower()
    method = _normalize_traversal_method(method)
    if domain not in ('real', 'reciprocal', 'both'):
        raise ValueError("domain must be one of 'real', 'reciprocal', or 'both'.")

    valid_methods = set(_TRAVERSAL_INDEX_GENERATORS) | {'coordinate_aligned'}
    if method not in valid_methods:
        valid = ', '.join(sorted(valid_methods | set(_TRAVERSAL_METHOD_ALIASES)))
        raise ValueError(f"method must be one of: {valid}.")
    if method == 'coordinate_aligned' and domain == 'both':
        raise ValueError("method='coordinate_aligned' supports domain='real' or 'reciprocal', not 'both'.")
    if domain == 'both' and method not in {'row_major', 'morton'}:
        raise NotImplementedError(
            "domain='both' currently supports method='row_major' and "
            "method='morton'. Other traversal pairings need an explicit design."
        )

    return domain, method


def _validate_resize_side(method, resize_side):
    """Validate a user-provided side length for a curve method."""
    resize_side = int(resize_side)
    base = _curve_base_for_method(method)
    if resize_side <= 0 or not _is_power_of(resize_side, base):
        raise ValueError(
            f"resize_side must be a positive power of {base} for method='{method}'."
        )
    return resize_side


def _select_resize_side(shape, method, resize_side=None, resize_side_mode='nearest'):
    """Select a compatible square side for curve resize mode."""
    if resize_side is not None:
        return _validate_resize_side(method, resize_side)

    resize_side_mode = resize_side_mode.lower()
    if resize_side_mode not in ('nearest', 'downsample', 'upsample'):
        raise ValueError("resize_side_mode must be 'nearest', 'downsample', or 'upsample'.")

    target = min(shape)
    base = _curve_base_for_method(method)
    if resize_side_mode == 'nearest':
        return _nearest_power(target, base)
    if resize_side_mode == 'downsample':
        return _largest_power_leq(target, base)
    return _smallest_power_geq(target, base)


def _center_crop_metadata(shape, method):
    """Return centered compatible crop metadata for special traversals."""
    height, width = tuple(int(v) for v in shape)
    if method in _BLOCK_TRAVERSAL_METHODS:
        block_size = _block_size_for_method(method)
        grid_y = _largest_multiple_leq(height, block_size)
        grid_x = _largest_multiple_leq(width, block_size)
    else:
        side = _largest_power_leq(min(height, width), _curve_base_for_method(method))
        grid_y = side
        grid_x = side

    y0 = (height - grid_y) // 2
    x0 = (width - grid_x) // 2
    y1 = y0 + grid_y
    x1 = x0 + grid_x
    return (grid_y, grid_x), y0, y1, x0, x1


def _excess_indices_for_crop(shape, y0, y1, x0, x1):
    """Return coordinates outside a centered compatible square crop."""
    height, width = shape
    excess = [
        (y_idx, x_idx)
        for y_idx in range(height)
        for x_idx in range(width)
        if not (y0 <= y_idx < y1 and x0 <= x_idx < x1)
    ]
    return np.array(excess, dtype=int).reshape((-1, 2))


def _validate_traversal_indices(indices, shape, method):
    """Ensure traversal coordinates are unique and within ``shape``."""
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"Traversal method '{method}' must return an (N, 2) array.")
    if indices.size == 0:
        raise ValueError(f"Traversal method '{method}' returned no coordinates.")

    height, width = shape
    if np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= height):
        raise ValueError(f"Traversal method '{method}' returned out-of-bounds y coordinates.")
    if np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= width):
        raise ValueError(f"Traversal method '{method}' returned out-of-bounds x coordinates.")

    if len(set(map(tuple, indices))) != len(indices):
        raise ValueError(f"Traversal method '{method}' returned duplicate coordinates.")
    return indices


def _get_traversal_indices(shape, method, curve_shape_strategy='center_crop',
                           resize_side=None, resize_side_mode='nearest'):
    """
    Return traversal indices and metadata for a 2D coordinate grid.

    Full-shape methods visit every coordinate in ``shape``. Curve methods use
    either a centered compatible square crop or a resized compatible square.
    Block methods such as ``meander-4`` and ``meander-5`` use a centered
    compatible rectangular crop whose dimensions are multiples of the required
    block size.
    """
    method = _normalize_traversal_method(method)
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or any(v <= 0 for v in shape):
        raise ValueError("Traversal shape must be a pair of positive integers.")
    if method not in _TRAVERSAL_INDEX_GENERATORS:
        valid = ', '.join(sorted(set(_TRAVERSAL_INDEX_GENERATORS) | set(_TRAVERSAL_METHOD_ALIASES)))
        raise ValueError(f"method='{method}' is not a traversal method. Valid methods are: {valid}.")

    if method in _FULL_SHAPE_TRAVERSAL_METHODS:
        indices = np.asarray(_TRAVERSAL_INDEX_GENERATORS[method](shape), dtype=int)
        expected_size = shape[0] * shape[1]
        if indices.shape != (expected_size, 2):
            raise ValueError(
                f"Traversal method '{method}' returned shape {indices.shape}; "
                f"expected {(expected_size, 2)}."
            )
        indices = _validate_traversal_indices(indices, shape, method)
        return indices, {
            'curve_shape_strategy': 'full_shape',
            'traversal_shape': shape,
        }

    if method == 'moore':
        _moore_indices(shape)

    special_methods = _CURVE_TRAVERSAL_METHODS | _BLOCK_TRAVERSAL_METHODS
    if method not in special_methods:
        raise ValueError(f"Unknown traversal method '{method}'.")

    if curve_shape_strategy is None:
        curve_shape_strategy = 'center_crop'
    if not isinstance(curve_shape_strategy, str):
        raise ValueError("curve_shape_strategy must be a string.")
    curve_shape_strategy = curve_shape_strategy.lower()

    if curve_shape_strategy == 'center_crop':
        grid_shape, y0, y1, x0, x1 = _center_crop_metadata(shape, method)
        local_indices = np.asarray(
            _TRAVERSAL_INDEX_GENERATORS[method](grid_shape),
            dtype=int,
        )
        if local_indices.shape != (grid_shape[0] * grid_shape[1], 2):
            raise ValueError(
                f"Traversal method '{method}' did not cover its selected "
                f"{grid_shape} grid exactly once."
            )
        indices = local_indices + np.array([y0, x0])
        indices = _validate_traversal_indices(indices, shape, method)
        traversal_metadata = {
            'curve_shape_strategy': 'center_crop',
            'curve_grid_shape': grid_shape,
            'crop_slices': {'y': (y0, y1), 'x': (x0, x1)},
            'kept_indices': indices,
            'excess_indices': _excess_indices_for_crop(shape, y0, y1, x0, x1),
            'traversal_shape': shape,
        }
        if method in _BLOCK_TRAVERSAL_METHODS:
            block_size = _block_size_for_method(method)
            traversal_metadata.update({
                'block_size': block_size,
                'block_grid_shape': (
                    grid_shape[0] // block_size,
                    grid_shape[1] // block_size,
                ),
            })
        return indices, traversal_metadata

    if curve_shape_strategy == 'resize':
        if method in _BLOCK_TRAVERSAL_METHODS:
            block_size = _block_size_for_method(method)
            raise NotImplementedError(
                f"method='{method}' currently supports "
                "curve_shape_strategy='center_crop' only. Its compatible "
                "domain is a centered rectangle with dimensions divisible by "
                f"{block_size}."
            )
        side = _select_resize_side(
            shape,
            method,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        indices = np.asarray(
            _TRAVERSAL_INDEX_GENERATORS[method]((side, side)),
            dtype=int,
        )
        if indices.shape != (side * side, 2):
            raise ValueError(
                f"Traversal method '{method}' did not cover its resized "
                f"{side}x{side} grid exactly once."
            )
        indices = _validate_traversal_indices(indices, (side, side), method)
        return indices, {
            'curve_shape_strategy': 'resize',
            'curve_grid_shape': (side, side),
            'resize_side': side,
            'resize_side_mode': resize_side_mode,
            'resized_traversal_shape': (side, side),
            'traversal_shape': (side, side),
        }

    raise ValueError("curve_shape_strategy must be 'center_crop' or 'resize'.")


def plot_traversals(method='all', grid_shape=(64, 64), *, ncols=3,
                    cmap='turbo', linewidth=1.5, show_grid=True,
                    mark_endpoints=True, show=True):
    """Plot one or all implemented 2D unfolding traversal paths.

    ``grid_shape`` is the maximum requested grid, not a promise that every
    method uses every cell. Full-shape methods traverse it entirely. Curve and
    block methods traverse their largest compatible centered crop within it;
    shaded cells are excluded. Each title shows both the used and maximum grid
    shapes. This visualizer does not resize data or require a ``HyperData``.

    Parameters
    ----------
    method : str, optional
        An implemented traversal name or ``'all'``. The latter plots the ten
        implemented canonical methods in registry order. ``'z_order'`` and
        ``'meander_4'``/``'meander_5'`` are accepted aliases for single plots.
        ``'moore'`` is not yet implemented; ``'coordinate_aligned'`` is an axis
        rearrangement rather than a 2D traversal.
    grid_shape : int or tuple[int, int], optional
        Maximum grid height and width, in ``(y, x)`` order. An integer requests
        a square. Compatible curves may use only a centered part of this grid.
    ncols : int, optional
        Maximum number of subplot columns. A single method uses one column.
    cmap : str or Colormap, optional
        Matplotlib colormap for the path's normalized start-to-end progress.
    linewidth : float, optional
        Width of the colored traversal line in points.
    show_grid, mark_endpoints : bool, optional
        Draw cell boundaries and colored start/end markers, respectively.
    show : bool, optional
        Display the figure with ``matplotlib.pyplot.show`` when True.

    Returns
    -------
    tuple[Figure, ndarray]
        Matplotlib figure and a 2D array of axes. Unused axes in ``'all'``
        mode are hidden. A block method that cannot fit inside ``grid_shape``
        gets an explanatory panel in ``'all'`` mode; a single-method request
        raises the underlying size error.

    Examples
    --------
    >>> fig, axes = plot_traversals('all', (64, 64), ncols=3)
    >>> fig, axes = plot_traversals('peano', 81, linewidth=2, show=False)
    """
    if isinstance(grid_shape, Integral) and not isinstance(grid_shape, (bool, np.bool_)):
        grid_shape = (grid_shape, grid_shape)
    else:
        grid_shape = np.asarray(grid_shape, dtype=object)
        if grid_shape.shape != (2,):
            raise ValueError("grid_shape must be a positive integer or (height, width).")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (Integral, np.integer)) or size < 1
        for size in grid_shape
    ):
        raise ValueError("grid_shape must contain positive integers.")
    height, width = map(int, grid_shape)

    if isinstance(ncols, (bool, np.bool_)) or not isinstance(ncols, Integral) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    if isinstance(linewidth, (bool, np.bool_)) or not isinstance(
        linewidth, (Integral, float, np.floating)
    ) or not np.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("linewidth must be a positive finite number.")

    requested_method = _normalize_traversal_method(method)
    implemented = tuple(
        name for name in _TRAVERSAL_INDEX_GENERATORS if name != 'moore'
    )
    if requested_method == 'all':
        methods = implemented
    elif requested_method == 'moore':
        _moore_indices((height, width))
    elif requested_method in implemented:
        if (
            requested_method in _BLOCK_TRAVERSAL_METHODS
            and min(height, width) < _block_size_for_method(requested_method)
        ):
            _get_traversal_indices((height, width), requested_method)
        methods = (requested_method,)
    else:
        valid = ', '.join(implemented)
        raise ValueError(
            f"method must be 'all' or one of: {valid}. "
            "coordinate_aligned is not a 2D traversal."
        )

    plot_cols = min(int(ncols), len(methods))
    plot_rows = int(np.ceil(len(methods) / plot_cols))
    fig, axes = plt.subplots(
        plot_rows, plot_cols, squeeze=False,
        figsize=(4.8 * plot_cols, 4.8 * plot_rows), layout='constrained',
    )
    progress_norm = Normalize(vmin=0, vmax=1)

    for ax, name in zip(axes.flat, methods):
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        if show_grid:
            grid_lines = [
                ((x - 0.5, -0.5), (x - 0.5, height - 0.5))
                for x in range(width + 1)
            ] + [
                ((-0.5, y - 0.5), (width - 0.5, y - 0.5))
                for y in range(height + 1)
            ]
            ax.add_collection(LineCollection(
                grid_lines, colors='0.75', linewidths=0.3, alpha=0.45, zorder=0.5,
            ))
        ax.add_patch(Rectangle(
            (-0.5, -0.5), width, height, fill=False,
            edgecolor='0.45', linewidth=0.8, zorder=1,
        ))

        if name in _BLOCK_TRAVERSAL_METHODS and min(height, width) < _block_size_for_method(name):
            block_size = _block_size_for_method(name)
            ax.set_title(f'{name}\nunavailable / max {height}x{width}')
            ax.text(
                (width - 1) / 2, (height - 1) / 2,
                f'Requires at least {block_size}x{block_size}',
                ha='center', va='center', fontsize=9,
            )
            ax.set_axis_off()
            continue

        indices, details = _get_traversal_indices(
            (height, width), name, curve_shape_strategy='center_crop',
        )
        used_height, used_width = details.get('curve_grid_shape', (height, width))
        crop_slices = details.get('crop_slices')
        if crop_slices is not None:
            y0, y1 = crop_slices['y']
            x0, x1 = crop_slices['x']
            excluded_regions = (
                (-0.5, -0.5, width, y0),
                (-0.5, y1 - 0.5, width, height - y1),
                (-0.5, y0 - 0.5, x0, y1 - y0),
                (x1 - 0.5, y0 - 0.5, width - x1, y1 - y0),
            )
            for x, y, region_width, region_height in excluded_regions:
                if region_width > 0 and region_height > 0:
                    ax.add_patch(Rectangle(
                        (x, y), region_width, region_height,
                        facecolor='0.88', edgecolor='none', alpha=0.7, zorder=0.2,
                    ))
            ax.add_patch(Rectangle(
                (x0 - 0.5, y0 - 0.5), used_width, used_height,
                fill=False, edgecolor='0.35', linestyle='--',
                linewidth=0.9, zorder=3,
            ))

        xy = indices[:, [1, 0]].astype(np.float32)
        if len(xy) > 1:
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            path = LineCollection(
                segments, cmap=cmap, norm=progress_norm,
                linewidths=linewidth, zorder=2,
            )
            path.set_array(np.linspace(0, 1, len(segments), dtype=np.float32))
            ax.add_collection(path)
        if mark_endpoints:
            ax.scatter(
                xy[0, 0], xy[0, 1], s=28, marker='o', c='limegreen',
                edgecolors='black', linewidths=0.5, zorder=4,
            )
            ax.scatter(
                xy[-1, 0], xy[-1, 1], s=34, marker='X', c='crimson',
                edgecolors='black', linewidths=0.5, zorder=4,
            )
        ax.set_title(f'{name}\nused {used_height}x{used_width} / max {height}x{width}')
        ax.set_axis_off()

    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(
        ScalarMappable(norm=progress_norm, cmap=cmap),
        ax=list(axes.flat)[:len(methods)], shrink=0.7, pad=0.02,
    )
    colorbar.set_label('Traversal progress (start to end)')
    if show:
        plt.show()
    return fig, axes


def _inverse_transpose_order(order):
    """Return the inverse permutation for a transpose order."""
    inverse = [0] * len(order)
    for idx, axis in enumerate(order):
        inverse[axis] = idx
    return tuple(inverse)


def _copy_traversal_metadata(prefix, metadata):
    """Copy traversal metadata, optionally namespaced for domain='both'."""
    if prefix is None:
        return dict(metadata)
    return {f"{prefix}_{key}": value for key, value in metadata.items()}


def _build_unfold_metadata(original_shape, domain='real', method='row_major',
                           curve_shape_strategy='center_crop',
                           preserve_excess=True, resize_side=None,
                           resize_side_mode='nearest', resize_method='linear',
                           preserve_original=False, working_shape=None):
    """Build metadata needed to undo an unfolding exactly when possible."""
    original_shape = _validate_unfold_shape(original_shape)
    working_shape = _validate_unfold_shape(working_shape or original_shape)
    domain, method = _normalize_unfold_request(domain=domain, method=method)
    ry, rx, ky, kx = working_shape

    metadata = {
        'version': 2,
        'original_shape': original_shape,
        'working_shape': working_shape,
        'domain': domain,
        'method': method,
        'preserve_excess': bool(preserve_excess),
        'preserve_original': bool(preserve_original),
    }

    if method == 'coordinate_aligned':
        if domain == 'real':
            transpose_order = (0, 2, 1, 3)
            intermediate_shape = (ry, ky, rx, kx)
            output_shape = (ry * ky, rx * kx)
            interpretation = (
                "Rows combine real-space y with reciprocal-space y; columns "
                "combine real-space x with reciprocal-space x."
            )
        else:
            transpose_order = (2, 0, 3, 1)
            intermediate_shape = (ky, ry, kx, rx)
            output_shape = (ky * ry, kx * rx)
            interpretation = (
                "Rows combine reciprocal-space y with real-space y; columns "
                "combine reciprocal-space x with real-space x."
            )

        metadata.update({
            'representation': 'coordinate_aligned_matrix',
            'curve_shape_strategy': 'not_applicable',
            'transpose_order': transpose_order,
            'inverse_transpose_order': _inverse_transpose_order(transpose_order),
            'intermediate_shape': intermediate_shape,
            'output_shape': output_shape,
            'interpretation': interpretation,
        })
        return metadata

    if domain == 'real':
        indices, traversal_meta = _get_traversal_indices(
            (ry, rx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update(_copy_traversal_metadata(None, traversal_meta))
        metadata.update({
            'representation': 'real_stack',
            'traversal_indices': indices,
            'output_shape': (len(indices), ky, kx),
            'resize_method': resize_method,
            'resized_shape': working_shape if traversal_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "The real-space scan grid is traversed into a stack of "
                "diffraction patterns."
            ),
        })
    elif domain == 'reciprocal':
        indices, traversal_meta = _get_traversal_indices(
            (ky, kx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update(_copy_traversal_metadata(None, traversal_meta))
        metadata.update({
            'representation': 'reciprocal_stack',
            'traversal_indices': indices,
            'output_shape': (len(indices), ry, rx),
            'resize_method': resize_method,
            'resized_shape': working_shape if traversal_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "The reciprocal-space diffraction grid is traversed into a "
                "stack of real-space images."
            ),
        })
    else:
        real_indices, real_meta = _get_traversal_indices(
            (ry, rx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        reciprocal_indices, reciprocal_meta = _get_traversal_indices(
            (ky, kx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update({
            'representation': 'both_matrix',
            'real_traversal_indices': real_indices,
            'reciprocal_traversal_indices': reciprocal_indices,
            'output_shape': (len(real_indices), len(reciprocal_indices)),
            'resize_method': resize_method,
            'resized_shape': working_shape if real_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "Rows traverse real-space positions and columns traverse "
                "reciprocal-space pixels."
            ),
        })
        metadata.update(_copy_traversal_metadata('real', real_meta))
        metadata.update(_copy_traversal_metadata('reciprocal', reciprocal_meta))
        metadata['curve_shape_strategy'] = real_meta['curve_shape_strategy']

    return metadata


def _require_metadata(metadata):
    """Validate unfolding metadata and return a shallow copy."""
    if metadata is None:
        raise ValueError(
            "metadata is required for undo=True. Use return_metadata=True when "
            "unfolding, or call HyperData.unfold(undo=True) on an unfolded "
            "HyperData object that still has attached metadata."
        )
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary returned by unfold(...).")

    required = {'original_shape', 'domain', 'method', 'representation', 'output_shape'}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"metadata is missing required field(s): {', '.join(missing)}.")

    clean = dict(metadata)
    clean['original_shape'] = _validate_unfold_shape(clean['original_shape'])
    clean['working_shape'] = _validate_unfold_shape(
        clean.get('working_shape', clean['original_shape'])
    )
    clean['output_shape'] = tuple(int(v) for v in clean['output_shape'])
    return clean


def _validate_unfolded_shape(array, metadata):
    """Raise a helpful error if an unfolded array does not match metadata."""
    expected_shape = tuple(metadata['output_shape'])
    if array.shape != expected_shape:
        raise ValueError(
            "Unfolded array shape does not match metadata: "
            f"got {array.shape}, expected {expected_shape}."
        )


def _extract_excess_values(array, metadata):
    """Store excluded data required for exact center-crop undo."""
    representation = metadata['representation']
    if metadata.get('curve_shape_strategy') != 'center_crop':
        return
    if not metadata.get('preserve_excess', False):
        return

    if representation == 'real_stack':
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        metadata['excess_values'] = array[
            excess_indices[:, 0],
            excess_indices[:, 1],
            :,
            :,
        ].copy()
    elif representation == 'reciprocal_stack':
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        values = np.empty(
            (len(excess_indices), array.shape[0], array.shape[1]),
            dtype=array.dtype,
        )
        for idx, (ky_idx, kx_idx) in enumerate(excess_indices):
            values[idx] = array[:, :, ky_idx, kx_idx]
        metadata['excess_values'] = values
    elif representation == 'both_matrix':
        real_excess = np.asarray(metadata['real_excess_indices'], dtype=int)
        reciprocal_excess = np.asarray(
            metadata['reciprocal_excess_indices'], dtype=int,
        )
        real_kept = np.asarray(metadata['real_traversal_indices'], dtype=int)
        metadata['excess_values'] = {
            'real': array[
                real_excess[:, 0], real_excess[:, 1], :, :
            ].copy(),
            'reciprocal': array[
                real_kept[:, 0, None],
                real_kept[:, 1, None],
                reciprocal_excess[None, :, 0],
                reciprocal_excess[None, :, 1],
            ].copy(),
        }
        metadata['excess_values_encoding'] = 'separated_domains'


def _unfold_payload_arrays(metadata):
    """Yield large saved values; traversal coordinates remain independent."""
    for key in ('original_values', 'excess_values'):
        value = metadata.get(key)
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, dict):
            yield from (
                item for item in value.values()
                if isinstance(item, np.ndarray)
            )


def _clone_unfold_metadata(metadata, *, share_payload=False):
    """Copy metadata without recopying immutable undo payloads by default."""
    if metadata is None:
        return None
    if not share_payload:
        return deepcopy(metadata)
    memo = {}
    for values in _unfold_payload_arrays(metadata):
        values.setflags(write=False)
        memo[id(values)] = values
    return deepcopy(metadata, memo)


def _crop_restore_shape(metadata):
    """Return the shape restored when center-crop excess was not preserved."""
    representation = metadata['representation']
    working_shape = metadata['working_shape']
    ry, rx, ky, kx = working_shape

    if representation == 'real_stack':
        side_y, side_x = metadata['curve_grid_shape']
        return (side_y, side_x, ky, kx)
    if representation == 'reciprocal_stack':
        side_y, side_x = metadata['curve_grid_shape']
        return (ry, rx, side_y, side_x)
    if representation == 'both_matrix':
        real_side_y, real_side_x = metadata['real_curve_grid_shape']
        reciprocal_side_y, reciprocal_side_x = metadata['reciprocal_curve_grid_shape']
        return (real_side_y, real_side_x, reciprocal_side_y, reciprocal_side_x)
    raise ValueError(f"Unsupported crop restore representation '{representation}'.")


def _local_crop_indices(indices, crop_slices):
    """Convert original-array crop coordinates to local cropped coordinates."""
    indices = np.asarray(indices, dtype=int)
    y0 = crop_slices['y'][0]
    x0 = crop_slices['x'][0]
    return indices - np.array([y0, x0])


def _restore_real_stack(array, metadata):
    """Undo a real-domain stack unfolding."""
    indices = np.asarray(metadata['traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        local_indices = _local_crop_indices(indices, metadata['crop_slices'])
        restored[local_indices[:, 0], local_indices[:, 1], :, :] = array
        return restored

    if strategy == 'resize' and metadata.get('preserve_original', False):
        return np.array(metadata['original_values'], copy=True)

    restored_shape = metadata['working_shape']
    restored = np.empty(restored_shape, dtype=array.dtype)

    if strategy == 'center_crop' and metadata.get('preserve_excess', False):
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        restored[excess_indices[:, 0], excess_indices[:, 1], :, :] = metadata['excess_values']

    restored[indices[:, 0], indices[:, 1], :, :] = array
    return restored


def _restore_reciprocal_stack(array, metadata):
    """Undo a reciprocal-domain stack unfolding."""
    indices = np.asarray(metadata['traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        local_indices = _local_crop_indices(indices, metadata['crop_slices'])
        for flat_idx, (ky_idx, kx_idx) in enumerate(local_indices):
            restored[:, :, ky_idx, kx_idx] = array[flat_idx]
        return restored

    if strategy == 'resize' and metadata.get('preserve_original', False):
        return np.array(metadata['original_values'], copy=True)

    restored_shape = metadata['working_shape']
    restored = np.empty(restored_shape, dtype=array.dtype)

    if strategy == 'center_crop' and metadata.get('preserve_excess', False):
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        for flat_idx, (ky_idx, kx_idx) in enumerate(excess_indices):
            restored[:, :, ky_idx, kx_idx] = metadata['excess_values'][flat_idx]

    for flat_idx, (ky_idx, kx_idx) in enumerate(indices):
        restored[:, :, ky_idx, kx_idx] = array[flat_idx]
    return restored


def _restore_both_matrix(array, metadata):
    """Undo a 2D mixed real/reciprocal unfolding."""
    real_indices = np.asarray(metadata['real_traversal_indices'], dtype=int)
    reciprocal_indices = np.asarray(metadata['reciprocal_traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        real_local = _local_crop_indices(real_indices, metadata['real_crop_slices'])
        reciprocal_local = _local_crop_indices(
            reciprocal_indices,
            metadata['reciprocal_crop_slices'],
        )
    else:
        if strategy == 'resize' and metadata.get('preserve_original', False):
            return np.array(metadata['original_values'], copy=True)
        if strategy == 'center_crop' and metadata.get('preserve_excess', False):
            encoding = metadata.get('excess_values_encoding', 'full_tensor')
            if encoding == 'full_tensor':
                # Saved files from the original implementation use this form.
                restored = np.array(metadata['excess_values'], copy=True)
            elif encoding == 'separated_domains':
                restored = np.empty(metadata['working_shape'], dtype=array.dtype)
                real_excess = np.asarray(metadata['real_excess_indices'], dtype=int)
                reciprocal_excess = np.asarray(
                    metadata['reciprocal_excess_indices'], dtype=int,
                )
                values = metadata['excess_values']
                restored[real_excess[:, 0], real_excess[:, 1], :, :] = values['real']
                for flat_idx, (ry_idx, rx_idx) in enumerate(real_indices):
                    restored[
                        ry_idx, rx_idx,
                        reciprocal_excess[:, 0], reciprocal_excess[:, 1],
                    ] = values['reciprocal'][flat_idx]
            else:
                raise ValueError(
                    f"Unsupported excess-values encoding '{encoding}'."
                )
        else:
            restored = np.empty(metadata['working_shape'], dtype=array.dtype)
        real_local = real_indices
        reciprocal_local = reciprocal_indices

    for real_flat_idx, (ry_idx, rx_idx) in enumerate(real_local):
        restored[
            ry_idx,
            rx_idx,
            reciprocal_local[:, 0],
            reciprocal_local[:, 1],
        ] = array[real_flat_idx]
    return restored


def _infer_square_side(value, label):
    """Infer a square side length from a flattened grid size."""
    value = int(value)
    side = int(np.sqrt(value))
    if side * side != value:
        raise ValueError(
            f"Cannot infer a square {label} grid from {value} elements. "
            "Pass original_shape explicitly."
        )
    return side


def _infer_row_major_original_shape(array, domain, original_shape=None):
    """Infer or validate the original 4D shape for row-major undo without metadata."""
    domain, _ = _normalize_unfold_request(domain=domain, method='row_major')

    if original_shape is not None:
        original_shape = tuple(int(v) for v in original_shape)
        if len(original_shape) == 4:
            return _validate_unfold_shape(original_shape)
        if len(original_shape) == 2 and domain == 'real' and array.ndim == 3:
            ry, rx = original_shape
            return _validate_unfold_shape((ry, rx, array.shape[1], array.shape[2]))
        if len(original_shape) == 2 and domain == 'reciprocal' and array.ndim == 3:
            ky, kx = original_shape
            return _validate_unfold_shape((array.shape[1], array.shape[2], ky, kx))
        raise ValueError(
            "original_shape must be (Ry, Rx, Ky, Kx). For 3D row-major "
            "domain='real' undo, (Ry, Rx) is also accepted; for "
            "domain='reciprocal' undo, (Ky, Kx) is also accepted."
        )

    if domain == 'real' and array.ndim == 3:
        ry = rx = _infer_square_side(array.shape[0], 'real-space')
        return (ry, rx, array.shape[1], array.shape[2])

    if domain == 'reciprocal' and array.ndim == 3:
        ky = kx = _infer_square_side(array.shape[0], 'reciprocal-space')
        return (array.shape[1], array.shape[2], ky, kx)

    if domain == 'both' and array.ndim == 2:
        ry = rx = _infer_square_side(array.shape[0], 'real-space')
        ky = kx = _infer_square_side(array.shape[1], 'reciprocal-space')
        return (ry, rx, ky, kx)

    raise ValueError(
        "metadata-less undo requires original_shape unless a square row-major "
        "shape can be inferred from the unfolded array."
    )


def _build_row_major_undo_metadata(array, domain='real', method='row_major',
                                   original_shape=None):
    """Build minimal row-major metadata for undo when no unfold metadata exists."""
    domain, method = _normalize_unfold_request(domain=domain, method=method)
    if method != 'row_major':
        raise ValueError(
            "metadata-less undo only supports method='row_major'. Pass metadata "
            "for non-row-major traversal methods."
        )

    original_shape = _infer_row_major_original_shape(
        array,
        domain=domain,
        original_shape=original_shape,
    )
    return _build_unfold_metadata(
        original_shape=original_shape,
        working_shape=original_shape,
        domain=domain,
        method='row_major',
        preserve_excess=False,
    )


def _unfold_array(array, domain='real', method='row_major',
                  curve_shape_strategy='center_crop', preserve_excess=True,
                  resize_side=None, resize_side_mode='nearest',
                  resize_method='linear', preserve_original=False,
                  original_shape=None, original_values=None, undo=False,
                  metadata=None, return_metadata=False):
    """
    Unfold or restore a 4D-STEM tensor using explicit domain/method metadata.

    Examples
    --------
    >>> original = np.arange(5*7*4*6).reshape(5, 7, 4, 6)
    >>> unfolded, meta = _unfold_array(original, domain='real',
    ...                                method='hilbert',
    ...                                preserve_excess=True,
    ...                                return_metadata=True)
    >>> restored = _unfold_array(unfolded, undo=True, metadata=meta)
    >>> np.array_equal(restored, original)
    True
    """
    array = np.asarray(array)

    if undo:
        if metadata is None:
            metadata = _build_row_major_undo_metadata(
                array,
                domain=domain,
                method=method,
                original_shape=original_shape,
            )
        metadata = _require_metadata(metadata)
        _validate_unfolded_shape(array, metadata)
        representation = metadata['representation']

        if representation == 'coordinate_aligned_matrix':
            restored = array.reshape(metadata['intermediate_shape'])
            restored = np.transpose(restored, metadata['inverse_transpose_order'])

        elif representation == 'real_stack':
            restored = _restore_real_stack(array, metadata)

        elif representation == 'reciprocal_stack':
            restored = _restore_reciprocal_stack(array, metadata)

        elif representation == 'both_matrix':
            restored = _restore_both_matrix(array, metadata)

        else:
            raise ValueError(f"Unsupported metadata representation '{representation}'.")

        if return_metadata:
            return restored, metadata
        return restored

    if array.ndim != 4:
        raise ValueError(
            "Forward unfolding expects a 4D-STEM tensor with shape "
            "(Ry, Rx, Ky, Kx)."
        )

    metadata = _build_unfold_metadata(
        original_shape=original_shape or array.shape,
        working_shape=array.shape,
        domain=domain,
        method=method,
        curve_shape_strategy=curve_shape_strategy,
        preserve_excess=preserve_excess,
        resize_side=resize_side,
        resize_side_mode=resize_side_mode,
        resize_method=resize_method,
        preserve_original=preserve_original,
    )

    if preserve_original and original_values is not None:
        metadata['original_values'] = np.array(original_values, copy=True)

    _extract_excess_values(array, metadata)
    payloads = tuple(_unfold_payload_arrays(metadata))
    metadata['preserved_values_nbytes'] = sum(values.nbytes for values in payloads)
    for values in payloads:
        values.setflags(write=False)
    representation = metadata['representation']

    if representation == 'coordinate_aligned_matrix':
        unfolded = np.transpose(array, metadata['transpose_order'])
        unfolded = unfolded.reshape(metadata['output_shape'])

    elif representation == 'real_stack':
        indices = metadata['traversal_indices']
        unfolded = array[indices[:, 0], indices[:, 1], :, :]

    elif representation == 'reciprocal_stack':
        indices = metadata['traversal_indices']
        unfolded = np.empty(metadata['output_shape'], dtype=array.dtype)
        for flat_idx, (ky_idx, kx_idx) in enumerate(indices):
            unfolded[flat_idx] = array[:, :, ky_idx, kx_idx]

    elif representation == 'both_matrix':
        real_indices = metadata['real_traversal_indices']
        reciprocal_indices = metadata['reciprocal_traversal_indices']
        unfolded = np.empty(metadata['output_shape'], dtype=array.dtype)
        for flat_idx, (ry_idx, rx_idx) in enumerate(real_indices):
            unfolded[flat_idx] = array[
                ry_idx,
                rx_idx,
                reciprocal_indices[:, 0],
                reciprocal_indices[:, 1],
            ]

    else:
        raise ValueError(f"Unsupported unfolding representation '{representation}'.")

    metadata['output_shape'] = unfolded.shape

    if return_metadata:
        return unfolded, metadata
    return unfolded


def clip_values(array, a_min=1, a_max=None):
    """Clip values in a numpy array to a specified range.

    This function clips the values in the input array to lie within the specified minimum and maximum 
    limits. Values less than `a_min` are set to `a_min`, and values greater than `a_max` (if provided) 
    are set to `a_max`. If `a_max` is not specified, no upper limit is applied.

    Parameters
    ----------
    array : np.ndarray
        The input numpy array whose values are to be clipped.
    a_min : float or int, optional
        The minimum value to which the elements in the array are clipped. Default is 1.
    a_max : float or int, optional
        The maximum value to which the elements in the array are clipped. If not specified, no upper 
        limit is applied.

    Returns
    -------
    clipped_array : np.ndarray
        The numpy array with values clipped to the specified range.

    Examples
    --------
    >>> arr = np.array([0, 2, 5, 10])
    >>> clip_values(arr, a_min=1, a_max=5)
    array([1, 2, 5, 5])

    Notes
    -----
    This function is a wrapper around `numpy.clip`, which performs the actual clipping operation. 
    The `a_min` and `a_max` parameters are inclusive, meaning that any value equal to `a_min` or 
    `a_max` will remain unchanged.
    """
    return np.clip(array, a_min, a_max)


def plot_centers_of_mass_with_histograms(centers_data, colors, labels=None, drawConvexHull=True,
                                         transparency=0.5, hist_height=0.16, bins=150,
                                         alpha=0.5, density=True, label_size=22, tick_label_size=18,
                                         x_range=(-0.9, 0.9), y_range=(-0.9, 0.9), hist_title_size=16):
    """
    Plots the centers of mass for all features from each dataset, centered at (0, 0),
    with histograms of the x and y coordinates.

    centers_data: The dataset with shape (n_datasets, A1, A2, n_spots, 2)
    hist_height: Height of the histograms as a fraction of total figure height
    label_size: Font size for the labels
    """

    # Check data compatibility
    assert type(drawConvexHull) is bool, "'drawConvexHull' must be a boolean (True/False) variable"
    assert type(density) is bool, "'density' must be a boolean  (True/False) variable"
    assert len(colors) == centers_data.shape[0], "The number of colors must match the number of datasets to plot."
    assert all(isinstance(item, str) for item in colors), "Not all elements are strings."

    n_datasets = centers_data.shape[0]

    # Create the main plot
    fig = plt.figure(figsize=(9, 9))
    ax_scatter = plt.axes([0.1, 0.1, 0.65, 0.65])
    ax_histx = plt.axes([0.1, 0.75, 0.65, hist_height], sharex=ax_scatter)
    ax_histy = plt.axes([0.75, 0.1, hist_height, 0.65], sharey=ax_scatter)

    # Disable labels on histogram to prevent overlap
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    # Initialize standard deviation lists
    std_dev_y = []
    std_dev_x = []

    markers = ['*', 'D', 's', '.', 'v', 'o', 'P', 'X']  # Different markers
    line_styles = ['--', '-', '-.', ':']  # Different line styles

    for i in range(n_datasets):
        all_x_coords = []
        all_y_coords = []

        # Collect all coordinates from all feature indices for the current dataset
        for feature_index in range(centers_data.shape[3]):
            y_coords = centers_data[i, :, :, feature_index, 0].flatten()
            x_coords = centers_data[i, :, :, feature_index, 1].flatten()
            y_mean, x_mean = np.mean(y_coords), np.mean(x_coords)

            all_y_coords.extend(y_coords - y_mean)
            all_x_coords.extend(x_coords - x_mean)

            # Scatter plot for each feature of the dataset with different markers
            if feature_index == 1:
                transparency /= 3
            ax_scatter.scatter(x_coords - x_mean, y_coords - y_mean, color=colors[i], alpha=transparency,
                               marker=markers[i % len(markers)])

        # Calculate and store standard deviations
        std_dev_y.append(np.std(all_y_coords))
        std_dev_x.append(np.std(all_x_coords))

        # Combine all x and y coordinates
        combined_coords = np.column_stack((all_x_coords, all_y_coords))

        # Draw convex hull for the combined coordinates of the dataset with different line styles
        if drawConvexHull and len(combined_coords) > 2:
            hull = ConvexHull(combined_coords)
            for simplex in hull.simplices:
                ax_scatter.plot(combined_coords[simplex, 0], combined_coords[simplex, 1], color=colors[i],
                                linewidth=2, linestyle=line_styles[i % len(line_styles)])

        # Add label for the dataset
        if labels is not None:
            ax_scatter.plot([], [], color=colors[i], label=labels[i], linestyle='None',
                            marker=markers[i % len(markers)], markerfacecolor=colors[i])
        else:
            ax_scatter.plot([], [], color=colors[i], label=f'Dataset {i+1}', linestyle='None',
                            marker=markers[i % len(markers)], markerfacecolor=colors[i],)

        # Plot histograms
        ax_histx.hist(all_x_coords, bins=bins, color=colors[i], alpha=alpha,
                      density=density, label=rf'$\sigma$ = {std_dev_x[-1]:.2f}')
        ax_histy.hist(all_y_coords, bins=bins, color=colors[i], alpha=alpha, orientation='horizontal',
                      density=density, label=rf'$\sigma$ = {std_dev_y[-1]:.2f}')

        print(rf'$\sigma$ = {std_dev_x[-1]:.2f}')
        print(rf'$\sigma$ = {std_dev_y[-1]:.2f}')

    # Set labels and title for the scatter plot
    ax_scatter.set_xlabel(r'$k_x$ Displacement (px.)', fontsize=label_size)
    ax_scatter.set_ylabel(r'$k_y$ Displacement (px.)', fontsize=label_size)
    # ax_scatter.set_title('Centers of Mass with Histograms', fontsize=label_size)
    # ax_scatter.legend(fontsize=label_size)
    
    ax_scatter.set_xlim(x_range)
    ax_scatter.set_ylim(y_range)
    ax_scatter.set_xticks(np.linspace(-.8,.8,9))
    ax_scatter.set_yticks(np.linspace(-.8,.8,9))

    
    ax_scatter.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histx.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histy.tick_params(axis='both', which='major', labelsize=tick_label_size)
    ax_histx.set_title('Center of Mass Precision', fontsize=25, pad=30)

    # Set the tick labels for the histograms
    ax_histx.set_yticks([0, 1, 2, 3])
    ax_histy.set_xticks([0, 1, 2, 3])
    
    # ax_histx.set_title('Probability\n Density', fontsize=hist_title_size, rotation=270,pad=50)
    ax_histy.set_title('Noisy Dataset\n \n Denoised Dataset', fontsize=hist_title_size,pad=50)
    
    plt.show()

def plotStrain(strain_data, title='Strain', axis='on', lim_val=0.05, cmap='RdBu', **kwargs):
    """ Strain/Rotation map plotting """

    plt.figure(figsize=(4.5, 10))

    im1 = plt.imshow(strain_data, vmin=-lim_val, vmax=lim_val, cmap=cmap, **kwargs)

    # plt.xticks(np.arange(0, 255, 10))
    # plt.yticks(np.arange(0, 110, 10))
    # plt.yticks(np.arange(0, 60, step=10), size = 20)
    
    plt.axis(axis)
    plt.title(title)
    # plt.grid()
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = plt.colorbar(im1, cax=cax)
    cb.ax.tick_params(labelsize=10)

    plt.show()

def plotHist_andClusters(data, clusters, cluster_indices, bins=100, xrange=None, yrange=None, 
                         axis_title_size=18, tick_label_size=14, color='blue', outline_color='red'):
    """
    Plot a histogram from "data" and outline histograms for specified clusters.

    :param data: Data to be plotted.
    :param clusters: Array of integers indicating cluster membership for each data point.
    :param cluster_indices: List of cluster indices for which to plot histogram outlines.
    :param bins: Number of bins in the histogram.
    :param xrange: Tuple specifying the (min, max) range of the x-axis.
    :param color: Color of the main histogram.
    :param outline_color: Color of the outlines for specified clusters.
    """

    # Flatten data
    flattened_data = data.flatten()

    plt.figure(figsize=(6.5, 5))

    # Plot the main histogram
    counts, bin_edges, _ = plt.hist(
        flattened_data, bins=bins, range=xrange, color=color, alpha=0.5, label='All Data')

    # Calculate bin centers from edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    for index in cluster_indices:
        # Extract data for the current cluster
        cluster_data = flattened_data[clusters.flatten() == index]

        # Calculate histogram for the current cluster
        cluster_counts, _ = np.histogram(cluster_data, bins=bin_edges)

        # Plot histogram outline for the current cluster with the user-specified outline color
        plt.plot(bin_centers, cluster_counts,
                 label=f'Cluster {index}', color=outline_color, drawstyle='steps-mid')
    
    # Set the x-axis and y-axis range if specified
    if xrange:
        plt.xlim(xrange)
    if yrange:
        plt.ylim(yrange)
    
    for tick in plt.gca().get_xticks():
        plt.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
    
    # Set tick label sizes
    plt.xticks(fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)
    
    plt.xlabel('Strain (%)', fontsize=axis_title_size)
    plt.ylabel('Frequency', fontsize=axis_title_size)
    # plt.legend()
    plt.show()

def inpaint_background(image, centers=None, radius=6, mask=None,
                       method='biharmonic', split_into_regions=False):
    """
    Estimate a smooth background by inpainting selected diffraction features.

    The inpainted region can be defined by one or more circular spot masks,
    an explicit Boolean mask, or the union of both. The default biharmonic
    method is useful for filling Bragg disks/spots with a locally smooth
    background estimate.

    Parameters
    ----------
    image : ndarray
        Two-dimensional diffraction pattern or image to inpaint.
    centers : array-like or None, optional
        One ``(ky, kx)`` center or an ``(N, 2)`` array of centers. Each center
        defines a circular region to inpaint. Required when ``mask`` is not
        provided.
    radius : float or array-like of floats, optional
        Radius, in pixels, of the circular region around each center. A scalar
        radius is applied to every center; otherwise provide one radius per
        center. Defaults to 6.
    mask : ndarray of bool or None, optional
        Explicit mask with the same shape as ``image``. True-valued pixels are
        inpainted. If both ``mask`` and circular regions are provided, their
        union is used.
    method : {'biharmonic'}, optional
        Inpainting method. 
    split_into_regions : bool, optional
        Passed to ``skimage.restoration.inpaint_biharmonic``. Splitting can be
        faster for disconnected masks, while ``False`` preserves the previous
        4Denoise behavior.

    Returns
    -------
    ndarray
        Image with selected pixels replaced by inpainted background values.
    """
    from skimage.restoration import inpaint_biharmonic

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"inpaint_background requires a 2D image; got shape {image.shape}."
        )
    if np.iscomplexobj(image):
        raise TypeError("inpaint_background requires a real-valued image.")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values.")

    if not isinstance(method, str):
        raise ValueError("method must be 'biharmonic'.")
    normalized_method = method.strip().lower().replace('-', '_')
    if normalized_method not in {'biharmonic', 'biharmonic_inpaint'}:
        raise NotImplementedError(
            "Only method='biharmonic' is currently implemented for "
            "inpaint_background."
        )
    if not isinstance(split_into_regions, (bool, np.bool_)):
        raise ValueError("split_into_regions must be a boolean.")

    h, w = image.shape
    inpaint_mask = np.zeros((h, w), dtype=bool)

    if mask is not None:
        explicit_mask = np.asarray(mask, dtype=bool)
        if explicit_mask.shape != image.shape:
            raise ValueError(
                f"mask must have shape {image.shape}; got "
                f"{explicit_mask.shape}."
            )
        inpaint_mask |= explicit_mask

    if centers is not None:
        center_values = np.asarray(centers, dtype=float)
        if center_values.shape == (2,):
            center_values = center_values.reshape(1, 2)
        elif center_values.ndim != 2 or center_values.shape[1] != 2:
            raise ValueError(
                "centers must be one (ky, kx) pair or an (N, 2) array."
            )
        if center_values.shape[0] == 0:
            raise ValueError("centers must contain at least one center.")
        if not np.all(np.isfinite(center_values)):
            raise ValueError("centers must contain only finite values.")

        radius_values = np.asarray(radius, dtype=float)
        if radius_values.ndim == 0:
            radius_values = np.full(
                center_values.shape[0],
                float(radius_values),
            )
        elif (
            radius_values.ndim == 1
            and radius_values.size == center_values.shape[0]
        ):
            pass
        else:
            raise ValueError(
                "radius must be a scalar or contain one value per center."
            )
        if (
            not np.all(np.isfinite(radius_values))
            or np.any(radius_values <= 0)
        ):
            raise ValueError("All radii must be positive and finite.")

        y_grid, x_grid = np.ogrid[:h, :w]
        for (center_y, center_x), spot_radius in zip(center_values, radius_values):
            dist_sq = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
            inpaint_mask |= dist_sq <= float(spot_radius) ** 2

    if centers is None and mask is None:
        raise ValueError("Provide centers/radius, an explicit mask, or both.")
    if not np.any(inpaint_mask):
        raise ValueError("The inpainting mask selects no pixels.")
    if np.all(inpaint_mask):
        raise ValueError("The inpainting mask cannot cover the entire image.")

    inpaint_input = image.astype(np.result_type(image.dtype, np.float32), copy=True)
    return inpaint_biharmonic(
        inpaint_input,
        inpaint_mask,
        split_into_regions=bool(split_into_regions),
    )

def sort_peaks(peak_centers, center, order_length=None):
    """
    Sorts peak centers by distance from a center point and optionally resorts 
    them by angle in groups.

    Parameters
    ----------
    peak_centers : ndarray
        Array of shape (A, 2) containing the coordinates of the peak centers.
    center : tuple
        Tuple (center_y, center_x) specifying the center point for distance calculation.
    order_length : int, optional
        The number of elements in each group to sort based on angle after initial distance sorting.
        If None, no secondary sorting is performed.

    Returns
    -------
    sorted_array : ndarray
        Array of peak centers sorted by distance and optionally resorted by angle in groups.
    """
    center_y, center_x = center

    # Calculate distances from the center point
    distances = np.sqrt((peak_centers[:, 0] - center_y) ** 2 + (peak_centers[:, 1] - center_x) ** 2)
    sorted_indices = np.argsort(distances)
    sorted_peaks = peak_centers[sorted_indices]

    if order_length is not None and order_length > 1:
        # Secondary sorting by angle within groups defined by order_length
        num_full_groups = len(sorted_peaks) // order_length
        resorted_array = []

        for i in range(num_full_groups):
            start_idx = i * order_length
            end_idx = start_idx + order_length
            subgroup = sorted_peaks[start_idx:end_idx]
            angles = np.arctan2(subgroup[:, 1] - center_x, subgroup[:, 0] - center_y)
            angle_indices = np.argsort(angles)
            resorted_array.append(subgroup[angle_indices])

        # Process the remaining elements
        if len(sorted_peaks) % order_length != 0:
            start_idx = num_full_groups * order_length
            remaining_group = sorted_peaks[start_idx:]
            remaining_angles = np.arctan2(remaining_group[:, 1] - center_x, remaining_group[:, 0] - center_y)
            remaining_angle_indices = np.argsort(remaining_angles)
            resorted_array.append(remaining_group[remaining_angle_indices])

        # Concatenate all the groups back into a single array
        sorted_peaks = np.concatenate(resorted_array, axis=0)

    return sorted_peaks

def reconstruct_height(xGrad, yGrad, y_bds_flat, x_bds_flat, iterations=10, threshold_percent=0.5, 
                       returnGradients=True, max_window_size=7):
    
    """
    Make 3D reconstruction based on xGrad and yGrad information. For each 
    iteration, the gradient sign (without changing its magnitude) is refined 
    for surface continuity.
    """
    
    # Use the adaptive median filter for initial sign correction
    xCorr = fix_sign_errors_adaptive(xGrad, max_window_size=max_window_size, initial_window_size=3)
    yCorr = fix_sign_errors_adaptive(yGrad, max_window_size=max_window_size, initial_window_size=3)

    initial_threshold_percent = threshold_percent
    for i in tqdm(range(iterations), desc="Reconstructing height"):
        current_threshold_percent = initial_threshold_percent * (iterations - i) / iterations
               
        h_map = reconFromGradDir(yCorr, xCorr, plot=False)
        hmap_fixed = fix_tilt_and_height(h_map, y_bds_flat, x_bds_flat)

        if i < iterations - 1:
            
            # Obtain the gradients from the reconstructed map
            xGrad_recon, yGrad_recon = compute_gradients(hmap_fixed)

            # Compare the original, modified gradients xCorr, yCorr with xGrad_recon, yGrad_recon
            xDiff = np.abs(xCorr - xGrad_recon) 
            yDiff = np.abs(yCorr - yGrad_recon)

            # Determine threshold based on percentile
            xThreshold = np.percentile(xDiff, 100 - current_threshold_percent)
            yThreshold = np.percentile(yDiff, 100 - current_threshold_percent)
            
            # Flip signs where the difference exceeds the threshold
            xCorr[xDiff > xThreshold] *= -1
            yCorr[yDiff > yThreshold] *= -1
            
            # Fix signs of gradients using the adaptive function
            xCorr = fix_sign_errors_adaptive(xCorr, max_window_size=max_window_size, initial_window_size=3)
            yCorr = fix_sign_errors_adaptive(yCorr, max_window_size=max_window_size, initial_window_size=3)
                
            # Flip signs where the difference exceeds the threshold
            xCorr[xDiff > xThreshold] = 0.5 * (xCorr[xDiff > xThreshold] + xGrad_recon[xDiff > xThreshold])
            yCorr[yDiff > yThreshold] = 0.5 * (yCorr[yDiff > yThreshold] + yGrad_recon[yDiff > yThreshold])
                               
    if returnGradients:
        return hmap_fixed, xCorr, yCorr
        
    else: 
        
        return hmap_fixed


def fix_sign_errors_adaptive(img, max_window_size=7, initial_window_size=3, variance_threshold=0.05):
    """Flip pixels whose signs disagree with an adaptive local median.

    Parameters
    ----------
    img : numpy.ndarray
        Two-dimensional array of signed values. It is not modified.
    max_window_size : int, optional
        Largest odd neighborhood width considered around each pixel.
    initial_window_size : int, optional
        Starting odd neighborhood width; it must not exceed
        ``max_window_size``.
    variance_threshold : float, optional
        Stop expanding a neighborhood once its variance is below this value.

    Returns
    -------
    numpy.ndarray
        Copy of ``img`` with sign-disagreeing pixels multiplied by -1.

    Notes
    -----
    Neighborhoods are clipped at image edges. If no window falls below the
    variance threshold, the median of the largest examined window is used.
    """

    # Ensure both dimensions of the initial window size are odd for center pixel calculation
    if initial_window_size % 2 == 0 or max_window_size % 2 == 0:
        raise ValueError("Both initial_window_size and max_window_size must be odd.")

    # Create a copy of the original image to store the corrected values.
    corrected_img = np.copy(img)

    # Get the dimensions of the image.
    rows, cols = img.shape

    # Slide through the image using the adaptive window.
    for i in range(rows):
        for j in range(cols):
            # Start with the initial window size
            window_size = initial_window_size
            margin = window_size // 2

            # Get the center pixel value
            center_val = img[i, j]

            # Loop to adaptively increase window size
            while window_size <= max_window_size:
                # Ensure the window does not exceed the image boundaries
                r_start = max(0, i - margin)
                r_end = min(rows, i + margin + 1)
                c_start = max(0, j - margin)
                c_end = min(cols, j + margin + 1)

                # Extract the current neighborhood window
                window = img[r_start:r_end, c_start:c_end]
                
                # Compute the median and variance of the current window
                window_median = np.median(window)
                window_variance = np.var(window)

                # Check if the variance is below a threshold, meaning it's not noisy
                if window_variance < variance_threshold:
                    break  # Stop increasing window size, variance is low
                
                # Otherwise, increase the window size and try again
                window_size += 2  # Increase the window size by 2 (to maintain odd size)
                margin = window_size // 2

            # Check the sign of the center pixel and the median
            if np.sign(window_median) != np.sign(center_val):
                corrected_img[i, j] *= -1

    return corrected_img


def reconFromGradDir(grad_x, grad_y, im_height=None, plot=True):
    
    """
    Function based on MATLAB code written by Colin Ophus (MM/YYYY) 
    """
    
    padding = 2  # padding of reconstruction space
    qMin = 0.00  # min spatial frequency in 1/pixels
    qMax = 0.25  # max spatial frequency in 1/pixels
    num_iter = 50
    step_size = 0.99

    # Coordinates
    im_size = grad_x.shape
    N = tuple(s * padding for s in im_size)
    qxa, qya = makeFourierCoords(N, 1)
    q2a = qxa**2 + qya**2

    # Operators
    q2inv = np.reciprocal(q2a)
    q2inv[0, 0] = 0
    qFilt = np.exp(q2a / (-2 * qMax**2))
    if qMin > 0:
        qFilt = qFilt * (1 - np.exp(q2a / (-2 * qMin**2)))
    qxOp = (-1j / 4) * qxa * q2inv * qFilt
    qyOp = (-1j / 4) * qya * q2inv * qFilt

    # Normalize the gradients
    grad_x = grad_x - np.median(grad_x)
    grad_y = grad_y - np.median(grad_y)

    # Mask updates
    vx = np.arange(im_size[0])
    vy = np.arange(im_size[1])
    mask = np.zeros(N, dtype=bool)
    mask[np.ix_(vx, vy)] = True
    mask_inv = ~mask

    # Reconstruct height
    recon_height = np.zeros(N)
    for a0 in range(num_iter):
        grad_x_recon = (np.roll(recon_height, shift=-1, axis=0) -
                        np.roll(recon_height, shift=1, axis=0)) / 2
        grad_y_recon = (np.roll(recon_height, shift=-1, axis=1) -
                        np.roll(recon_height, shift=1, axis=1)) / 2

        # Difference and masking
        grad_x_recon[mask] = grad_x_recon[mask] - grad_x.ravel()
        grad_y_recon[mask] = grad_y_recon[mask] - grad_y.ravel()
        grad_x_recon[mask_inv] = 0
        grad_y_recon[mask_inv] = 0

        recon_update = np.fft.ifft2(
            np.fft.fft2(grad_x_recon) * qxOp +
            np.fft.fft2(grad_y_recon) * qyOp
        ).real

        recon_height -= step_size * recon_update

    # Crop
    recon_height = recon_height[np.ix_(vx, vy)]

    # Plotting
    if plot:
        plt.figure(11)
        plt.clf()
        ax = plt.gca()
        Ip1 = recon_height - np.median(recon_height)
        if im_height is None:
            im = ax.imshow(Ip1, cmap='turbo')
        else:
            Ip2 = im_height - np.median(im_height)
            im = ax.imshow(np.hstack((Ip1, Ip2)), cmap='turbo')

        plt.axis('equal')
        plt.axis('off')

        # Adjust the colorbar to have the same height as the image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()

    return recon_height

def makeFourierCoords(size, spacing):
    """
    This function is based on MATLAB code written by Colin Ophus (MM/YYYY).
    """
    nx, ny = size
    qx = np.fft.fftfreq(nx, spacing)
    qy = np.fft.fftfreq(ny, spacing)
    qxa, qya = np.meshgrid(qx, qy, indexing='ij')
    return qxa, qya

def fix_tilt_and_height(height_map, y_bds, x_bds):
    """
    Helper function to perform linear regression and return the slope
    """
    def get_slope(ave_height):
        x = np.arange(ave_height.size)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, _ = np.linalg.lstsq(A, ave_height, rcond=None)[0]
        return slope

    # 1. Linear regression for x-axis and adjust tilt
    ave_height_x = np.mean(height_map, axis=0)
    slope_x = get_slope(ave_height_x)
    height_map = rotate_height(
        height_map, np.arctan(slope_x) * 180 / np.pi, axis='x')

    # 2. Linear regression for y-axis and adjust tilt
    ave_height_y = np.mean(height_map, axis=1)
    slope_y = get_slope(ave_height_y)
    height_map = rotate_height(
        height_map, np.arctan(slope_y) * 180 / np.pi, axis='y')

    # 3. Adjust the average height based on a chosen region
    height_map = height_map - np.mean(height_map[y_bds[0]:y_bds[1], x_bds[0]:x_bds[1]])

    return height_map

def compute_gradients(height_map):
    """
    Compute gradients given a 3D map
    """
    
    phi, theta = get_surface_tilt_and_direction(height_map, units='rad', show_results=False)
    
    grads = get_gradients(np.stack((phi, theta), axis=-1), rot_angle=180)

    return grads[...,1], grads[...,0]  

def rotate_height(height, angle, axis='x'):
    """Rotate the height map based on the specified axis and angle.

    Parameters:
    - height: 2D numpy array representing the height map.
    - angle: float, rotation angle in degrees.
    - axis: str, 'x' for x-axis rotation, 'y' for y-axis rotation.

    Returns:
    - 2D numpy array representing the rotated height map.
    """

    shape_array = np.shape(height)

    if axis == 'x':
        i_values = np.arange(shape_array[1])
        adjustment = i_values * np.tan(angle * np.pi / 180)
        height_rot = height[:, :shape_array[1]] - adjustment
    elif axis == 'y':
        i_values = np.arange(shape_array[0])
        adjustment = i_values * np.tan(angle * np.pi / 180)
        height_rot = height[:shape_array[0], :] - adjustment[:, np.newaxis]

    return height_rot

def low_std_mask_and_average(data, percentile=5):
    """
    Generates a mask for the X% of pixels with the lowest standard deviation along the N axis
    and returns a vector of length N by averaging those masked pixels.

    Parameters:
    - data: numpy array of shape (A, B, N) where N is the number of features
    - percentile: Percentage of pixels to be considered with the lowest standard deviation

    Returns:
    - avg_vector: numpy array of length N representing the averaged values across the masked pixels
    """

    # Calculate standard deviation along the last axis
    std_dev = np.std(data, axis=2)
    
    # Determine the threshold for the lowest X% standard deviation
    threshold = np.percentile(std_dev, percentile)

    # Generate the mask where std_dev is less than or equal to the threshold
    mask = std_dev <= threshold

    # Plot the mask with flat pixels
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Flat Mask ({percentile}%)')
    plt.axis('off')
    plt.show()

    # Apply the mask and calculate mean intensities
    masked_data = data[mask, :]
    avg_vector = np.mean(masked_data, axis=0)

    return avg_vector

def mask_and_average(data, function='std', percentile=5, threshold='lower', 
                     return_mask=False, show_mask=True):
    """
    Generates a mask for the X% of pixels based on the specified metric along the last axis
    and returns a vector of length N (or N, M) by averaging those masked pixels.

    Parameters:
    - data: numpy array of shape (A, B, N) or (A, B, N, M)
    - function: Metric to calculate ('std_1d', 'sum_2d', etc.)
    - percentile: Percentage of pixels to be considered based on the metric
    - threshold: 'lower' or 'upper' to specify the percentile direction

    Returns:
    - avg_vector: numpy array representing the averaged values across the masked pixels
    """

    # Calculate the specified metric along the last axis
    metric = _calculate_metric(data, function)

    # Determine the threshold based on the specified percentile and direction
    if threshold == 'lower':
        thresh_value = np.percentile(metric, percentile)
        mask = metric <= thresh_value
    elif threshold == 'upper':
        thresh_value = np.percentile(metric, 100 - percentile)
        mask = metric >= thresh_value
    else:
        raise ValueError(f"Unsupported threshold type: {threshold}")
        
    if show_mask:
        # Plot the mask
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask ({threshold} {percentile}%) based on {function}')
        plt.axis('off')
        plt.show()

    # Apply the mask and calculate the mean across the masked pixels
    masked_data = data[mask]
    avg = np.mean(masked_data, axis=0)
    
    if return_mask:
        return avg, mask
    
    else:
        return avg

def _calculate_metric(data, function):
    """
    Calculate the metric based on the specified function.

    Parameters:
    - data: numpy array of shape (A, B, N) or (A, B, N, M)
    - function: String specifying the metric to calculate ('std', 'sum', etc.)

    Returns:
    - metric: 2D numpy array of shape (A, B) containing the calculated metric
    """
    if function == 'std_1d':
        metric = np.std(data, axis=-1)
    elif function == 'sum_2d':
        metric = np.sum(data, axis=(-2, -1))
    else:
        raise ValueError(f"Unsupported function: {function}")
    return metric

def get_average_clusters(dataset, cluster_map,
                         plot_averages=True, vmin=4, vmax=14,cmap='turbo',logScale=False):
    """Average diffraction patterns at scan positions with the same cluster ID.

    Parameters
    ----------
    dataset : HyperData
        Four-dimensional data with shape ``(Ry, Rx, Ky, Kx)``.
    cluster_map : numpy.ndarray
        Integer labels with shape ``(Ry, Rx)``. The current implementation
        expects contiguous labels ``0, 1, ..., K-1``.
    plot_averages : bool, optional
        Display each cluster average before returning it.
    vmin, vmax : float, optional
        Color limits for the displayed images only.
    cmap : str, optional
        Matplotlib colormap for the displayed images.
    logScale : bool, optional
        Display ``log(average + 1)`` rather than ``average + 1``. This does
        not change the returned data.

    Returns
    -------
    HyperData
        Stack of cluster-mean patterns with shape ``(K, Ky, Kx)``. The
        returned object does not currently inherit the input calibration.
    """

    A, B, C, D = dataset.shape
    E, F = cluster_map.shape
    
    assert A == E and B == F, "The 1st and 2nd dimensions of 'dataset' must match the dimensions of 'cluster_map'" 
    
    # Count number of clusters
    n_clusters = len(np.unique(cluster_map))
    average_values = np.zeros((n_clusters, C, D))
    
    # Loop over each cluster
    for cluster_idx in range(n_clusters):
            
        # Calculate the average across all selected patterns for this cluster
        average_values[cluster_idx] = np.mean(dataset.array[cluster_map == cluster_idx], axis=0)
        
    if plot_averages:
        for i in range(n_clusters):
            if logScale:
                plt.imshow(np.log(average_values[i]+1), vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                plt.imshow(average_values[i]+1,vmin=vmin,vmax=vmax,cmap=cmap)
            plt.axis(False)
            plt.show()
    
    return HyperData(average_values)

def get_cluster_masks(clusterMap):
    """
    Separate the clusters in cluster map
    """
    
    A, B = clusterMap.shape
    labels = np.unique(clusterMap)

    cluster_masks = np.zeros((len(labels),A,B), dtype=bool)
    
    for label in labels:
        
        cluster_masks[label][clusterMap == label] = clusterMap[clusterMap == label]
        
def shift_mask_integer(mask: np.ndarray, 
                       dy: int = 0, 
                       dx: int = 0) -> np.ndarray:
    """
    Shift a boolean mask by integer dy, dx with padding of False (no wrap).

    Parameters
    ----------
    mask : np.ndarray of bool
    dy : int
        Shift in the y–direction (positive moves content *down*).
    dx : int
        Shift in the x–direction (positive moves content *right*).

    Returns
    -------
    shifted : np.ndarray of bool
    """
    h, w = mask.shape
    # create empty output
    shifted = np.zeros_like(mask, dtype=bool)

    # source and destination slicing
    # dest y-range
    y0_dst = max(dy, 0)
    y1_dst = min(h + dy, h)
    # dest x-range
    x0_dst = max(dx, 0)
    x1_dst = min(w + dx, w)

    # corresponding source ranges
    y0_src = max(-dy, 0)
    y1_src = min(h - dy, h)
    x0_src = max(-dx, 0)
    x1_src = min(w - dx, w)

    shifted[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return shifted

def fill_nans_1d(arr: np.ndarray) -> np.ndarray:
    """
    Return a copy of the 1D array `arr` where NaNs have been replaced by
    linear interpolation between the surrounding valid points.
    
    Leading/trailing NaNs are filled with the first/last valid value.
    
    Parameters
    ----------
    arr : np.ndarray, shape (N,)
        Input array containing floats and possibly NaNs.
    
    Returns
    -------
    out : np.ndarray, shape (N,)
        Copy of `arr` with NaNs replaced by interpolated values.
    
    Raises
    ------
    ValueError
        If `arr` is not 1D or if all values are NaN.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported")
    
    x = np.arange(arr.size)
    mask_good = ~np.isnan(arr)
    
    if not mask_good.any():
        raise ValueError("Array contains only NaNs; cannot interpolate")
    
    # np.interp: for x < first_good it returns arr[first_good],
    # for x > last_good returns arr[last_good].
    filled = arr.copy()
    filled[np.isnan(arr)] = np.interp(
        x[np.isnan(arr)],
        x[mask_good],
        arr[mask_good]
    )
    return filled

def merge_cluster_maps(cluster_maps):
    """
    Merge N cluster maps (same shape) by fragmenting overlapping regions.
    Each pixel has an N‐tuple of original labels; we assign a new unique
    label to each distinct N‐tuple, and remap background (all zeros) → 0.

    Parameters
    ----------
    cluster_maps : list of np.ndarray, each shape (H, W), integer labels
        The input cluster maps to merge. 0 is background.

    Returns
    -------
    merged_map : np.ndarray[int], shape (H, W)
        Integer map where each unique combination of input‐map labels
        has its own cluster ID (0 for background).
    """
    # stack into (H, W, N) and flatten to (H*W, N)
    stack = np.stack(cluster_maps, axis=-1)
    H, W, N = stack.shape
    flat = stack.reshape(-1, N)

    # build map from tuple of original labels → new ID
    combo_to_id = {}
    next_id = 1
    merged_flat = np.zeros(flat.shape[0], dtype=int)

    for i, row in enumerate(flat):
        key = tuple(row)
        if key not in combo_to_id:
            combo_to_id[key] = next_id
            next_id += 1
        merged_flat[i] = combo_to_id[key]

    # background key is all zeros
    bg_key = tuple([0]*N)
    if bg_key in combo_to_id:
        bg_id = combo_to_id[bg_key]
        # set background pixels to 0
        merged_flat[merged_flat == bg_id] = 0
        # shift down any IDs > bg_id by 1 so labels remain contiguous
        merged_flat = merged_flat - (merged_flat > bg_id).astype(int)

    # reshape back to (H, W)
    merged_map = merged_flat.reshape(H, W)
    return merged_map

from scipy.ndimage import binary_dilation

def filter_and_split_cluster_map(cluster_map: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Split each integer-labeled region into connected-component "islands",
    reassign any island smaller than min_size to the majority neighboring island,
    and renumber islands so labels run from 0..M-1 consecutively.

    Notes
    -----
    - There is no special "background" label; every integer label in `cluster_map`
      is treated the same (including 0, if present).
    - Connected components use 4-connectivity (same default as scipy.ndimage.label).
    - Neighbor voting uses an 8-neighborhood around the island boundary.

    Parameters
    ----------
    cluster_map : np.ndarray[int], shape (H, W)
        Input 2D array of cluster labels (all labels treated equally).
    min_size : int
        Minimum island size to keep. Islands smaller than this are reassigned
        to the most common neighboring island label when possible.

    Returns
    -------
    np.ndarray[int], shape (H, W)
        Filtered and fully-split cluster map, with new labels 0..M-1.
    """
    cluster_map = np.asarray(cluster_map)
    H, W = cluster_map.shape

    comp_global = np.zeros((H, W), dtype=int)
    next_id = 0
    island_sizes: dict[int, int] = {}

    # Step 1: split every label (including 0) into connected-component islands
    for lab in np.unique(cluster_map):
        mask_lab = (cluster_map == lab)
        if not mask_lab.any():
            continue
        comp_lab, num_comp = label(mask_lab)  # default 4-connectivity
        for comp_idx in range(1, num_comp + 1):
            comp_mask = (comp_lab == comp_idx)
            comp_size = int(comp_mask.sum())
            comp_global[comp_mask] = next_id
            island_sizes[next_id] = comp_size
            next_id += 1

    # Step 2: reassign small islands based on majority neighbor (no background special-casing)
    neigh_struct = np.ones((3, 3), dtype=bool)  # 8-neighborhood for boundary expansion

    for island_id in sorted(island_sizes, key=island_sizes.get):
        if island_sizes[island_id] >= min_size:
            continue
        if not np.any(comp_global == island_id):  # may have been merged already
            continue

        mask = (comp_global == island_id)
        border = binary_dilation(mask, structure=neigh_struct) & ~mask
        neigh = comp_global[border]
        neigh = neigh[neigh != island_id]

        if neigh.size:
            target = int(np.bincount(neigh).argmax())
            comp_global[mask] = target
        # else: no neighbors (e.g., single-island map); keep it as-is

    # Step 3: renumber islands consecutively (0..M-1)
    old_ids = np.unique(comp_global)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    new_map = np.vectorize(remap.get, otypes=[int])(comp_global).astype(int)

    return new_map

#%%

def spiral_matrix(matrix, return_indices=True):
    """
    Extract elements from a 2D NumPy array in a spiral order or returns the flat indices of the spiral order.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D array from which to extract elements in a spiral order.
    indices : bool, optional
        If True, returns the flat indices of the elements in spiral order.
        If False, returns the elements themselves.
    
    Returns
    -------
    numpy.ndarray
        An array of values or indices in spiral order.
    
    Examples
    --------
    >>> matrix = np.array([[10, 20], [30, 40]])
    >>> spiral_matrix(matrix, indices=False)
    array([10, 20, 40, 30])
    >>> spiral_matrix(matrix, indices=True)
    array([0, 1, 3, 2])
    
    Notes
    -----
    This function assumes the input matrix is 2D and at least 1x1 in size. The indices are computed relative to the flattened matrix.
    """
    
    
    result = []
    while matrix.size > 0:
        # Add the first row
        result.append(matrix[0, :])
        matrix = matrix[1:]  # Remove the first row
        
        if matrix.size == 0:
            break
        
        # Add the last column
        result.append(matrix[:, -1])
        matrix = matrix[:, :-1]  # Remove the last column
        
        if matrix.size == 0:
            break
        
        # Add the last row reversed
        result.append(matrix[-1, ::-1])
        matrix = matrix[:-1]  # Remove the last row
        
        if matrix.size == 0:
            break
        
        # Add the first column reversed
        result.append(matrix[::-1, 0])
        matrix = matrix[:, 1:]  # Remove the first column

    return np.concatenate(result)

def gradient_ascent(data, start, learning_rate=0.1, max_iters=100):
    """Walk uphill in a 2D image using its local numerical gradient.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional scalar field.
    start : tuple of float
        Initial ``(y, x)`` position in pixel coordinates.
    learning_rate : float, optional
        Step multiplier applied to the gradient at each visited pixel.
    max_iters : int, optional
        Maximum number of update steps.

    Returns
    -------
    tuple of float
        Final ``(y, x)`` position, which can be subpixel.

    Notes
    -----
    Gradient values are sampled at integer-truncated coordinates. The walk
    does not constrain positions to the image bounds; an oversized step can
    produce an indexing error.
    """
    y, x = start
    for i in range(max_iters):
        grad_y, grad_x = np.gradient(data)
        y += learning_rate * grad_y[int(y), int(x)]
        x += learning_rate * grad_x[int(y), int(x)]
        if grad_y[int(y), int(x)] == 0 and grad_x[int(y), int(x)] == 0:
            break
    return y, x

# Gaussian fitting
def gaussian_2d(xdata, y0, x0, yalpha, xalpha, amplitude, offset):
    """Evaluate an axis-aligned 2D Gaussian in ``(y, x)`` coordinates.

    Parameters
    ----------
    xdata : pair of array_like
        Broadcastable ``(y, x)`` coordinate arrays, or a ``(2, N)`` array.
    y0, x0 : float
        Center in pixel coordinates.
    yalpha, xalpha : float
        Gaussian standard deviations along the y and x axes.
    amplitude : float
        Gaussian peak amplitude relative to the baseline.
    offset : float
        Constant baseline.

    Returns
    -------
    numpy.ndarray
        Model values with the broadcast shape of the coordinates.
    """
    y, x = xdata
    return offset + amplitude * np.exp(
        -(((y - y0) ** 2 / (2 * yalpha ** 2)) + ((x - x0) ** 2 / (2 * xalpha ** 2)))
    )

def fit_gaussian_2d(data):
    """Fit an axis-aligned Gaussian and return its ``(y, x)`` center.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image to fit with :func:`gaussian_2d`.

    Returns
    -------
    tuple of float
        Fitted center ``(y0, x0)`` in pixel coordinates. Other fitted
        parameters are not returned.

    Notes
    -----
    The fit is unweighted and unconstrained. Its initial center is the middle
    pixel, with unit widths, maximum image value as amplitude, and zero offset.
    """
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((y.ravel(), x.ravel()))
    initial_guess = (data.shape[0]//2, data.shape[1]//2, 1, 1, data.max(), 0)
    popt, _ = curve_fit(gaussian_2d, xdata, data.ravel(), p0=initial_guess)
    return popt[0], popt[1]

# Elliptical Gaussian fitting
def elliptical_gaussian_2d(xdata, y0, x0, yalpha, xalpha, theta, amplitude, offset):
    """Evaluate a rotated elliptical Gaussian in ``(y, x)`` coordinates.

    Parameters
    ----------
    xdata : pair of array_like
        Broadcastable ``(y, x)`` coordinate arrays, or a ``(2, N)`` array.
    y0, x0 : float
        Center in pixel coordinates.
    yalpha, xalpha : float
        Gaussian standard deviations along the rotated principal axes.
    theta : float
        In-plane rotation angle in radians.
    amplitude : float
        Gaussian peak amplitude relative to the baseline.
    offset : float
        Constant baseline.

    Returns
    -------
    numpy.ndarray
        Model values with the broadcast shape of the coordinates.
    """
    y, x = xdata
    a = (np.cos(theta)**2 / (2 * xalpha**2)) + (np.sin(theta)**2 / (2 * yalpha**2))
    b = -(np.sin(2*theta) / (4 * xalpha**2)) + (np.sin(2*theta) / (4 * yalpha**2))
    c = (np.sin(theta)**2 / (2 * xalpha**2)) + (np.cos(theta)**2 / (2 * yalpha**2))
    return offset + amplitude * np.exp(-(a * ((x - x0)**2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0)**2)))

def fit_elliptical_gaussian_2d(data):
    """Fit a rotated elliptical Gaussian and return its ``(y, x)`` center.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image to fit with :func:`elliptical_gaussian_2d`.

    Returns
    -------
    tuple of float
        Fitted center ``(y0, x0)`` in pixel coordinates. Widths, angle,
        amplitude, and offset are fitted but not returned.

    Notes
    -----
    The fit is unweighted and unconstrained. It starts at the middle pixel
    with unit widths, zero rotation, maximum image value as amplitude, and
    zero offset.
    """
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((y.ravel(), x.ravel()))
    initial_guess = (data.shape[0]//2, data.shape[1]//2, 1, 1, 0, data.max(), 0)
    popt, _ = curve_fit(elliptical_gaussian_2d, xdata, data.ravel(), p0=initial_guess)
    return popt[0], popt[1]

def mask_corrupted_pixels(arr, threshold=3, window_size=3):
    """
    Identify and mask corrupted pixels in a 2D array based on a threshold difference
    from the neighboring area, which is defined by the window size.

    Parameters
    ----------
    arr : np.ndarray
        The input 2D array.
    threshold : float, optional
        The threshold difference to identify corrupted pixels. A pixel is considered
        corrupted if its value differs from the mean of its neighbors by more than
        this threshold. Default is 3.
    window_size : int, optional
        The size of the window used to define the neighboring area around each pixel.
        Must be an odd integer. Default is 3.

    Returns
    -------
    mask : np.ndarray
        A 2D boolean array where `True` indicates a corrupted pixel.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")
    
    # Initialize the mask with False
    mask = np.zeros_like(arr, dtype=bool)
    
    # Calculate the offset based on the window size
    offset = window_size // 2
    
    # Get the dimensions of the array
    rows, cols = arr.shape
    
    # Iterate over each pixel (excluding the border pixels)
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            # Extract the pixel's neighboring area based on the window size
            neighbors = arr[i-offset:i+offset+1, j-offset:j+offset+1]
            neighbors_mean = np.std(neighbors)
            
            # Calculate the difference between the pixel and the mean of its neighbors
            if np.abs(arr[i, j] - neighbors_mean) > threshold:
                mask[i, j] = True
                
    return mask

def get_gradients(sol_array, rot_angle=0, show_result=False, invertY=False, invertX=False, cbar_fraction=0.04):
    """Calculate directional gradients from a solution array of (phi, theta) values.

    This function computes the directional gradients (dz/dx and dz/dy) from a solution array that 
    contains azimuthal (phi) and elevation (theta) angles. The gradients are computed considering 
    a specified rotation angle. Optionally, the gradients can be visualized as 2D images.

    Parameters
    ----------
    sol_array : ndarray
        A 3D numpy array of shape (rows, cols, 2) where the last dimension represents the angles 
        (phi, theta) in radians.
    rot_angle : float, optional
        The rotation angle in degrees to adjust the azimuthal angle (phi). Default is 0.
    show_result : bool, optional
        If True, displays the gradients as images with colorbars. Default is False.
    invertY : bool, optional
        If True, inverts the y-axis gradient. Default is False.
    invertX : bool, optional
        If True, inverts the x-axis gradient. Default is False.

    Returns
    -------
    array_gradients : ndarray
        A 3D numpy array of shape (rows, cols, 2) containing the computed gradients. 
        The last dimension holds the gradients (dz/dx, dz/dy).

    Examples
    --------
    Given a solution array `sol_array` with shape (100, 100, 2):
    
    >>> gradients = get_gradients(sol_array, rot_angle=45, show_result=True)
    This will calculate the gradients after rotating phi by 45 degrees and display the results.

    Notes
    -----
    The function assumes that the input array contains azimuthal angles in the first channel 
    (phi) and elevation angles in the second channel (theta). The rotation angle is applied 
    to the azimuthal angle before computing the gradients. The option to invert the gradients 
    allows flexibility in adjusting the gradient directions based on the specific application.
    """

    # Adjust phi by the rotation angle
    phi = sol_array[:, :, 0] - rot_angle * np.pi / 180
    theta = sol_array[:, :, 1]
    
    # Compute gradients
    dzdy = np.tan(theta) * np.cos(phi)
    dzdx = np.tan(theta) * np.sin(phi)

    # Apply inversion if required
    if invertY:
        dzdy = -dzdy
    if invertX:
        dzdx = -dzdx

    # Stack gradients into a single array
    array_gradients = np.stack((dzdx, dzdy), axis=-1)
    
    if show_result:
        # Plotting the gradients
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Gradient X
        im1 = axs[0].imshow(array_gradients[:,:,0], cmap='RdBu')
        axs[0].axis('off')
        axs[0].set_title('Gradient X', fontsize=10)
        cbar1 = fig.colorbar(im1, ax=axs[0], fraction=cbar_fraction)
        cbar1.ax.set_title(r'$\nabla_X$', fontsize=8, pad=5)
        
        # Gradient Y
        im2 = axs[1].imshow(array_gradients[:,:,1], cmap='RdBu')
        axs[1].axis('off')
        axs[1].set_title('Gradient Y', fontsize=10)
        cbar2 = fig.colorbar(im2, ax=axs[1], fraction=cbar_fraction)
        cbar2.ax.set_title(r'$\nabla_Y$', fontsize=8, pad=5)
        
        plt.tight_layout()
        plt.show()
    
    return array_gradients

def correct_strain(exx, eyy, exy, phi, theta):
    """
    Correct strain arrays for in-plane surface tilt
    
    The input strain arrays `exx`, `eyy`, and `exy` must be corrected to account for 
    the in-plane tilt of the surface, described by the azimuthal (`phi`) and elevation (`theta`) angles.
    
    Parameters
    ----------
    exx : array
       Uniaxial strain array in the x-direction
    eyy : array
       Uniaxial strain array in the y-direction
    exy : array
       Shear strain array
    phi : array
       Array of azimuthal angles for each position's local tilt
    theta : array
       Array of elevation angles for each position's local tilt
    
    Returns
    -------
    corrected_exx : array
       Corrected uniaxial strain array in the x-direction
    corrected_eyy : array
       Corrected uniaxial strain array in the y-direction
    corrected_exy : array
       Corrected shear strain array
    
    Examples
    --------
    Correct strain arrays with given tilt angles:
    >>> corrected_exx, corrected_eyy, corrected_exy = correct_strain(exx, eyy, exy, phi, theta)
    Corrected arrays aligned to the tilt angles
    
    Notes
    -----
    This function applies Mohr's circle transformations to align with `phi` 
    and then corrects for the tilt in `theta`.
    """
                    
    # Apply Mohr's circle transformations
    e_parallel =       (exx + eyy)/2 + (exx - eyy)/2*np.cos(2*phi) + exy*np.sin(2*phi)
    e_perpendicular =  (exx + eyy)/2 - (exx - eyy)/2*np.cos(2*phi) - exy*np.sin(2*phi)
    e_shear =         -(exx - eyy)/2*np.sin(2*phi) + exy*np.cos(2*phi)
    
    # Apply strain correction using physical tilt information
    e_parallel = (e_parallel + 1)/np.cos(theta) - 1
    
    # Return to original orientation
    exx_corr =  (e_parallel + e_perpendicular)/2 + (e_parallel - e_perpendicular)/2*np.cos(-2*phi) + e_shear*np.sin(-2*phi)
    eyy_corr =  (e_parallel + e_perpendicular)/2 - (e_parallel - e_perpendicular)/2*np.cos(-2*phi) - e_shear*np.sin(-2*phi)
    exy_corr =  -(e_parallel - e_perpendicular)/2*np.sin(-2*phi) + e_shear*np.cos(-2*phi)

    return exx_corr, eyy_corr, exy_corr


def rotate_strain(exx, eyy, exy, alpha):
    """

    Parameters
    ----------
    exx : array
       Uniaxial strain array in the x-direction
    eyy : array
       Uniaxial strain array in the y-direction
    exy : array
       Shear strain array
    alpha : float
       Angle (radians)
    
    Returns
    -------
    e_parallel : array
       uniaxial strain array in the alpha-direction
    e_perpendicular : array
       uniaxial strain array in the direction perpendicular to alpha
    e_shear : array
       shear strain array
    
    Examples
    --------
    Correct strain arrays with given tilt angles:
    >>> corrected_exx, corrected_eyy, corrected_exy = correct_strain(exx, eyy, exy, phi, theta)
    Corrected arrays aligned to the tilt angles
    
    Notes
    -----
    This function applies Mohr's circle transformations to align with `alpha` 
    and then corrects for the tilt in `theta`.
    """
                    
    # Apply Mohr's circle transformations
    e_parallel =       (exx + eyy)/2 + (exx - eyy)/2*np.cos(2*alpha) + exy*np.sin(2*alpha)
    e_perpendicular =  (exx + eyy)/2 - (exx - eyy)/2*np.cos(2*alpha) - exy*np.sin(2*alpha)
    e_shear =         -(exx - eyy)/2*np.sin(2*alpha) + exy*np.cos(2*alpha)
    
    return e_parallel, e_perpendicular, e_shear       

def bin_array_with_padding(array, new_shape, padding=10):
    """
    Bin an array by averaging over the first two axes (ny, nx) 
    to resize it to the new shape (ny_new, nx_new), with white padding.
    
    Parameters:
        array (ndarray): Input array of shape (ny, nx, ky, kx).
        new_shape (tuple): Desired shape for the first two axes (ny_new, nx_new).
        padding (int): Number of pixels for the white padding between images.
        
    Returns:
        visualization_grid (ndarray): 2D array representing the padded grid visualization.
    """
    ny, nx, ky, kx = array.shape
    ny_new, nx_new = new_shape

    # Compute binning factors
    bin_size_y = ny / ny_new
    bin_size_x = nx / nx_new

    # Initialize the binned array
    binned_array = np.zeros((ny_new, nx_new, ky, kx))

    for i in range(ny_new):
        for j in range(nx_new):
            # Determine the slice indices for the bin
            y_start = int(i * bin_size_y)
            y_end = int((i + 1) * bin_size_y)
            x_start = int(j * bin_size_x)
            x_end = int((j + 1) * bin_size_x)

            # Take the average over the selected slice
            binned_array[i, j] = array[y_start:y_end, x_start:x_end].mean(axis=(0, 1))

    # Pad each image with white space
    padded_images = np.array(
        [
            [
                np.pad(
                    binned_array[i, j], 
                    pad_width=((padding, padding), (padding, padding)), 
                    mode="constant", 
                    constant_values=0
                )
                for j in range(nx_new)
            ]
            for i in range(ny_new)
        ]
    )

    # Combine the padded images into a grid
    visualization_grid = np.block([[padded_images[i, j] for j in range(nx_new)] for i in range(ny_new)])

    return visualization_grid


def rolling_ball_background(image, radius, smooth=True, smooth_sigma_frac=0.25):
    """
    Estimate a smooth 2D background using a rolling-ball (sphere) algorithm.

    Parameters
    ----------
    image : (H, W) array_like
        Input 2D image (e.g., diffraction pattern). Casts to float.
    radius : float
        Ball radius in pixels. Sets the length scale of the background.
    smooth : bool, optional
        If True, apply a small Gaussian blur to the background at the end.
    smooth_sigma_frac : float, optional
        Fraction of `radius` used as Gaussian sigma when `smooth=True`.

    Returns
    -------
    background : (H, W) ndarray
        Estimated background image.
    """
    im = np.asarray(image, dtype=float)

    # Build a spherical structuring element (ball) in 2D
    r = int(np.ceil(radius))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    dist2 = xx*xx + yy*yy
    mask = dist2 <= radius**2

    ball = np.zeros_like(dist2, dtype=float)
    ball[mask] = np.sqrt(radius**2 - dist2[mask])
    # Top of the ball at 0, rest <= 0 (what grey_* expects)
    ball -= ball.max()

    # Morphological opening with a spherical structuring element
    eroded = grey_erosion(im, footprint=mask, structure=ball)
    background = grey_dilation(eroded, footprint=mask, structure=ball)

    if smooth and radius > 0:
        sigma = smooth_sigma_frac * radius
        background = gaussian_filter(background, sigma=sigma)

    return background

def split_disconnected_clusters(cluster_map,
                                connectivity: int = 2,
                                background: int = -1):
    """
    Split disconnected "islands" within each cluster label into separate clusters.

    Parameters
    ----------
    cluster_map : (H, W) ndarray of int
        2D map of cluster labels. Pixels with the same integer label belong
        to the same cluster (before splitting).
    connectivity : {1, 2}, optional
        Connectivity for defining local adjacency:
        - 1 → 4-connected (up, down, left, right)
        - 2 → 8-connected (also includes diagonals)
    background : int, optional
        Label to treat as background (left unchanged and not split).

    Returns
    -------
    new_map : (H, W) ndarray of int
        Relabeled map where every spatially connected component that was part
        of a given label gets its own unique new label. New labels are
        consecutive integers starting from 0 (excluding background).
    mapping : dict
        Dictionary describing how old labels were split:
        keys:   (old_label, component_index) where component_index ∈ {1..n_islands}
        values: new_label (int) in `new_map`.

        Example:
            mapping[(3, 1)] = 0   # first island of old label 3 → new label 0
            mapping[(3, 2)] = 1   # second island of old label 3 → new label 1

    Notes
    -----
    - Two regions with the same original label but not touching (given the
      specified connectivity) will become distinct clusters in `new_map`.
    - The background label is preserved as-is wherever it appears.
    """

    cluster_map = np.asarray(cluster_map)
    if cluster_map.ndim != 2:
        raise ValueError("cluster_map must be a 2D array.")

    H, W = cluster_map.shape
    new_map = np.full((H, W), background, dtype=int)

    labels = np.unique(cluster_map)
    struct = ndimage.generate_binary_structure(2, connectivity)

    next_label = 0
    mapping = {}

    for lab in labels:
        if lab == background:
            # copy background pixels directly
            new_map[cluster_map == lab] = background
            continue

        mask = (cluster_map == lab)
        if not mask.any():
            continue

        # Label connected components within this mask
        comp_map, n_comp = ndimage.label(mask, structure=struct)

        for comp_id in range(1, n_comp + 1):
            comp_mask = (comp_map == comp_id)
            new_map[comp_mask] = next_label
            mapping[(lab, comp_id)] = next_label
            next_label += 1

    return new_map, mapping

def combine_strain_maps(strain_maps: np.ndarray,
                        all_intensities: np.ndarray,
                        peaks: np.ndarray,
                        n_fold: int = 6,
                        intensity_power: float = 1.0,
                        intensity_percentile: float | None = None,
                        min_total_intensity_frac: float = 0.0,
                        eps: float = 1e-12) -> np.ndarray:
    """
    Combine per-order strain maps into a final 4-component strain field using
    "smart" intensity weighting.

    Parameters
    ----------
    strain_maps : ndarray, shape (num_orders, 4, Ny, Nx)
        Strain/rotation maps per order (group of n_fold peaks).
        `strain_maps[o, s, y, x]` is the s-th strain component for order o
        at pixel (y, x). s = 0,1,2,3 → e_xx, e_yy, e_xy, rot (or similar).
    all_intensities : ndarray, shape (Ny, Nx, n_peaks)
        Bragg peak intensities for each peak at each pixel.
    peaks : ndarray, shape (n_peaks, 2)
        Peak coordinates (not used directly, only n_peaks is checked).
        Assumed grouped in contiguous blocks of size n_fold:
        [0..n_fold-1], [n_fold..2*n_fold-1], ...
    n_fold : int, optional
        Number of peaks per order (e.g. 6 for hexagonal, 4 for square).
    intensity_power : float, optional
        Exponent applied to intensities before weighting:
          - 1.0  → linear weighting by intensity
          - 0.5  → sub-linear (softens dominance of very bright peaks)
          - 2.0  → super-linear (emphasizes very bright peaks)
        Effective intensity is I_eff = I ** intensity_power.
    intensity_percentile : float in [0, 100] or None, optional
        If not None, intensities are clipped at this percentile of I_eff
        (over all peaks/pixels) before weighting. This prevents a few
        extreme pixels from dominating:
          vmax = percentile(|I_eff|, intensity_percentile)
          I_eff = clip(I_eff, 0, vmax)
    min_total_intensity_frac : float in [0, 1], optional
        Threshold for masking low-intensity pixels in the final map.
        Let T(x,y) = sum over all peaks of I_eff(x,y,peak). We compute
        T_max = T.max() and treat pixels with
            T(x,y) < min_total_intensity_frac * T_max
        as unreliable (weights too small). For those pixels, output is set
        to zero.
        Set = 0.0 to disable masking.
    eps : float, optional
        Small constant to avoid divide-by-zero.

    Returns
    -------
    final_strain_maps : ndarray, shape (4, Ny, Nx)
        Combined strain/rotation maps, intensity-weighted in a "smart" way.

    Notes
    -----
    The combination is:

        For each order o:
            group_weight_o(x,y) = sum_{k in group(o)} I_eff_k(x,y)

        Numerator_s(x,y) = sum_o [ strain_maps[o, s, x, y] * group_weight_o(x,y) ]
        Denominator(x,y) = sum_o group_weight_o(x,y)

        final_strain_maps[s, x, y] = Numerator_s(x,y) / Denominator(x,y)

    where I_eff_k is the possibly exponentiated and clipped intensity.
    """
    # ----------------- basic sanity checks ----------------- #
    Ny, Nx, n_peaks = all_intensities.shape
    num_orders = peaks.shape[0] // n_fold

    assert strain_maps.shape[0] == num_orders, (
        f"strain_maps.shape[0]={strain_maps.shape[0]} does not match "
        f"num_orders={num_orders} inferred from peaks and n_fold."
    )
    assert all_intensities.shape[2] == peaks.shape[0], (
        "Last dimension of all_intensities must match number of peaks."
    )

    # ----------------- build effective intensities ----------------- #
    I_eff = all_intensities.astype(float)

    # exponent on intensity
    if intensity_power != 1.0:
        I_eff = np.power(I_eff, intensity_power)

    # optional global clipping (percentile)
    if intensity_percentile is not None:
        vals = I_eff.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            vmax = np.percentile(vals, intensity_percentile)
            if vmax > 0:
                I_eff = np.clip(I_eff, 0.0, vmax)

    # ----------------- group weights per order ----------------- #
    # group_weight[o, y, x] = sum_{k in group o} I_eff[y, x, k]
    group_weights = np.zeros((num_orders, Ny, Nx), dtype=float)

    for o in range(num_orders):
        sl = slice(o * n_fold, (o + 1) * n_fold)
        group_weights[o] = np.sum(I_eff[:, :, sl], axis=2)

    # Denominator: sum over all groups (equiv. sum over all peaks)
    denominator = np.sum(group_weights, axis=0)  # (Ny, Nx)

    # ----------------- build numerator ----------------- #
    final_strain_maps = np.zeros((4, Ny, Nx), dtype=float)
    numerator = np.zeros_like(final_strain_maps)

    for o in range(num_orders):
        gw = group_weights[o]  # (Ny, Nx)
        for s in range(4):
            numerator[s] += strain_maps[o, s] * gw

    # ----------------- normalize, apply masking ----------------- #
    # global threshold for low-intensity pixels
    if min_total_intensity_frac > 0.0:
        T_max = np.nanmax(denominator) if np.isfinite(denominator).any() else 0.0
        thresh = min_total_intensity_frac * T_max
        valid_mask = denominator > max(thresh, eps)
    else:
        valid_mask = denominator > eps

    with np.errstate(divide='ignore', invalid='ignore'):
        final_strain_maps = np.divide(
            numerator,
            denominator[np.newaxis, :, :],
            out=np.zeros_like(numerator),
            where=valid_mask[np.newaxis, :, :]
        )

    return final_strain_maps


def _peak_radii(r, num_peaks):
    """Return one integration radius per peak without changing the input."""
    radii = np.asarray(r, dtype=float)
    if radii.ndim == 0:
        return np.full(num_peaks, radii.item())
    if radii.shape != (num_peaks,):
        raise ValueError("r must be a scalar or contain one radius per peak.")
    return radii


def _spot_center_from_array(array, ky, kx, r, method='CoM', plotSpot=False):
    """Refine one peak in a local zero-padded window of a 2D array."""
    pad_width = int(np.ceil(r))
    ky_padded, kx_padded = ky + pad_width, kx + pad_width
    area_size = int(np.ceil(r * 2))
    half = area_size // 2
    mask = circular_mask(half, half, r)

    ymin, ymax = int(ky_padded - half), int(ky_padded + half) + 1
    xmin, xmax = int(kx_padded - half), int(kx_padded + half) + 1
    height, width = array.shape
    py0, py1, _ = slice(ymin, ymax).indices(height + 2 * pad_width)
    px0, px1, _ = slice(xmin, xmax).indices(width + 2 * pad_width)
    sy0, sy1 = max(py0, pad_width), min(py1, pad_width + height)
    sx0, sx1 = max(px0, pad_width), min(px1, pad_width + width)

    if sy0 == py0 and sy1 == py1 and sx0 == px0 and sx1 == px1:
        spot_data = array[
            sy0 - pad_width:sy1 - pad_width,
            sx0 - pad_width:sx1 - pad_width,
        ]
    else:
        spot_data = np.zeros(
            (max(py1 - py0, 0), max(px1 - px0, 0)), dtype=array.dtype
        )
        if sy0 < sy1 and sx0 < sx1:
            spot_data[
                sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0
            ] = array[
                sy0 - pad_width:sy1 - pad_width,
                sx0 - pad_width:sx1 - pad_width,
            ]

    if spot_data.shape != mask.shape:
        side = min(*spot_data.shape, *mask.shape)
        spot_data, mask = spot_data[:side, :side], mask[:side, :side]
    masked_spot_data = spot_data * mask

    if method == 'CoM':
        com_y, com_x = center_of_mass(masked_spot_data)
    elif method == 'gaussian':
        com_y, com_x = fit_gaussian_2d(masked_spot_data)
    elif method == 'elliptical_gaussian':
        com_y, com_x = fit_elliptical_gaussian_2d(masked_spot_data)
    else:
        raise ValueError("method must be 'CoM', 'gaussian', or 'elliptical_gaussian'.")

    if plotSpot:
        base_cmap = plt.cm.turbo
        custom_cmap = ListedColormap(np.concatenate((
            [np.array([1, 1, 1, 1])],
            base_cmap(np.linspace(0, 1, 2**12))[1:],
        ), axis=0))
        plt.imshow(masked_spot_data, cmap=custom_cmap)
        plt.colorbar()
        plt.scatter(com_x, com_y, color='yellow', s=50, label='Refined center')
        plt.show()

    return com_y + ymin - pad_width, com_x + xmin - pad_width


def _peak_centers_from_array(array, r, ref_coords, method='CoM', show=False):
    """Refine all reference peaks in one 2D diffraction pattern."""
    ref_coords = np.asarray(ref_coords)
    num_peaks = len(ref_coords)
    radii = _peak_radii(r, num_peaks)
    centers = np.zeros((num_peaks, 2))
    for j in range(num_peaks):
        centers[j] = _spot_center_from_array(
            array, ref_coords[j, 0], ref_coords[j, 1],
            radii[j] + 1e-10, method, show,
        )
    return centers


def _peak_intensities_from_array(array, r, centers, return_pixel_counts=False):
    """Integrate local circular windows, optionally counting included pixels."""
    num_peaks = len(centers)
    radii = _peak_radii(r, num_peaks)
    ints = np.zeros(num_peaks)
    pixel_counts = np.zeros(num_peaks, dtype=int) if return_pixel_counts else None
    for int_idx, radius in enumerate(radii):
        cy, cx = centers[int_idx]
        y0 = max(0, min(array.shape[0], round(cy - (radius + 0.5))))
        y1 = max(0, min(array.shape[0], round(cy + (radius + 0.5))))
        x0 = max(0, min(array.shape[1], round(cx - (radius + 0.5))))
        x1 = max(0, min(array.shape[1], round(cx + (radius + 0.5))))
        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        local_mask = (yy - cy)**2 + (xx - cx)**2 <= (radius + 1e-10)**2
        ints[int_idx] = np.sum(array[y0:y1, x0:x1] * local_mask)
        if return_pixel_counts:
            pixel_counts[int_idx] = np.count_nonzero(local_mask)
    return (ints, pixel_counts) if return_pixel_counts else ints

#%% Strain maps and their scan calibration

@dataclass(slots=True, repr=False, eq=False)
class StrainResult:
    """Strain maps and fit quality from :meth:`HyperData.get_strains`.

    ``exx``, ``eyy``, and ``exy`` are dimensionless; ``erot`` is in radians.
    ``fit_rmse`` is the final peak-position residual in the coordinate units
    supplied to ``get_strains``. ``relative_fit_rmse`` divides it by the RMS
    reference-peak distance from ``peak_origin`` and is dimensionless. A low
    residual with only a few matched peaks does not establish a reliable fit;
    inspect ``match_counts`` and ``valid_mask`` too.

    The maps use the source scan calibration only when their shape matches the
    source 4D scan. ``as_real_space`` is available for 2D maps. The result can
    also be indexed or unpacked like the historical four-array return; when
    ``return_transform=True``, the diagnostics dictionary is item five.
    """

    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    erot: np.ndarray
    fit_rmse: np.ndarray
    relative_fit_rmse: np.ndarray
    match_counts: np.ndarray
    initial_match_counts: np.ndarray
    outlier_counts: np.ndarray
    peak_origin: tuple[float, float]
    peak_origin_source: str
    basis_angle_deg: float
    ewpc: bool
    peak_units: str = 'pixels'
    real_units: str | None = None
    real_conv_factor: float | tuple[float, float] | None = None
    real_origin: tuple[float, float] = (0.0, 0.0)
    diagnostics: dict | None = None

    @property
    def valid_mask(self):
        """Locations with a successful, finite strain fit."""
        return (
            (self.match_counts > 0)
            & np.isfinite(self.exx)
            & np.isfinite(self.eyy)
            & np.isfinite(self.exy)
            & np.isfinite(self.erot)
        )

    def as_real_space(self, component='exx'):
        """Return a 2D map with the source scan-axis calibration.

        ``component`` may be a strain component, ``fit_rmse``,
        ``relative_fit_rmse``, a peak-count map, or ``valid_mask``.
        """
        quantities = {
            'exx': ('Strain exx', None),
            'eyy': ('Strain eyy', None),
            'exy': ('Strain exy', None),
            'erot': ('Rotation', 'rad'),
            'fit_rmse': ('Peak-fit RMSE', self.peak_units),
            'relative_fit_rmse': ('Relative peak-fit RMSE', None),
            'match_counts': ('Matched peaks', 'peaks'),
            'initial_match_counts': ('Initially matched peaks', 'peaks'),
            'outlier_counts': ('Rejected peaks', 'peaks'),
            'valid_mask': ('Valid strain fits', None),
        }
        if component not in quantities:
            raise ValueError(
                f"Unknown strain component {component!r}; choose from "
                f"{', '.join(quantities)}."
            )
        values = getattr(self, component)
        if values.ndim != 2:
            raise ValueError(
                'as_real_space requires a 2D scan map; this result is a 1D stack.'
            )
        quantity, value_units = quantities[component]
        return RealSpace(
            values,
            units=self.real_units,
            conv_factor=self.real_conv_factor,
            origin=self.real_origin,
            quantity=quantity,
            value_units=value_units,
        )

    def _tuple_values(self):
        maps = (self.exx, self.eyy, self.exy, self.erot)
        return maps if self.diagnostics is None else maps + (self.diagnostics,)

    def __iter__(self):
        return iter(self._tuple_values())

    def __len__(self):
        return 4 if self.diagnostics is None else 5

    def __getitem__(self, index):
        return self._tuple_values()[index]

    def __repr__(self):
        return (
            f'StrainResult(shape={self.exx.shape}, '
            f'real_units={self.real_units!r}, '
            f'diagnostics={self.diagnostics is not None})'
        )


#%% The main 4D-STEM object

class HyperData:
    """Hold diffraction data and its real- and reciprocal-space calibration.

    The final two axes are diffraction coordinates ``(Ky, Kx)``. A 4D array
    has scan axes ``(Ry, Rx, Ky, Kx)``; a 3D array is a stack of patterns
    ``(N, Ky, Kx)`` whose original 2D scan layout is not assumed. A 2D array
    represents one image or pattern. Input may be an array or a supported
    data-file path. Methods provide loading,
    visualization, preprocessing, denoising, peak analysis, and strain maps.

    See :meth:`__init__` for file-format, calibration, and metadata options.
    """

    def __init__(self, data,
                 real_units: str = None,
                 real_conv_factor: float | tuple[float, float] = None,
                 reciprocal_units: str = None,
                 reciprocal_conv_factor: float = None,
                 polar_metadata: dict = None,
                 center_beam_metadata: dict = None,
                 hdf5_dataset=None,
                 flip_axis=None,
                 real_origin=None,
                 clip_on_load=False,
                 repair_nans=False,
                 raw_shape=None,
                 raw_dtype=np.float32,
                 raw_order='C',
                 raw_trim_meta=None,
                 raw_trim_dims=(128, 128),
                 mat_variable=None,
                 rq_calibration=None):
        """Wrap an array or load a dataset with optional axis reversal.

        ``flip_axis`` accepts one axis or a sequence of axes to reverse. For
        4D data, axes ``(0, 1, 2, 3)`` mean ``(Ry, Rx, Ky, Kx)``. For 3D
        data, axes ``(0, 1, 2)`` mean ``(pattern, Ky, Kx)``. Negative axes
        follow NumPy conventions. Flipping a NumPy array creates a view.
        ``real_conv_factor`` may be scalar or ``(y, x)`` units per pixel.
        Negative steps indicate axes reversed relative to physical coordinates.
        ``real_origin`` is the calibrated coordinate of scan pixel ``(0, 0)``
        and defaults to ``(0, 0)``.
        ``scan_shape`` contains all leading (non-pattern) axes and
        ``pattern_shape`` contains the last two axes. ``real_shape`` is the
        2D scan grid only for 4D data; a 3D stack has ``scan_shape=(N,)``
        and ``real_shape=None`` because its scan geometry is unknown.
        For generic files, ``clip_on_load`` replaces values below 1 with 1 and
        ``repair_nans`` replaces NaN-containing patterns with neighbor averages.
        Both are opt-in; loading preserves the stored data by default.
        For ``.raw`` files, ``raw_shape`` specifies the full stored shape;
        ``raw_dtype`` and ``raw_order`` specify the binary layout. Explicit
        ``raw_shape`` disables EMPAD metadata-row trimming by default. Set
        ``raw_trim_meta=True`` and ``raw_trim_dims`` to request that crop.
        For MATLAB ``.mat`` files, ``mat_variable`` selects a named numeric
        array; it is required when the file has multiple suitable arrays.
        Assigning a new ``array`` later refreshes the shape, dtype, and
        denoising engine; shape-dependent metadata is cleared when necessary.
        ``rq_calibration`` is an optional :class:`RQCalibration` or saved
        calibration dictionary describing real-to-detector orientation.
        Saved files restore it automatically; axis flips update its frame.
        """
        loaded_metadata = {}

        # Read dataset from file path if input object is string/path-like.
        if isinstance(data, (str, Path)):
            data_path = Path(data).expanduser()
            if mat_variable is not None and data_path.suffix.lower() != '.mat':
                raise ValueError("mat_variable applies only to .mat files.")
            if raw_shape is not None and data_path.suffix.lower() != '.raw':
                raise ValueError("raw_shape applies only to .raw files.")
            if _is_hyperdata_hdf5_file(data_path):
                data, loaded_metadata = _load_hyperdata_hdf5(data_path)
            elif data_path.suffix.lower() == '.4denoise':
                if not data_path.exists():
                    raise FileNotFoundError(f"'{data_path}' does not exist.")
                raise ValueError(
                    f"'{data_path}' has a .4denoise extension but is not a "
                    "valid 4Denoise HyperData save file."
                )
            else:
                data = read_4D(
                    str(data_path),
                    hdf5_dataset=hdf5_dataset,
                    mat_variable=mat_variable,
                    clip=clip_on_load,
                    repair_nans=repair_nans,
                    raw_shape=raw_shape,
                    raw_dtype=raw_dtype,
                    raw_order=raw_order,
                    trim_meta=raw_trim_meta,
                    trim_dims=raw_trim_dims,
                )
        elif mat_variable is not None:
            raise ValueError("mat_variable applies only to .mat files.")

        if (
            real_units is None
            and real_conv_factor is None
            and loaded_metadata
        ):
            real_units = loaded_metadata.get('real_units')
            real_conv_factor = loaded_metadata.get('real_conv_factor')
        if (
            reciprocal_units is None
            and reciprocal_conv_factor is None
            and loaded_metadata
        ):
            reciprocal_units = loaded_metadata.get('reciprocal_units')
            reciprocal_conv_factor = loaded_metadata.get('reciprocal_conv_factor')
        if polar_metadata is None and loaded_metadata:
            polar_metadata = loaded_metadata.get('polar_metadata')
        if center_beam_metadata is None and loaded_metadata:
            center_beam_metadata = loaded_metadata.get('center_beam_metadata')
        if real_origin is None:
            real_origin = loaded_metadata.get('real_origin', (0.0, 0.0))
        if rq_calibration is None:
            rq_calibration = loaded_metadata.get('rq_calibration')

        if data.ndim < 2:
            raise ValueError(
                "HyperData requires at least two spatial axes; expected a "
                "2D image, 3D image stack, or 4D scan."
            )

        flip_axes = self._normalize_flip_axes(flip_axis, data.ndim)
        if polar_metadata is not None and any(
            axis in (data.ndim - 2, data.ndim - 1) for axis in flip_axes
        ):
            raise ValueError(
                "Flipping polar radius or angle axes would invalidate polar "
                "metadata. Convert to Cartesian coordinates before flipping "
                "reciprocal-space axes."
            )
        if flip_axes:
            data = np.flip(data, axis=flip_axes)
            if data.ndim == 4 and real_conv_factor is not None:
                steps = list(_real_spacing_pair(
                    _normalize_real_spacing(real_conv_factor)
                ))
                origin = list(_normalize_real_origin(real_origin))
                for axis in (0, 1):
                    if axis in flip_axes:
                        origin[axis] += (data.shape[axis] - 1) * steps[axis]
                        steps[axis] *= -1
                real_origin = tuple(origin)
                real_conv_factor = (
                    steps[0] if np.isclose(steps[0], steps[1])
                    else tuple(steps)
                )

        self.array = data
        self.real_units = None
        self.real_conv_factor = None
        self.real_origin = _normalize_real_origin(real_origin)
        self.rq_calibration = rq_calibration
        if flip_axes and self.rq_calibration is not None:
            real_flip = np.eye(2)
            q_flip = np.eye(2)
            for axis in flip_axes:
                if axis == self.ndim - 1:
                    q_flip[0, 0] = -1
                elif axis == self.ndim - 2:
                    q_flip[1, 1] = -1
                elif self.ndim == 4:
                    real_flip[1 - axis, 1 - axis] = -1
            self.rq_calibration = _reframe_rq_calibration(
                self.rq_calibration, real=real_flip, reciprocal=q_flip,
            )
        self.reciprocal_units = None
        self.reciprocal_conv_factor = None
        self.unfold_metadata = _clone_unfold_metadata(
            loaded_metadata.get('unfold_metadata')
            if loaded_metadata and not flip_axes
            else None,
            share_payload=True,
        )
        self.polar_metadata = deepcopy(polar_metadata) if polar_metadata is not None else None
        self.center_beam_metadata = (
            deepcopy(center_beam_metadata)
            if center_beam_metadata is not None
            else None
        )
        if self.center_beam_metadata is not None and any(
            axis in (self.ndim - 2, self.ndim - 1) for axis in flip_axes
        ):
            self._flip_center_beam_metadata(flip_axes)

        if real_units is not None or real_conv_factor is not None:
            self.set_real_scale(real_units, real_conv_factor)
        if reciprocal_units is not None or reciprocal_conv_factor is not None:
            self.set_reciprocal_scale(reciprocal_units, reciprocal_conv_factor)

    @property
    def array(self):
        """Data array; assignment refreshes cached geometry and denoising state."""
        return self._array

    @array.setter
    def array(self, data):
        """Replace data while invalidating metadata tied to changed axes."""
        new_array = np.asarray(data)
        if new_array.ndim < 2:
            raise ValueError(
                "HyperData requires at least two spatial axes; expected a "
                "2D image, 3D image stack, or 4D scan."
            )

        old_shape = self.__dict__.get('shape')
        engine = _DenoiseEngine(new_array)
        self._array = new_array
        self.ndim = new_array.ndim
        self.shape = new_array.shape
        self.scan_shape = tuple(new_array.shape[:-2])
        self.pattern_shape = tuple(new_array.shape[-2:])
        self.real_shape = tuple(new_array.shape[:2]) if new_array.ndim == 4 else None
        self.k_shape = self.pattern_shape
        self.dtype = new_array.dtype
        self._denoise_engine = engine

        if old_shape is not None and old_shape != new_array.shape:
            self.unfold_metadata = None
            if old_shape[-2:] != new_array.shape[-2:]:
                self.polar_metadata = None
                self.center_beam_metadata = None
                self.rq_calibration = None

    @staticmethod
    def _normalize_flip_axes(flip_axis, ndim):
        """Validate and normalize NumPy axis indices for ``flip_axis``."""
        if flip_axis is None:
            return ()
        if isinstance(flip_axis, (Integral, np.integer)) and not isinstance(
            flip_axis, (bool, np.bool_)
        ):
            axes = (flip_axis,)
        elif isinstance(flip_axis, (list, tuple, np.ndarray)):
            axes_array = np.asarray(flip_axis)
            if axes_array.ndim == 0:
                axes = (axes_array.item(),)
            elif axes_array.ndim == 1:
                axes = tuple(axes_array)
            else:
                raise ValueError("flip_axis must be an integer or a 1D sequence.")
        else:
            raise TypeError("flip_axis must be an integer or a 1D sequence of integers.")

        normalized = []
        for axis in axes:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(
                axis, (Integral, np.integer)
            ):
                raise TypeError("flip_axis must contain only integer axis indices.")
            axis = int(axis)
            if not -ndim <= axis < ndim:
                raise ValueError(
                    f"flip_axis index {axis} is out of range for {ndim}D data."
                )
            axis %= ndim
            if axis in normalized:
                raise ValueError(f"flip_axis contains axis {axis} more than once.")
            normalized.append(axis)
        return tuple(normalized)

    def _flip_center_beam_metadata(self, flip_axes):
        """Keep current center coordinates aligned after reciprocal flips."""
        metadata = self.center_beam_metadata
        if not isinstance(metadata, dict):
            return
        flip_y = self.ndim - 2 in flip_axes
        flip_x = self.ndim - 1 in flip_axes
        for key in (
            'center_px', 'mean_fit_center_px', 'std_fit_center_px',
            'target_center_px',
        ):
            if key not in metadata:
                continue
            center = np.asarray(metadata[key], dtype=float)
            if center.shape != (2,):
                continue
            center = center.copy()
            if flip_y:
                center[0] = self.k_shape[0] - 1 - center[0]
            if flip_x:
                center[1] = self.k_shape[1] - 1 - center[1]
            metadata[key] = tuple(center)
        if 'center_calibrated' in metadata:
            center = np.asarray(metadata['center_calibrated'], dtype=float)
            if center.shape == (2,):
                center = center.copy()
                if flip_y:
                    center[0] *= -1
                if flip_x:
                    center[1] *= -1
                metadata['center_calibrated'] = tuple(center)

    @property
    def is_polar(self):
        """Return True when the last two axes represent polar ``(r, theta)``."""
        return self.polar_metadata is not None

    @staticmethod
    def _validate_scale(units, conv_factor, label):
        """Validate a units-per-pixel calibration pair."""
        if units is None or conv_factor is None:
            raise ValueError(
                f"'{label}_units' and '{label}_conv_factor' must both be provided."
            )
        if not isinstance(units, str) or not units.strip():
            raise ValueError(f"'{label}_units' must be a non-empty string.")
        if label == 'real':
            return units.strip(), _normalize_real_spacing(conv_factor)
        try:
            factor = float(conv_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{label}_conv_factor' must be a positive finite scalar.") from exc
        if not np.isscalar(conv_factor) or not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"'{label}_conv_factor' must be a positive finite scalar.")
        return units.strip(), factor

    @property
    def rq_calibration(self):
        """Optional immutable real-to-detector orientation calibration."""
        return self._rq_calibration

    @rq_calibration.setter
    def rq_calibration(self, value):
        self._rq_calibration = _coerce_rq_calibration(value)

    def set_rq_calibration(self, rotation_deg=None, mirror_axis=None, *, calibration=None):
        """Store real-to-detector orientation without changing array values.

        Parameters
        ----------
        rotation_deg : float or None, optional
            Counterclockwise real-to-detector angle in Cartesian image axes
            (x right, y up). None defaults to zero when no calibration is given.
        mirror_axis : {None, 'x', 'y'}, optional
            Reflect about this real-frame axis before the rotation. Reflection
            about x negates the y component; reflection about y negates x.
        calibration : RQCalibration or dict, optional
            Reuse a calibration or its saved dictionary instead of supplying
            an angle/reflection. For example, ``calibration=viewer.calibration``.

        Returns
        -------
        HyperData
            This object, with updated metadata. Assign ``rq_calibration=None``
            to clear it. Save/load and copy preserve the calibration.
        """
        if calibration is not None:
            if rotation_deg is not None or mirror_axis is not None:
                raise ValueError("Supply calibration or rotation_deg/mirror_axis, not both.")
            self.rq_calibration = calibration
        else:
            self.rq_calibration = RQCalibration(
                0.0 if rotation_deg is None else rotation_deg, mirror_axis,
            )
        return self

    def compare_rq(self, real_image, reciprocal_image, *, layout='overlay',
                   real_alpha=0.7, reciprocal_alpha=0.7, interactive=True,
                   scale=None, translation=(0.0, 0.0), real_show_kwargs=None,
                   reciprocal_show_kwargs=None, figsize=(12, 8), show=True,
                   export_path='rq_alignment.png'):
        """Compare a fixed real-space image with a movable diffraction image.

        This manual calibration viewer is useful with a virtual STEM image and
        a corresponding defocused shadow image. Only the two 2D inputs are
        copied. Original images and the full dataset are never resampled.
        Use an interactive Matplotlib backend (``%matplotlib widget`` with
        ipympl in Jupyter, or Qt in Spyder) for the sliders and buttons.

        Parameters
        ----------
        real_image : RealSpace or 2D ndarray
            Fixed reference. An array matching the dataset's 2D scan shape
            inherits its real-space calibration; other arrays use pixels.
        reciprocal_image : ReciprocalSpace or 2D ndarray
            Cartesian diffraction/shadow image. Its own beam metadata takes
            precedence. A plain array matching ``pattern_shape`` inherits this
            dataset's reciprocal calibration and beam metadata. The pivot is
            ``center_px``, then ``mean_fit_center_px``, then a calibrated center
            converted to pixels, or finally ``((Ky-1)/2, (Kx-1)/2)``. Stale or
            malformed center metadata raises an error rather than being used.
        layout : {'overlay', 'side_by_side'}, optional
            Overlay both images or show the fixed reference and transformed
            diffraction image on separate axes in the same reference frame.
        real_alpha, reciprocal_alpha : float, optional
            Independent image opacities in [0, 1]. GUI sliders control both.
        interactive : bool, optional
            Show rotation, opacity, scale, shift, mirror, reset, apply and
            export controls. Noninteractive viewers can still be updated
            programmatically with ``viewer.set_parameters(...)``.
        scale : positive float or None, optional
            Real horizontal pixels per diffraction pixel. None fits the image
            inside the reference field. This registration scale does not change
            either object's physical calibration. Unequal scan spacings are
            respected when rotating, even when tick labels use pixels.
        translation : (float, float), optional
            ``(dy, dx)`` real-image pixels relative to its geometric midpoint.
            The diffraction pivot is placed at that midpoint plus this shift.
        real_show_kwargs, reciprocal_show_kwargs : dict or None, optional
            Display options: ``cmap``, ``vmin``, ``vmax``, ``power``, ``logScale``,
            ``percentiles``, ``symmetric``, ``title`` and ``interpolation``.
            Diffraction display defaults to logScale=True; real display is
            linear. The real options additionally accept ``axis_units``,
            ``axes``, ``grid``, ``scale_bar``, ``scale_bar_color``, and
            ``scale_bar_label`` for the shared reference frame.
        figsize : (float, float), optional
            Figure size in inches.
        show : bool, optional
            Call Matplotlib show. Set False for scripted composition/testing.
        export_path : path-like, optional
            Initial filename in the GUI's editable export box.

        Returns
        -------
        RQComparison
            Retain this object as ``viewer``. Its ``calibration`` is the inverse
            of the displayed diffraction correction, with reflection included.
            Apply calibration stores that orientation on this dataset. Scale,
            shift and opacity remain preview settings; ``data.save(...)``
            persists the applied orientation along with the other metadata.

        Examples
        --------
        >>> viewer = data.compare_rq(virtual_image, shadow_image)
        >>> viewer.set_parameters(rotation_deg=12.5, real_alpha=0.6)
        >>> calibration = viewer.apply()
        """
        from weakref import ref

        if not isinstance(real_image, RealSpace):
            array = np.asarray(real_image)
            real_image = (
                self._spawn_real(array) if array.shape == self.real_shape
                else RealSpace(array)
            )
        if not isinstance(reciprocal_image, ReciprocalSpace):
            array = np.asarray(reciprocal_image)
            reciprocal_image = (
                self._spawn_reciprocal(array) if array.shape == self.pattern_shape
                else ReciprocalSpace(array)
            )
        owner_ref = ref(self)

        def apply_calibration(calibration):
            owner = owner_ref()
            if owner is None:
                raise RuntimeError("The source HyperData no longer exists; use viewer.calibration.")
            owner.set_rq_calibration(calibration=calibration)

        return RQComparison(
            real_image, reciprocal_image, calibration=self.rq_calibration,
            on_apply=apply_calibration, layout=layout,
            real_alpha=real_alpha, reciprocal_alpha=reciprocal_alpha,
            interactive=interactive, scale=scale, translation=translation,
            real_show_kwargs=real_show_kwargs, reciprocal_show_kwargs=reciprocal_show_kwargs,
            figsize=figsize, show=show, export_path=export_path,
        )

    def set_real_scale(self, units: str, conv_factor):
        """
        Attach scalar or ``(y, x)`` real-space units per pixel.

        Signed steps preserve physical coordinates when a scan axis is flipped.
        """
        units, conv_factor = self._validate_scale(units, conv_factor, 'real')
        self.real_units = units
        self.real_conv_factor = conv_factor
        return self

    def set_real_origin(self, origin):
        """Set the physical ``(y, x)`` coordinate of scan pixel ``(0, 0)``."""
        self.real_origin = _normalize_real_origin(origin)
        return self

    def set_reciprocal_scale(self, units: str, conv_factor: float):
        """
        Attach a reciprocal-space calibration in units per pixel.
        """
        units, conv_factor = self._validate_scale(
            units, conv_factor, 'reciprocal'
        )
        self.reciprocal_units = units
        self.reciprocal_conv_factor = conv_factor
        return self

    def clear_real_scale(self):
        """Remove the stored real-space calibration."""
        self.real_units = None
        self.real_conv_factor = None
        self.real_origin = (0.0, 0.0)
        return self

    def clear_reciprocal_scale(self):
        """Remove the stored reciprocal-space calibration."""
        self.reciprocal_units = None
        self.reciprocal_conv_factor = None
        return self

    @classmethod
    @contextmanager
    def open_hdf5(cls, filename, hdf5_dataset=None):
        """Open a 3D/4D HDF5 dataset for bounded-memory reading.

        This reader is intentionally separate from ``HyperData(path)``: it
        never pretends that every HyperData analysis method is lazy. Use
        ``get_dp`` for one pattern or ``iter_chunks`` for scan blocks. Each
        returned chunk is an independent in-memory HyperData object; unfolding
        metadata is not attached to partial chunks.

        Examples
        --------
        >>> with HyperData.open_hdf5('experiment.h5') as source:
        ...     for scan_slices, block in source.iter_chunks((16, 16)):
        ...         process(block)
        """
        path = Path(filename).expanduser()
        if path.suffix.lower() not in {'.h5', '.hdf5', '.hdf', '.4denoise'}:
            raise ValueError("open_hdf5 requires an HDF5 file path.")
        if not path.is_file():
            raise FileNotFoundError(f"'{path}' does not exist or is not a file.")

        with h5py.File(path, 'r') as file:
            file_format = _decode_hdf5_value(
                file.attrs.get('fourdenoise_format', None)
            )
            if file_format == _HYPERDATA_HDF5_FORMAT:
                version = _checked_hyperdata_format_version(file, path)
                if hdf5_dataset is not None and str(hdf5_dataset).strip('/\\') != 'array':
                    raise ValueError(
                        "Saved HyperData files contain their data at '/array'; "
                        "omit hdf5_dataset or use '/array'."
                    )
                if 'array' not in file:
                    raise ValueError(f"'{path}' does not contain a saved data array.")
                dataset = file['array']
                metadata = {}
                if 'metadata' in file:
                    group = file['metadata']
                    for key in (
                        'real_units', 'real_conv_factor', 'real_origin',
                        'reciprocal_units', 'reciprocal_conv_factor',
                        'polar_metadata', 'center_beam_metadata',
                        'rq_calibration',
                    ):
                        if key in group:
                            metadata[key] = _read_hdf5_value(group, key)
                metadata = _migrate_hyperdata_metadata(metadata, version)
            else:
                dataset = _select_hdf5_dataset(file, hdf5_dataset)
                metadata = {}

            reader = _HDF5ChunkReader(dataset, cls, metadata)
            try:
                yield reader
            finally:
                reader._closed = True

    @classmethod
    def to_polar_hdf5(cls, source, destination, *, hdf5_dataset=None,
                      chunk_shape=8, center=None, r_max=None,
                      output_shape=None, order=1, fill_value=0.0,
                      clip=False, compression='gzip', compression_opts=4,
                      overwrite=False, atomic=True, progress=True):
        """Transform HDF5 diffraction data to polar coordinates without a full output array.

        Read scan blocks through :meth:`open_hdf5`, apply the same numerical
        transform as :meth:`to_polar`, and write each result directly into a
        chunked HyperData HDF5 file. ``chunk_shape`` bounds the number of scan
        positions held in memory; it is a scalar or one value per scan axis.
        The saved result can be opened with ``HyperData.open_hdf5`` for later
        bounded-memory access or loaded fully with ``HyperData(destination)``.
        The destination must differ from the source. Atomic output is the
        default so an incomplete transform is never published.

        Parameters shared with :meth:`to_polar` have the same pixel-coordinate
        and interpolation meanings. ``hdf5_dataset`` selects an array from a
        generic HDF5 file; saved HyperData files use ``/array`` automatically.
        """
        source_path = Path(source).expanduser()
        destination_path = Path(destination).expanduser()
        if not destination_path.suffix:
            destination_path = destination_path.with_suffix('.4denoise')
        if destination_path.suffix.lower() not in {
            '.4denoise', '.h5', '.hdf5', '.hdf',
        }:
            raise ValueError(
                "destination must end in .4denoise, .h5, .hdf5, or .hdf."
            )
        if source_path.resolve() == destination_path.resolve():
            raise ValueError("source and destination must be different files.")
        if destination_path.exists() and not overwrite:
            raise FileExistsError(
                f"'{destination_path}' exists; use overwrite=True to replace it."
            )
        if not destination_path.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory '{destination_path.parent}' does not exist."
            )
        for name, value in (('overwrite', overwrite), ('atomic', atomic), ('progress', progress)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean.")
        if compression is not None and not isinstance(compression, str):
            raise ValueError("compression must be a string or None.")

        temporary_path = None
        if atomic:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f'.{destination_path.name}.', suffix='.tmp',
                dir=destination_path.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temp_name)
        write_path = temporary_path or destination_path

        try:
            with cls.open_hdf5(source_path, hdf5_dataset=hdf5_dataset) as reader:
                scan_shape = reader.shape[:-2]
                with h5py.File(write_path, 'w' if atomic or overwrite else 'x') as file:
                    file.attrs['fourdenoise_format'] = _HYPERDATA_HDF5_FORMAT
                    file.attrs['format_version'] = _HYPERDATA_HDF5_VERSION
                    file.attrs['saved_class'] = 'HyperData'
                    dataset = None
                    with tqdm(
                        total=int(np.prod(scan_shape)),
                        desc='Diffraction patterns', disable=not progress,
                    ) as bar:
                        for scan_slices, block in reader.iter_chunks(chunk_shape):
                            polar_block = block.to_polar(
                                center=center, r_max=r_max,
                                output_shape=output_shape, order=order,
                                fill_value=fill_value, clip=clip,
                                progress=False,
                            )
                            if dataset is None:
                                output_pattern_shape = polar_block.shape[-2:]
                                output_shape_full = scan_shape + output_pattern_shape
                                dataset = file.create_dataset(
                                    'array', shape=output_shape_full,
                                    dtype=polar_block.dtype,
                                    chunks=(1,) * len(scan_shape) + output_pattern_shape,
                                    **_hdf5_dataset_kwargs(
                                        polar_block.array, compression,
                                        compression_opts,
                                    ),
                                )
                                file.attrs['array_ndim'] = polar_block.ndim
                                file.attrs['array_dtype'] = str(polar_block.dtype)
                                metadata = {
                                    'real_units': polar_block.real_units,
                                    'real_conv_factor': polar_block.real_conv_factor,
                                    'real_origin': polar_block.real_origin,
                                    'reciprocal_units': polar_block.reciprocal_units,
                                    'reciprocal_conv_factor': polar_block.reciprocal_conv_factor,
                                    'polar_metadata': polar_block.polar_metadata,
                                    'unfold_metadata': None,
                                    'center_beam_metadata': None,
                                    'rq_calibration': (
                                        polar_block.rq_calibration.to_dict()
                                        if polar_block.rq_calibration is not None else None
                                    ),
                                }
                                metadata_group = file.create_group('metadata')
                                metadata_group.attrs['kind'] = 'dict'
                                for key, value in metadata.items():
                                    _write_hdf5_value(
                                        metadata_group, key, value,
                                        compression=compression,
                                        compression_opts=compression_opts,
                                    )
                            dataset[scan_slices + (slice(None), slice(None))] = polar_block.array
                            bar.update(int(np.prod(polar_block.scan_shape)))
                    if dataset is None:
                        raise ValueError("source has no diffraction patterns to transform.")

            if atomic:
                if overwrite:
                    os.replace(temporary_path, destination_path)
                else:
                    os.link(temporary_path, destination_path)
            return str(destination_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def save(self, filename, overwrite=False, compression='gzip',
             compression_opts=4, atomic=True):
        """
        Save this HyperData object with its metadata.

        The saved file can be loaded directly with ``HyperData(filename)``.
        The recommended extension is ``.4denoise``; ``.h5`` and ``.hdf5`` are
        also accepted.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output filename. If no extension is supplied, ``.4denoise`` is
            appended.
        overwrite : bool, optional
            If False, raise an error when the output file already exists.
        compression : str or None, optional
            HDF5 compression filter for array-like datasets. Default is
            ``'gzip'``. Use ``None`` to save without compression.
        compression_opts : int or None, optional
            Compression level/options passed to HDF5. For gzip, values usually
            range from 0 to 9.
        atomic : bool, optional
            Write to a temporary file in the destination directory and publish
            it only after a complete save. This protects an existing file from
            partial writes, but temporarily requires space for both copies
            when overwriting. Set False to write directly as before.

        Returns
        -------
        str
            The saved file path.

        Examples
        --------
        >>> data.save('experiment.4denoise')
        >>> loaded = HyperData('experiment.4denoise')
        """
        path = Path(filename).expanduser()
        if path.suffix == '':
            path = path.with_suffix('.4denoise')

        valid_suffixes = {'.4denoise', '.h5', '.hdf5'}
        if path.suffix.lower() not in valid_suffixes:
            raise ValueError(
                "HyperData.save supports '.4denoise', '.h5', and '.hdf5' "
                f"files; got '{path.suffix}'."
            )
        if not isinstance(overwrite, (bool, np.bool_)):
            raise ValueError("overwrite must be a boolean.")
        if not isinstance(atomic, (bool, np.bool_)):
            raise ValueError("atomic must be a boolean.")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"'{path}' already exists. Use overwrite=True to replace it."
            )
        if path.parent and not path.parent.exists():
            raise FileNotFoundError(
                f"Output directory '{path.parent}' does not exist."
            )
        if compression is not None and not isinstance(compression, str):
            raise ValueError("compression must be a string or None.")

        metadata = {
            'real_units': self.real_units,
            'real_conv_factor': self.real_conv_factor,
            'real_origin': self.real_origin,
            'reciprocal_units': self.reciprocal_units,
            'reciprocal_conv_factor': self.reciprocal_conv_factor,
            'polar_metadata': self.polar_metadata,
            'unfold_metadata': self.unfold_metadata,
            'center_beam_metadata': self.center_beam_metadata,
            'rq_calibration': (
                self.rq_calibration.to_dict() if self.rq_calibration is not None else None
            ),
        }

        def _write_file(target):
            with h5py.File(target, 'w') as file:
                file.attrs['fourdenoise_format'] = _HYPERDATA_HDF5_FORMAT
                file.attrs['format_version'] = _HYPERDATA_HDF5_VERSION
                file.attrs['saved_class'] = 'HyperData'
                file.attrs['array_ndim'] = self.ndim
                file.attrs['array_dtype'] = str(self.dtype)

                file.create_dataset(
                    'array',
                    data=self.array,
                    **_hdf5_dataset_kwargs(
                        self.array,
                        compression,
                        compression_opts,
                    ),
                )
                metadata_group = file.create_group('metadata')
                metadata_group.attrs['kind'] = 'dict'
                for key, value in metadata.items():
                    _write_hdf5_value(
                        metadata_group,
                        key,
                        value,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

        if not atomic:
            _write_file(path)
            return str(path)

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent,
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            _write_file(temp_path)
            if overwrite:
                os.replace(temp_path, path)
            else:
                # Linking publishes the finished file without a race that
                # could overwrite a destination created by another process.
                os.link(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

        return str(path)

    def _spawn(self, data,
               real_units=_SCALE_UNSET,
               real_conv_factor=_SCALE_UNSET,
               reciprocal_units=_SCALE_UNSET,
               reciprocal_conv_factor=_SCALE_UNSET,
               polar_metadata=_SCALE_UNSET,
               center_beam_metadata=_SCALE_UNSET,
               real_origin=_SCALE_UNSET,
               rq_calibration=_SCALE_UNSET,
               preserve_unfold=False):
        """Create an object; retain unfolding only for shape-preserving values."""
        if real_units is _SCALE_UNSET:
            real_units = self.real_units
        if real_conv_factor is _SCALE_UNSET:
            real_conv_factor = self.real_conv_factor
        if reciprocal_units is _SCALE_UNSET:
            reciprocal_units = self.reciprocal_units
        if reciprocal_conv_factor is _SCALE_UNSET:
            reciprocal_conv_factor = self.reciprocal_conv_factor
        if polar_metadata is _SCALE_UNSET:
            polar_metadata = self.polar_metadata
        if center_beam_metadata is _SCALE_UNSET:
            center_beam_metadata = (
                self.center_beam_metadata
                if tuple(data.shape[-2:]) == self.k_shape else None
            )
        if real_origin is _SCALE_UNSET:
            real_origin = self.real_origin
        if rq_calibration is _SCALE_UNSET:
            rq_calibration = self.rq_calibration

        result = HyperData(
            data,
            real_units=real_units,
            real_conv_factor=real_conv_factor,
            reciprocal_units=reciprocal_units,
            reciprocal_conv_factor=reciprocal_conv_factor,
            polar_metadata=deepcopy(polar_metadata) if polar_metadata is not None else None,
            center_beam_metadata=(
                deepcopy(center_beam_metadata)
                if center_beam_metadata is not None
                else None
            ),
            real_origin=real_origin,
            rq_calibration=rq_calibration,
        )
        if preserve_unfold and result.shape == self.shape:
            result.unfold_metadata = _clone_unfold_metadata(
                self.unfold_metadata, share_payload=True,
            )
        return result

    def _spawn_reciprocal(self, data, units=_SCALE_UNSET,
                          conv_factor=_SCALE_UNSET,
                          polar_metadata=_SCALE_UNSET,
                          center_beam_metadata=_SCALE_UNSET):
        """Create a ReciprocalSpace object using this dataset's calibration."""
        if units is _SCALE_UNSET:
            units = self.reciprocal_units
        if conv_factor is _SCALE_UNSET:
            conv_factor = self.reciprocal_conv_factor
        if polar_metadata is _SCALE_UNSET:
            polar_metadata = self.polar_metadata
        if center_beam_metadata is _SCALE_UNSET:
            center_beam_metadata = self.center_beam_metadata
        return ReciprocalSpace(
            data,
            units=units,
            conv_factor=conv_factor,
            polar_metadata=deepcopy(polar_metadata) if polar_metadata is not None else None,
            center_beam_metadata=(
                deepcopy(center_beam_metadata)
                if center_beam_metadata is not None
                else None
            ),
        )

    def _spawn_real(self, data, units=_SCALE_UNSET, conv_factor=_SCALE_UNSET,
                    origin=_SCALE_UNSET, quantity='Intensity', value_units=None):
        """Create a RealSpace object using this dataset's calibration."""
        if units is _SCALE_UNSET:
            units = self.real_units
        if conv_factor is _SCALE_UNSET:
            conv_factor = self.real_conv_factor
        if origin is _SCALE_UNSET:
            origin = self.real_origin
        return RealSpace(
            data, units=units, conv_factor=conv_factor, origin=origin,
            quantity=quantity, value_units=value_units,
        )

    def copy(self):
        """
        Return an independent copy of this HyperData object.

        The numerical array is copied, and metadata such as calibration,
        polar-transform metadata, and unfolding metadata are copied so changes
        to the returned object do not mutate this object.
        """
        copied = self._spawn(np.array(self.array, copy=True))
        copied.unfold_metadata = _clone_unfold_metadata(self.unfold_metadata)
        return copied

    def _resolve_real_selection_units(self, selection_units):
        """Resolve how real-space selections passed to get_dp are interpreted."""
        return _resolve_unit_mode(
            selection_units,
            self.real_units,
            self.real_conv_factor,
            label='selection_units',
        )

    def _resolve_reciprocal_detector_units(self, detector_units):
        """Resolve how Cartesian reciprocal detector inputs are interpreted."""
        return _resolve_unit_mode(
            detector_units,
            self.reciprocal_units,
            self.reciprocal_conv_factor,
            label='detector_units',
        )

    def _resolve_polar_radius_units(self, detector_units):
        """
        Resolve units for radial detector inputs on polar data.

        The radial axis stores both original Cartesian-pixel radii and, when
        available, calibrated reciprocal radii. ``auto`` follows the calibrated
        path only if the source Cartesian calibration exists.
        """
        metadata = self.polar_metadata or {}
        mode = _normalize_unit_mode(detector_units, label='detector_units')
        units = metadata.get('cartesian_reciprocal_units')
        conv_factor = metadata.get('cartesian_reciprocal_conv_factor')

        if mode == 'pixels':
            return None, None, 'pixels'
        if units is None or conv_factor is None:
            if mode == 'calibrated':
                raise ValueError(
                    "detector_units='calibrated' requires polar metadata with "
                    "the original reciprocal-space calibration."
                )
            return None, None, 'pixels'
        if not np.isscalar(conv_factor) or conv_factor <= 0:
            raise ValueError(
                "polar_metadata['cartesian_reciprocal_conv_factor'] must be "
                "a positive scalar."
            )
        return str(units).strip(), float(conv_factor), 'calibrated'

    @staticmethod
    def _scale_values_to_pixels(values, conv_factor):
        """Convert calibrated distances to pixel distances."""
        return np.asarray(values, dtype=float) / float(conv_factor)

    @staticmethod
    def _calibrated_centers_to_pixels(centers, conv_factor, shape):
        """
        Convert calibrated ``(ky, kx)`` coordinates to pixel coordinates.

        Calibrated reciprocal coordinates are centered at the diffraction
        origin, with positive ``kx`` to the right and positive ``ky`` upward.
        """
        centers = np.asarray(centers, dtype=float)
        center_y = (shape[0] - 1) / 2.0
        center_x = (shape[1] - 1) / 2.0
        return np.column_stack((
            center_y - centers[:, 0] / float(conv_factor),
            center_x + centers[:, 1] / float(conv_factor),
        ))

    @property
    def available_denoising_methods(self):
        """Return denoising method names available through :meth:`denoise`."""
        return tuple(self._denoise_engine.available_methods)


    def denoising_method_info(self, method=None, include_doc=True, print_info=True):
        """
        Show the method-specific inputs accepted by a denoising method.

        Parameters
        ----------
        method : str or None, optional
            Name of the denoising method. If None, return the available method
            names and a contract summary for each one.
        include_doc : bool, optional
            If True, include the selected method's docstring in the returned
            dictionary.
        print_info : bool, optional
            If True, print a compact, notebook-friendly summary.

        Returns
        -------
        dict
            Signature, input dimensionality, 4D routing options, and output
            contract. The data array input is supplied by :meth:`denoise`, so
            it is reported separately from user-provided keyword arguments.
        """
        return self._denoise_engine.method_info(
            method_name=method,
            include_doc=include_doc,
            print_info=print_info,
        )


    def denoise_info(self, method=None, include_doc=True, print_info=True):
        """Alias for :meth:`denoising_method_info`."""
        return self.denoising_method_info(
            method=method,
            include_doc=include_doc,
            print_info=print_info,
        )

    def _finalize_denoise_result(self, result, method, return_array=False,
                                 allow_auxiliary=False, expected_shape=None,
                                 spawn_kwargs=None):
        """
        Validate and wrap a denoised reconstruction using this object's metadata.

        Opt-in decomposition payloads retain their existing method-specific
        return types. Every default reconstruction must preserve its expected
        shape before it can inherit dataset metadata.
        """
        if allow_auxiliary and isinstance(result, (tuple, list)):
            return result

        result = _DenoiseEngine.require_reconstruction(
            result, expected_shape or self.shape, method=method,
        )
        if return_array:
            return result

        return self._spawn(
            result, preserve_unfold=True, **(spawn_kwargs or {}),
        )

    @staticmethod
    def _normalize_scalar_rank(rank):
        """Return a positive numeric rank from a scalar."""
        if isinstance(rank, np.generic):
            rank = rank.item()

        if isinstance(rank, Integral):
            rank_value = int(rank)
        elif np.isscalar(rank):
            try:
                rank_value = float(rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Rank values must be positive numeric values; got {rank!r}."
                ) from exc
            if rank_value.is_integer():
                rank_value = int(rank_value)
        else:
            raise ValueError(
                "Each rank entry must be numeric or a 1D numeric sequence."
            )

        if not np.isfinite(rank_value) or rank_value <= 0:
            raise ValueError(
                f"Rank values must be positive finite numbers; got {rank_value}."
            )
        return rank_value

    @classmethod
    def _normalize_rank_value(cls, rank):
        """Normalize one scalar rank or one tuple/list/array rank value."""
        if isinstance(rank, np.ndarray):
            if rank.ndim == 0:
                return cls._normalize_scalar_rank(rank.item())
            if rank.ndim == 1:
                return tuple(cls._normalize_scalar_rank(value) for value in rank)
            raise ValueError(
                "Each rank entry must be scalar or a 1D numeric sequence."
            )

        if isinstance(rank, (list, tuple)):
            if len(rank) == 0:
                raise ValueError("Rank sequences cannot be empty.")
            return tuple(cls._normalize_scalar_rank(value) for value in rank)

        return cls._normalize_scalar_rank(rank)

    @classmethod
    def _normalize_rank_sweep(cls, ranks):
        """Normalize an arbitrary rank iterable into rank values."""
        if ranks is None:
            raise ValueError("ranks must be a non-empty iterable of ranks.")

        if isinstance(ranks, np.ndarray):
            if ranks.ndim == 0:
                raw_ranks = [ranks.item()]
            elif ranks.ndim == 1:
                raw_ranks = ranks.tolist()
            else:
                raw_ranks = [row for row in ranks]
        elif np.isscalar(ranks):
            raw_ranks = [ranks]
        else:
            try:
                raw_ranks = list(ranks)
            except TypeError as exc:
                raise ValueError(
                    "ranks must be a scalar rank or an iterable of ranks."
                ) from exc

        if len(raw_ranks) == 0:
            raise ValueError("ranks must contain at least one rank value.")

        return tuple(cls._normalize_rank_value(rank) for rank in raw_ranks)

    @staticmethod
    def _rank_label(rank):
        """Return a compact plot/table label for one rank value."""
        if isinstance(rank, tuple):
            return "(" + ", ".join(str(value) for value in rank) + ")"
        return str(rank)

    @staticmethod
    def _rank_scree_y_values(metric, relative_errors, residual_norms, fits):
        """Return plotted y-values and axis label for a rank scree metric."""
        metric = metric.lower()
        if metric in ('relative_error', 'error', 'rel_error'):
            return relative_errors, 'Relative reconstruction error'
        if metric in ('residual_norm', 'residual'):
            return residual_norms, 'Residual norm'
        if metric == 'fit':
            return fits, 'Fit = 1 - relative error'
        raise ValueError(
            "metric must be 'relative_error', 'residual_norm', or 'fit'."
        )

    @staticmethod
    def _rank_scree_block_norm(array, other=None, block_elements=1_000_000):
        """Compute an array or residual norm without a full-size temporary."""
        operands = (np.asarray(array),) if other is None else (
            np.asarray(array), np.asarray(other),
        )
        working_dtype = np.result_type(
            *(operand.dtype for operand in operands), np.float64,
        )
        chunks = np.nditer(
            operands,
            flags=['external_loop', 'buffered', 'zerosize_ok'],
            op_flags=[['readonly']] * len(operands),
            order='C',
            buffersize=block_elements,
        )
        norm = 0.0
        for chunk in chunks:
            if other is None:
                values = np.asarray(chunk, dtype=working_dtype)
            else:
                values = np.subtract(
                    chunk[0], chunk[1], dtype=working_dtype,
                )
            norm = hypot(norm, float(np.linalg.norm(values)))
        return norm

    @staticmethod
    def _apply_unfolded_denoiser(engine, method, expected_shape, kwargs):
        """Apply one method to an unfolding and retain requested errors."""
        result = engine.denoise(method, **kwargs)
        return_errors = bool(kwargs.get('return_errors', False))
        if return_errors:
            result, errors = engine.split_reconstruction_errors(result, method)
        reconstruction = engine.require_reconstruction(
            result, expected_shape, method=method,
            context='automatic unfold-denoise-refold',
        )
        return (reconstruction, errors) if return_errors else reconstruction

    def _denoised_unfold_spawn_kwargs(self, metadata, unfolded, restored_shape):
        """Keep calibration coherent when undo returns cropped/resized data."""
        if tuple(restored_shape) == self.shape:
            return {}

        strategy = metadata.get('curve_shape_strategy')
        options = {}
        if strategy == 'resize':
            options.update(
                real_units=unfolded.real_units,
                real_conv_factor=unfolded.real_conv_factor,
                real_origin=unfolded.real_origin,
                reciprocal_units=unfolded.reciprocal_units,
                reciprocal_conv_factor=unfolded.reciprocal_conv_factor,
            )
        elif strategy == 'center_crop' and metadata['domain'] in ('real', 'both'):
            crop = (
                metadata['real_crop_slices']
                if metadata['domain'] == 'both'
                else metadata['crop_slices']
            )
            real_step = _real_spacing_pair(self.real_conv_factor)
            options['real_origin'] = (
                self.real_origin[0] + crop['y'][0] * real_step[0],
                self.real_origin[1] + crop['x'][0] * real_step[1],
            )

        if tuple(restored_shape[-2:]) != self.k_shape:
            options['polar_metadata'] = None
            options['center_beam_metadata'] = None
        return options

    @staticmethod
    def _expected_denoised_refold_shape(metadata):
        """Return the documented undo geometry for one unfolding."""
        strategy = metadata.get('curve_shape_strategy')
        if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
            return _crop_restore_shape(metadata)
        if strategy == 'resize' and not metadata.get('preserve_original', False):
            return tuple(metadata['working_shape'])
        return tuple(metadata['original_shape'])

    @staticmethod
    def _validate_processing_unfold_options(method, options):
        """Reject undo options that would discard a processed resize result."""
        traversal = _normalize_traversal_method(method)
        strategy = options.get('curve_shape_strategy', 'center_crop')
        if (
            traversal in _CURVE_TRAVERSAL_METHODS
            and isinstance(strategy, str)
            and strategy.lower() == 'resize'
            and options.get('preserve_original', False)
        ):
            raise ValueError(
                "unfold_kwargs cannot combine curve_shape_strategy='resize' "
                "with preserve_original=True during denoising or rank_scree: "
                "undo would return the saved original and discard the "
                "processed tensor. Use preserve_original=False to obtain "
                "the denoised resized tensor."
            )

    def rank_scree(self, method, ranks, domain='reciprocal',
                   unfold_domain=None, unfold_method='row_major',
                   unfold_kwargs=None, metric='relative_error',
                   plot=True, ax=None, log_y=False, show=True,
                   progress=True, return_reconstructions=False,
                   error_chunk_elements=1_000_000, **kwargs):
        """
        Run a rank sweep and plot final reconstruction quality for each rank.

        This method performs one complete denoising/decomposition run per rank.
        It is therefore much more expensive than a convergence plot from one
        denoise call. ``ranks`` may be a ``range``, tuple, list, NumPy array, or
        any iterable of positive numeric rank values. Nested rank values, such as
        ``[(2, 2, 2), (4, 4, 4)]`` for Tucker-style rank specifications, are
        also accepted. When unfolding is requested, the input is unfolded once
        for the whole sweep. For reversible traversals, errors are measured in
        unfolded order without constructing full refolded arrays unless
        ``return_reconstructions=True``. For a cropped or resized traversal,
        fit quality is measured on the tensor actually denoised, not on
        untouched excess pixels or the original pre-resize tensor.

        Parameters
        ----------
        method : str
            Name of a denoising method that accepts a ``rank`` argument.
        ranks : iterable
            Rank values to evaluate. Examples: ``range(1, 31)``,
            ``[1, 2, 4, 8]``, ``np.array([5, 10, 20])``.
        domain : {'real', 'reciprocal'} or None, optional
            Passed to :meth:`denoise`.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Passed to :meth:`denoise`.
        unfold_method : str, optional
            Passed to :meth:`denoise`.
        unfold_kwargs : dict or None, optional
            Additional unfolding arguments passed to :meth:`denoise`.
            Resize traversals must use ``preserve_original=False`` so undo
            returns the processed resized tensor rather than the saved input.
        metric : {'relative_error', 'residual_norm', 'fit'}, optional
            Quantity to plot on the y-axis.
        plot : bool, optional
            If True, create a Matplotlib scree/elbow plot.
        ax : matplotlib.axes.Axes or None, optional
            Existing axes for plotting. If None, a new figure is created.
        log_y : bool, optional
            If True, use a logarithmic y-axis.
        show : bool, optional
            If True, call ``plt.show()`` after plotting.
        progress : bool, optional
            If True, show a progress bar over ranks.
        return_reconstructions : bool, optional
            If True, include each reconstructed array in the returned results.
            This intentionally retains one full array per rank in memory.
        error_chunk_elements : int, optional
            Maximum elements per block when computing reconstruction norms.
            Defaults to one million; lower this to reduce temporary memory.
        **kwargs
            Keyword arguments passed to :meth:`denoise` for every rank.

        Returns
        -------
        dict
            Dictionary containing ranks, residual norms, relative errors, fit,
            and plotting objects.
        """
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string.")

        reserved = {
            'rank',
            'return_array',
            'return_decomposition',
            'return_errors',
        }
        conflicts = reserved.intersection(kwargs)
        if conflicts:
            conflict_text = ', '.join(sorted(conflicts))
            raise ValueError(
                f"Do not pass {conflict_text} to rank_scree. "
                "rank_scree controls these internally."
            )

        method_obj = getattr(self._denoise_engine.methods, method, None)
        if method_obj is None or method.startswith('_'):
            raise ValueError(
                f"No such denoising method '{method}'. Available methods are: "
                f"{', '.join(self.available_denoising_methods)}"
            )
        if 'rank' not in inspect.signature(method_obj).parameters:
            raise ValueError(
                f"Method '{method}' does not expose a rank parameter."
            )
        if (
            isinstance(error_chunk_elements, (bool, np.bool_))
            or not isinstance(error_chunk_elements, (Integral, np.integer))
            or error_chunk_elements <= 0
        ):
            raise ValueError("error_chunk_elements must be a positive integer.")
        error_chunk_elements = int(error_chunk_elements)

        rank_values = self._normalize_rank_sweep(ranks)
        rank_labels = tuple(self._rank_label(rank) for rank in rank_values)
        original = np.asarray(self.array)

        unfolded = None
        unfold_metadata = None
        unfolded_engine = None
        if unfold_domain is not None:
            if self.ndim != 4:
                raise ValueError(
                    "unfold_domain can only be used with 4D HyperData. "
                    "For 2D or 3D data, denoise applies the method directly."
                )
            if unfold_kwargs is None:
                unfold_kwargs = {}
            if not isinstance(unfold_kwargs, dict):
                raise ValueError("unfold_kwargs must be a dictionary or None.")
            self._validate_processing_unfold_options(
                unfold_method, unfold_kwargs,
            )
            unfolded, unfold_metadata = self.unfold(
                domain=unfold_domain,
                method=unfold_method,
                return_metadata=True,
                **unfold_kwargs,
            )
            unfolded_engine = _DenoiseEngine(unfolded.array)

        comparison_input = original if unfolded is None else unfolded.array
        comparison_norm = self._rank_scree_block_norm(
            comparison_input, block_elements=error_chunk_elements,
        )
        if comparison_norm == 0:
            comparison_norm = np.nan

        iterator = rank_values
        if progress and len(rank_values) > 1:
            iterator = tqdm(rank_values, desc=f"{method} rank sweep")

        residual_norms = []
        relative_errors = []
        fits = []
        reconstructions = []

        for rank in iterator:
            reconstruction = None
            if unfolded is None:
                reconstruction = self.denoise(
                    method=method,
                    rank=rank,
                    domain=domain,
                    return_array=True,
                    **kwargs,
                )
                reconstruction = np.asarray(reconstruction)
                if reconstruction.shape != original.shape:
                    raise ValueError(
                        f"Rank {rank!r} returned shape {reconstruction.shape}, "
                        f"but expected {original.shape}. rank_scree requires "
                        "shape-preserving denoising."
                    )
                residual_norm = self._rank_scree_block_norm(
                    original, reconstruction,
                    block_elements=error_chunk_elements,
                )
            else:
                denoised_unfolded = self._apply_unfolded_denoiser(
                    unfolded_engine,
                    method,
                    unfolded.array.shape,
                    dict(kwargs, rank=rank),
                )
                residual_norm = self._rank_scree_block_norm(
                    unfolded.array,
                    denoised_unfolded,
                    block_elements=error_chunk_elements,
                )
                if return_reconstructions:
                    reconstruction = _unfold_array(
                        denoised_unfolded,
                        undo=True,
                        metadata=unfold_metadata,
                    )
            relative_error = residual_norm / comparison_norm
            fit = 1 - relative_error

            residual_norms.append(residual_norm)
            relative_errors.append(relative_error)
            fits.append(fit)
            if return_reconstructions:
                reconstructions.append(reconstruction)
            del reconstruction

        residual_norms = np.asarray(residual_norms, dtype=float)
        relative_errors = np.asarray(relative_errors, dtype=float)
        fits = np.asarray(fits, dtype=float)
        y_values, y_label = self._rank_scree_y_values(
            metric,
            relative_errors,
            residual_norms,
            fits,
        )

        if all(np.isscalar(rank) for rank in rank_values):
            x_values = np.asarray(rank_values, dtype=float)
            if np.all(np.equal(x_values, np.round(x_values))):
                x_values = x_values.astype(int)
            x_label = 'Rank'
            use_rank_tick_labels = False
        else:
            x_values = np.arange(1, len(rank_values) + 1)
            x_label = 'Rank configuration'
            use_rank_tick_labels = True

        figure = None
        if plot:
            if ax is None:
                figure, ax = plt.subplots(figsize=(7, 4))
            else:
                figure = ax.figure

            ax.plot(x_values, y_values, '-o')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{method} rank scree")
            ax.grid(True, alpha=0.3)
            if log_y:
                ax.set_yscale('log')
            if use_rank_tick_labels:
                ax.set_xticks(x_values)
                ax.set_xticklabels(rank_labels, rotation=45, ha='right')
            if show:
                plt.show()

        results = {
            'method': method,
            'ranks': rank_values,
            'rank_labels': rank_labels,
            'residual_norm': residual_norms,
            'relative_error': relative_errors,
            'errors': relative_errors,
            'fit': fits,
            'metric': metric,
            'comparison_shape': tuple(comparison_input.shape),
            'x': x_values,
            'y': y_values,
            'figure': figure,
            'ax': ax,
        }
        if return_reconstructions:
            results['reconstructions'] = tuple(reconstructions)

        return results


    @staticmethod
    def _plot_denoise_convergence(errors, method, ax=None, show=True):
        """Plot the history reported by one denoising run."""
        values = np.asarray(errors, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Method '{method}' must return a nonempty, finite 1D error "
                "history for convergence plotting."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        # Tensor-Ring callbacks include the initial, pre-iteration error.
        first_iteration = 0 if method.startswith('tensor_ring_als') else 1
        iterations = np.arange(first_iteration, first_iteration + values.size)
        ax.plot(iterations, values, '-o', markersize=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reported error')
        ax.set_title(f'{method} convergence')
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    def denoise(self, method, domain='reciprocal', unfold_domain=None,
                unfold_method='row_major', unfold_kwargs=None,
                return_array=False, convergence_plot=False,
                convergence_ax=None, convergence_show=True, **kwargs):
        """
        Denoise this dataset and return a new :class:`HyperData` object.

        Parameters
        ----------
        method : str
            Name of a method available in ``available_denoising_methods``.
        domain : {'real', 'reciprocal'} or None, optional
            Coordinate domain used for denoising 4D data. ``'real'`` selects
            axes 0 and 1, while ``'reciprocal'`` selects axes 2 and 3.
            Axis-aware methods such as ``median`` operate on those axes in one
            whole-array call; methods that require individual 2D images remain
            slice-wise. Use ``domain=None`` to apply a method directly to the
            whole array.
            For 2D and 3D data, methods are applied directly to the whole
            array regardless of domain.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Domain to unfold before denoising. If provided for 4D data, the
            data are unfolded, denoised as a whole lower-dimensional array, and
            then refolded automatically.
        unfold_method : str, optional
            Traversal method passed to :meth:`unfold` during generic
            unfold-denoise-refold routing. A 3D volume filter treats the
            traversal index as one spatial axis, which cannot preserve every
            neighbor relationship of the original 2D coordinate grid.
        unfold_kwargs : dict or None, optional
            Additional keyword arguments passed to :meth:`unfold`, such as
            ``curve_shape_strategy`` or ``preserve_excess``.
            A non-preserving crop or resize may return a smaller or resized
            4D dataset; its calibration is updated for that geometry.
            ``preserve_original=True`` is not supported with resize here,
            because undo would discard the denoising result.
        return_array : bool, optional
            If True, return the denoised ``ndarray`` instead of wrapping it in a
            new ``HyperData`` object.
        convergence_plot : bool, optional
            Plot the method's reported per-iteration errors from this same
            denoising run. Requires a method with ``return_errors`` support.
            Plotting does not change the default reconstruction-only return.
        convergence_ax : matplotlib.axes.Axes or None, optional
            Existing axes for the convergence plot. Only used when
            ``convergence_plot=True``.
        convergence_show : bool, optional
            Call ``plt.show()`` for the convergence plot. Set False to compose
            or save the figure yourself through ``convergence_ax``.
        **kwargs
            Keyword arguments passed to the selected denoising method.
            ``return_decomposition=True`` retains the method's existing
            auxiliary return type for direct denoising. It is not supported
            with slice-wise 4D routing or automatic unfolding.
            For methods supporting ``return_errors=True``, return
            ``(reconstruction, convergence_errors)``. Use ``domain=None``
            or ``unfold_domain`` for 4D data; separate 2D slices do not have
            one shared convergence history.
            The plotted history is the selected method's reported error,
            not a separately calculated residual; its definition may differ
            between methods. Tensor-Ring histories include iteration zero.

        Returns
        -------
        HyperData or ndarray, or tuple
            Denoised data. By default this is a new ``HyperData`` object that
            preserves this object's calibration metadata. With
            ``return_errors=True``, the first tuple element is the same
            reconstruction and the second is TensorLy's error sequence.

        Examples
        --------
        >>> denoised = my_dataset.denoise(
        ...     method='median',
        ...     domain='reciprocal',
        ...     window_size=3,
        ... )
        >>> denoised = my_dataset.denoise(
        ...     method='some_3d_method',
        ...     unfold_domain='real',
        ...     unfold_method='meander-4',
        ... )
        >>> denoised = my_dataset.denoise(
        ...     method='some_whole_array_method',
        ...     domain=None,
        ... )
        >>> denoised, errors = my_dataset.denoise(
        ...     method='parafac', rank=5, return_errors=True,
        ... )
        >>> denoised = my_dataset.denoise(
        ...     method='parafac', rank=5, convergence_plot=True,
        ... )
        """
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string.")

        requested_errors = bool(kwargs.get('return_errors', False)) and not bool(
            kwargs.get('return_decomposition', False)
        )
        if convergence_ax is not None and not convergence_plot:
            raise ValueError("convergence_ax requires convergence_plot=True.")
        if convergence_plot:
            if kwargs.get('return_decomposition', False):
                raise ValueError(
                    "convergence_plot requires return_decomposition=False."
                )
            numerical_method = _DenoiseEngine(self.array)._resolve_method(method)
            if 'return_errors' not in inspect.signature(numerical_method).parameters:
                raise ValueError(
                    f"Method '{method}' does not report per-iteration errors; "
                    "convergence_plot is unavailable."
                )
            kwargs = {**kwargs, 'return_errors': True}

        if unfold_domain is not None:
            if self.ndim != 4:
                raise ValueError(
                    "unfold_domain can only be used with 4D HyperData. "
                    "For 2D or 3D data, denoise applies the method directly."
                )
            if unfold_kwargs is None:
                unfold_kwargs = {}
            if not isinstance(unfold_kwargs, dict):
                raise ValueError("unfold_kwargs must be a dictionary or None.")
            self._validate_processing_unfold_options(
                unfold_method, unfold_kwargs,
            )
            if kwargs.get('return_decomposition', False):
                raise ValueError(
                    "Automatic unfold-denoise-refold requires a single "
                    "reconstruction; use return_decomposition=False."
                )

            unfolded, unfold_metadata = self.unfold(
                domain=unfold_domain,
                method=unfold_method,
                return_metadata=True,
                **unfold_kwargs,
            )

            if kwargs.get('mask') is not None:
                mask = np.asarray(kwargs['mask'])
                if mask.shape == self.shape:
                    mask_options = dict(unfold_kwargs)
                    mask_options['preserve_excess'] = False
                    mask_options['preserve_original'] = False
                    mask_options['plot_traversal'] = False
                    mask_options['warn_on_crop'] = False
                    if mask_options.get('curve_shape_strategy') == 'resize':
                        mask_options['resize_method'] = 'nearest'
                        if mask.dtype == np.dtype('bool'):
                            mask = mask.astype(np.uint8)
                    mask = HyperData(mask).unfold(
                        domain=unfold_domain, method=unfold_method,
                        **mask_options,
                    ).array
                elif mask.shape != unfolded.shape:
                    raise ValueError(
                        "mask must match the original 4D data or the "
                        f"unfolded shape {unfolded.shape}; got {mask.shape}."
                    )
                kwargs = {**kwargs, 'mask': mask}

            denoised_unfolded = self._apply_unfolded_denoiser(
                _DenoiseEngine(unfolded.array),
                method,
                unfolded.array.shape,
                kwargs,
            )
            return_errors = bool(kwargs.get('return_errors', False))
            if return_errors:
                denoised_unfolded, errors = denoised_unfolded
            refolded = _unfold_array(
                denoised_unfolded,
                undo=True,
                metadata=unfold_metadata,
            )
            result = self._finalize_denoise_result(
                refolded,
                method=method,
                return_array=return_array,
                expected_shape=self._expected_denoised_refold_shape(
                    unfold_metadata,
                ),
                spawn_kwargs=self._denoised_unfold_spawn_kwargs(
                    unfold_metadata, unfolded, refolded.shape,
                ),
            )
            if convergence_plot:
                self._plot_denoise_convergence(
                    errors, method, ax=convergence_ax, show=convergence_show,
                )
            return (result, errors) if requested_errors else result

        engine = _DenoiseEngine(self.array)
        denoised = engine.apply(
            method=method,
            domain=domain,
            **kwargs,
        )
        return_errors = bool(kwargs.get('return_errors', False)) and not bool(
            kwargs.get('return_decomposition', False)
        )
        if return_errors:
            denoised, errors = engine.split_reconstruction_errors(denoised, method)
        result = self._finalize_denoise_result(
            denoised,
            method=method,
            return_array=return_array,
            allow_auxiliary=bool(kwargs.get('return_decomposition', False)),
        )
        if convergence_plot:
            self._plot_denoise_convergence(
                errors, method, ax=convergence_ax, show=convergence_show,
            )
        return (result, errors) if requested_errors else result


    @staticmethod
    def _center_crop_notice(metadata):
        """Describe coordinates excluded from a center-crop unfolding."""
        if metadata.get('curve_shape_strategy') != 'center_crop':
            return None

        domain = metadata['domain']
        domains = ('real', 'reciprocal') if domain == 'both' else (domain,)
        coverage = []
        has_excess = False
        for selected_domain in domains:
            prefix = f'{selected_domain}_' if domain == 'both' else ''
            grid = metadata[f'{prefix}traversal_shape']
            crop = metadata[f'{prefix}crop_slices']
            included = len(metadata[f'{prefix}kept_indices'])
            excluded = len(metadata[f'{prefix}excess_indices'])
            has_excess |= excluded > 0
            coverage.append(
                f"{selected_domain}-space: {included}/{grid[0] * grid[1]} "
                f"coordinates included (centered y[{crop['y'][0]}:{crop['y'][1]}], "
                f"x[{crop['x'][0]}:{crop['x'][1]}]); {excluded} excluded"
            )

        if not has_excess:
            return None
        if metadata['preserve_excess']:
            outcome = (
                "Excluded values are saved in metadata and restored unchanged "
                "by undo=True; operations on the unfolded tensor do not "
                "affect them."
            )
        else:
            outcome = (
                "preserve_excess=False does not save excluded values; "
                "undo=True returns only the cropped tensor."
            )
        return (
            f"Center-crop unfolding with method='{metadata['method']}' excludes "
            "coordinates from the unfolded tensor: "
            f"{'; '.join(coverage)}. {outcome}"
        )

    def unfold(self, domain='real', method='row_major',
               curve_shape_strategy='center_crop', preserve_excess=True,
               resize_side=None, resize_side_mode='nearest',
               resize_method='linear', preserve_original=False,
               undo=False, metadata=None, original_shape=None,
               return_metadata=False, plot_traversal=False, plot_kwargs=None,
               warn_on_crop=True):
        """
        Unfold or restore a 4D-STEM tensor using explicit domain/method choices.

        The tensor convention is ``(Ry, Rx, Ky, Kx)``. ``domain`` controls which
        coordinate grid is unfolded, while ``method`` controls the coordinate
        traversal or axis alignment strategy.

        Parameters
        ----------
        domain : {'real', 'reciprocal', 'both'}, optional
            Coordinate domain to unfold. ``'real'`` produces a stack of
            diffraction patterns, ``'reciprocal'`` produces a stack of
            real-space images along the last axis, and ``'both'`` produces a
            2D matrix.
        method : str, optional
            Traversal or axis-alignment method. Supported traversal methods are
            ``'row_major'``, ``'serpentine'``, ``'spiral'``,
            ``'diagonal_zigzag'``, ``'hilbert'``, ``'morton'``/``'z_order'``,
            ``'peano'``, ``'peano_meander'``, ``'meander-4'``, and
            ``'meander-5'``. ``'moore'`` is reserved and currently raises
            ``NotImplementedError``. ``coordinate_aligned`` is supported for
            ``domain='real'`` and ``domain='reciprocal'`` only.
        curve_shape_strategy : {'center_crop', 'resize'}, optional
            How compatible curve/block methods handle incompatible shapes.
            ``'center_crop'`` unfolds the centered compatible square or
            block-meander rectangle.
            ``'resize'`` first resizes the selected traversal domain to a
            compatible square. ``meander-4`` and ``meander-5`` currently support
            center-crop only.
        preserve_excess : bool, optional
            For center-crop curve methods, store cropped-out values in metadata
            so undo can reconstruct the original full tensor exactly. This
            consumes space proportional to the excluded data. Value-only
            operations on an unfolded object leave excluded values unchanged.
        warn_on_crop : bool, optional
            Warn when center-cropping excludes coordinates from the unfolded
            tensor. The warning reports the selected region and whether undo
            can restore the excluded values. Set False for an intentional crop.
        resize_side : int or None, optional
            Explicit compatible side length for resize mode. Hilbert, Morton,
            Z-order, and Moore require powers of 2; Peano requires powers of 3.
            ``meander-4`` and ``meander-5`` do not use this parameter.
        resize_side_mode : {'nearest', 'downsample', 'upsample'}, optional
            How to choose a compatible side when ``resize_side`` is omitted.
        resize_method : {'linear', 'nearest', 'area'}, optional
            Method passed to :meth:`resize` in resize mode.
        preserve_original : bool, optional
            For resize mode, store the original tensor in metadata so undo can
            reconstruct it exactly. This requires one full-tensor copy;
            otherwise undo returns the resized tensor. Shape-preserving
            operations share the stored read-only values rather than copying
            them again. ``metadata['preserved_values_nbytes']`` reports their
            total size.
        undo : bool, optional
            If True, restore an unfolded object using ``metadata`` or this
            object's attached ``unfold_metadata``. If no metadata is available,
            row-major undo is supported using ``original_shape`` or square-grid
            inference.
        metadata : dict or None, optional
            Metadata returned by a previous unfolding. Required for undo unless
            this object already has attached metadata, or the requested undo can
            be handled as row-major using ``original_shape``.
        original_shape : tuple or None, optional
            Shape used for metadata-less row-major undo. The full
            ``(Ry, Rx, Ky, Kx)`` shape is always accepted. For 3D
            ``domain='real'`` stacks, ``(Ry, Rx)`` is also accepted and
            ``(Ky, Kx)`` is taken from the stack. For 3D
            ``domain='reciprocal'`` stacks, ``(Ky, Kx)`` is also accepted and
            ``(Ry, Rx)`` is taken from the stack.
        return_metadata : bool, optional
            If True, return ``(result, metadata)``.
        plot_traversal : bool, optional
            Plot the selected traversal when unfolding. For ``domain='both'``,
            show separate real- and reciprocal-space paths. Not available for
            ``undo=True`` or ``method='coordinate_aligned'``.
        plot_kwargs : dict or None, optional
            Options passed to :func:`plot_traversals`, except ``method``, which
            always matches this unfolding. By default ``grid_shape`` is the
            actual traversal grid (after any resize). Set ``grid_shape`` for a
            smaller preview; that figure may differ from the exact path used
            on the data. Other options include ``cmap``, ``linewidth``,
            ``show_grid``, ``mark_endpoints``, ``ncols``, and ``show``.
            ``show=False`` leaves the figure open without displaying it.

        Returns
        -------
        HyperData or tuple[HyperData, dict]
            New object containing the unfolded or restored data. The original
            object is not modified.

        Examples
        --------
        >>> original = np.arange(2*3*4*5).reshape(2, 3, 4, 5)
        >>> hd = HyperData(original)
        >>> unfolded, meta = hd.unfold(domain='real',
        ...                            method='hilbert',
        ...                            preserve_excess=True,
        ...                            return_metadata=True)
        >>> restored = unfolded.unfold(undo=True)
        >>> np.array_equal(restored.array, original)
        True
        >>> restored = HyperData(unfolded.array).unfold(undo=True, metadata=meta)
        >>> np.array_equal(restored.array, original)
        True
        >>> restored = HyperData(unfolded.array).unfold(
        ...     undo=True,
        ...     domain='real',
        ...     original_shape=original.shape,
        ... )
        >>> preview = hd.unfold(
        ...     method='hilbert', plot_traversal=True,
        ...     plot_kwargs={'grid_shape': (16, 16), 'cmap': 'viridis'},
        ... )
        """
        if not isinstance(plot_traversal, (bool, np.bool_)):
            raise TypeError("plot_traversal must be a boolean.")
        if not isinstance(warn_on_crop, (bool, np.bool_)):
            raise TypeError("warn_on_crop must be a boolean.")
        if plot_kwargs is not None and not isinstance(plot_kwargs, dict):
            raise TypeError("plot_kwargs must be a dictionary or None.")
        if plot_kwargs and not plot_traversal:
            raise ValueError("Set plot_traversal=True to use plot_kwargs.")
        if plot_traversal and undo:
            raise ValueError("plot_traversal is only available when unfolding, not undoing.")

        plot_options = dict(plot_kwargs or {})
        if plot_traversal:
            if 'method' in plot_options:
                raise ValueError("plot_kwargs cannot override the unfolding method.")
            valid_plot_options = {
                'grid_shape', 'ncols', 'cmap', 'linewidth',
                'show_grid', 'mark_endpoints', 'show',
            }
            unknown_options = set(plot_options) - valid_plot_options
            if unknown_options:
                raise TypeError(
                    f"Unknown plot_kwargs: {', '.join(sorted(unknown_options))}."
                )

        if undo:
            if metadata is None:
                metadata = getattr(self, 'unfold_metadata', None)

            restored, metadata = _unfold_array(
                self.array,
                domain=domain,
                method=method,
                undo=True,
                metadata=metadata,
                original_shape=original_shape,
                return_metadata=True,
            )
            result = self._spawn(restored)
            result.unfold_metadata = None

            if return_metadata:
                return result, metadata
            return result

        domain, method = _normalize_unfold_request(domain=domain, method=method)
        if plot_traversal and method == 'coordinate_aligned':
            raise ValueError(
                "coordinate_aligned rearranges axes, so it has no 2D traversal to plot."
            )
        if not isinstance(curve_shape_strategy, str):
            raise ValueError("curve_shape_strategy must be a string.")
        curve_shape_strategy = curve_shape_strategy.lower()
        working = self
        original_values = None

        if method in _CURVE_TRAVERSAL_METHODS and curve_shape_strategy == 'resize':
            if domain == 'both':
                real_side = _select_resize_side(
                    self.shape[:2],
                    method,
                    resize_side=resize_side,
                    resize_side_mode=resize_side_mode,
                )
                reciprocal_side = _select_resize_side(
                    self.shape[2:4],
                    method,
                    resize_side=resize_side,
                    resize_side_mode=resize_side_mode,
                )
                working = working.resize(
                    (real_side, real_side),
                    domain='real',
                    method=resize_method,
                )
                working = working.resize(
                    (reciprocal_side, reciprocal_side),
                    domain='reciprocal',
                    method=resize_method,
                )
            elif domain == 'real':
                side = _select_resize_side(
                    self.shape[:2],
                    method,
                    resize_side=resize_side,
                    resize_side_mode=resize_side_mode,
                )
                working = working.resize(
                    (side, side),
                    domain='real',
                    method=resize_method,
                )
            elif domain == 'reciprocal':
                side = _select_resize_side(
                    self.shape[2:4],
                    method,
                    resize_side=resize_side,
                    resize_side_mode=resize_side_mode,
                )
                working = working.resize(
                    (side, side),
                    domain='reciprocal',
                    method=resize_method,
                )

            if preserve_original:
                original_values = self.array

        unfolded, metadata = _unfold_array(
            working.array,
            domain=domain,
            method=method,
            curve_shape_strategy=curve_shape_strategy,
            preserve_excess=preserve_excess,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
            resize_method=resize_method,
            preserve_original=preserve_original,
            original_shape=self.shape,
            original_values=original_values,
            return_metadata=True,
        )
        result = working._spawn(unfolded)
        result.unfold_metadata = metadata

        if warn_on_crop:
            notice = self._center_crop_notice(metadata)
            if notice is not None:
                warnings.warn(notice, UserWarning, stacklevel=2)

        if plot_traversal:
            preview_shape = plot_options.pop('grid_shape', None)
            show_plot = plot_options.pop('show', True)
            domains_to_plot = ('real', 'reciprocal') if domain == 'both' else (domain,)
            for plotted_domain in domains_to_plot:
                axis_slice = slice(0, 2) if plotted_domain == 'real' else slice(2, 4)
                grid_shape = tuple(working.shape[axis_slice])
                plotted_shape = grid_shape if preview_shape is None else preview_shape
                fig, _ = plot_traversals(
                    method, grid_shape=plotted_shape, show=False, **plot_options,
                )
                title = f"{plotted_domain.capitalize()}-space traversal"
                if preview_shape is not None:
                    title += f" preview (data grid {grid_shape[0]}x{grid_shape[1]})"
                else:
                    original_grid = tuple(self.shape[axis_slice])
                    if original_grid != grid_shape:
                        title += (
                            f" (resized from {original_grid[0]}x{original_grid[1]}"
                            f" to {grid_shape[0]}x{grid_shape[1]})"
                        )
                fig.suptitle(title)
            if show_plot:
                plt.show()

        if return_metadata:
            return result, metadata
        return result


    def reshape(self, *shape, order='C', return_array=False):
        """
        Return a row-major reshaped copy of this dataset.

        This is a thin HyperData wrapper around ``numpy.reshape``. It is useful
        when the data are already unfolded/folded in standard row-major order
        and the desired target shape is known directly.

        Parameters
        ----------
        *shape : int or tuple
            Target shape, either as separate dimensions or as one tuple. NumPy's
            ``-1`` inference is supported.
        order : {'C', 'F', 'A'}, optional
            Reshape order passed to ``numpy.reshape``. The default ``'C'`` is
            Python/NumPy row-major order.
        return_array : bool, optional
            If True, return the reshaped ndarray instead of a ``HyperData``.

        Examples
        --------
        >>> folded = HyperData(stack).reshape(Ry, Rx, Ky, Kx)
        >>> stack = HyperData(folded.array).reshape(Ry * Rx, Ky, Kx)
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
            new_shape = tuple(int(v) for v in shape[0])
        else:
            new_shape = tuple(int(v) for v in shape)

        if len(new_shape) < 2:
            raise ValueError("HyperData.reshape requires at least 2 dimensions.")
        if not isinstance(order, str):
            raise ValueError("order must be a string accepted by numpy.reshape.")

        try:
            reshaped = np.reshape(self.array, new_shape, order=order)
        except ValueError as exc:
            raise ValueError(
                f"Cannot reshape HyperData from {self.shape} to {new_shape}: {exc}"
            ) from exc

        if return_array:
            return reshaped
        return self._spawn(
            reshaped,
            rq_calibration=(self.rq_calibration if reshaped.shape[-2:] == self.pattern_shape else None),
        )


    def swap_domains(self):
        """
        Swap the real space and reciprocal space coordinates in the 4D dataset.
        For a dataset with dimensions (A, B, C, D),
        this method swaps them to (C, D, A, B).
        
        Returns:
        - swapped_data (numpy.ndarray): The 4D dataset with swapped dimensions.
        """
        
        # The original order is (0, 1, 2, 3) and we want to change to (2, 3, 0, 1)
        swapped_data = np.transpose(self.array, (2, 3, 0, 1))
        reciprocal_factor = self.real_conv_factor
        reciprocal_units = self.real_units
        if reciprocal_factor is not None:
            y_step, x_step = _real_spacing_pair(reciprocal_factor)
            if y_step > 0 and np.isclose(y_step, x_step):
                reciprocal_factor = y_step
            else:
                reciprocal_factor = None
                reciprocal_units = None
                warnings.warn(
                    "swap_domains cleared signed or anisotropic real-space "
                    "calibration because reciprocal-space calibration is "
                    "positive scalar-only.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._spawn(
            swapped_data,
            real_units=self.reciprocal_units,
            real_conv_factor=self.reciprocal_conv_factor,
            real_origin=(0.0, 0.0),
            reciprocal_units=reciprocal_units,
            reciprocal_conv_factor=reciprocal_factor,
            center_beam_metadata=None,
            rq_calibration=(
                RQCalibration.from_matrix(self.rq_calibration.inverse_matrix, source='domain_swap')
                if self.rq_calibration is not None else None
            ),
        )
    
    
    def alignment(self, r_center=5, iterations=1,
                  center=None, method='com', search_radius=None,
                  enforce_square=False, fit_radius=False, radius_range=None,
                  radius_step=1, radius_operation='mean',
                  radius_reference_dp=None, template='disk',
                  trench_width=1.0, kernel_amp=1.0, trench_amp=-0.5,
                  gaussian_sigma=None):
        """
        Align the diffraction patterns through the Center of mass of the center beam

        Parameters
        ----------
        r_center : float
            Radius of the central peak or disk in pixels.
        iterations : int
            Number of alignment refinement passes. For ``method='disk'`` or
            ``method='template'``, the first pass crops and aligns to a common
            output shape; additional passes re-fit the center on the aligned
            stack and apply residual subpixel shifts.
        center : tuple or None
            Approximate reference disk center ``(ky, kx)``. Defaults to the
            geometric k-space center ``((ky - 1)/2, (kx - 1)/2)``. When
            ``fit_radius=True`` or ``radius_range`` is provided, this center is
            refined on the representative diffraction pattern before aligning
            individual diffraction patterns.
        method : {'com', 'disk', 'template'}
            ``'com'`` uses the original center-of-mass workflow. ``'disk'`` or
            ``'template'`` uses circular template matching, centered cropping,
            and a final affine subpixel shift.
        template : {'disk', 'negative_trench', 'gaussian'}
            Template used by the template-matching workflow. ``'disk'`` is the
            historical behavior: a filled circular disk, mean-subtracted and
            normalized. ``'negative_trench'`` uses a positive disk surrounded
            by a negative ring. ``'gaussian'`` uses a centered 2D Gaussian.
        search_radius : float or None
            Maximum disk-center translation, in pixels, around the current
            reference center. During reference fitting, the search is centered
            on ``center``. After the representative disk center is found, the
            same radius is centered on that fitted reference center for all
            per-pattern template matching. If None, the full diffraction
            pattern is searched. This is a translation search radius, not a
            radius-range search for the disk size.
        enforce_square : bool
            If True, force the disk-template output to satisfy ``ky == kx``.
        fit_radius : bool
            If True, estimate the best disk-template radius from a
            representative diffraction pattern before aligning all diffraction
            patterns.
        radius_range : tuple or sequence or None
            Candidate radii used when ``fit_radius=True``. A two-value tuple is
            interpreted as ``(r_min, r_max)`` and sampled by ``radius_step``.
            A longer sequence is used directly as the candidate radii. If None,
            a local radius search around ``r_center`` is used.
        radius_step : float
            Step size for two-value ``radius_range`` searches.
        radius_operation : str
            Operation passed to ``get_dp(operation=...)`` to build the
            representative diffraction pattern used for radius fitting when
            ``radius_reference_dp`` is not provided.
        radius_reference_dp : array-like or data object, optional
            Direct 2D diffraction pattern used as the representative pattern
            for disk-radius fitting. This can be a plain ``ndarray`` or an
            object with an ``.array`` attribute, such as ``ReciprocalSpace``.
            The shape must match the reciprocal-space shape ``(ky, kx)``.
            Providing this parameter implies radius fitting and bypasses
            ``radius_operation`` for the reference pattern.
        trench_width : float
            Width of the negative ring used by ``template='negative_trench'``.
        kernel_amp : float
            Amplitude of the positive disk for ``template='negative_trench'``.
        trench_amp : float
            Amplitude of the surrounding trench for
            ``template='negative_trench'``. Negative values are usually useful.
        gaussian_sigma : float or None
            Standard deviation for ``template='gaussian'``. If None, defaults
            to ``r_center / 2`` for each tested radius.

        Notes
        -----
        ``method='com'`` is useful when the center beam is compact, bright,
        roughly isolated, and its center of mass is a stable estimate. Gaussian
        template matching is useful when the central beam is more peak-like or
        blobby than disk-like. Negative trench template matching is better when
        the central beam is disk-like with a clear edge, especially when the
        local background is sloped or slowly varying.
        """

        if self.ndim != 4:
            raise ValueError("alignment currently requires a 4D dataset.")
        if not isinstance(method, str):
            raise ValueError("method must be 'com', 'disk', or 'template'.")

        method = method.lower()
        if method in ('center_of_mass', 'centre_of_mass'):
            method = 'com'
        elif method in ('circle', 'template_matching', 'template'):
            method = 'disk'
        if method not in ('com', 'disk'):
            raise ValueError("method must be 'com', 'disk', or 'template'.")
        if radius_reference_dp is not None and method != 'disk':
            raise ValueError(
                "radius_reference_dp is only valid with method='disk' or "
                "method='template'."
            )
        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            raise ValueError("iterations must be a positive integer.")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer.")

        y, x, ky, kx = self.shape
        if center is None:
            center = ((ky - 1) / 2, (kx - 1) / 2)
        center_y, center_x = tuple(float(v) for v in center)
        if search_radius is not None and float(search_radius) < 0:
            raise ValueError("search_radius must be non-negative or None.")

        def _normalize_alignment_template_name(template_name):
            if not isinstance(template_name, str):
                raise ValueError(
                    "template must be 'disk', 'negative_trench', or 'gaussian'."
                )
            normalized = template_name.strip().lower().replace('-', '_')
            if normalized in ('disk', 'circle', 'filled_disk'):
                return 'disk'
            if normalized in (
                'negative_trench',
                'neg_trench',
                'disk_trench',
                'trench',
            ):
                return 'negative_trench'
            if normalized in ('gaussian', 'gauss'):
                return 'gaussian'
            raise ValueError(
                "template must be 'disk', 'negative_trench', or 'gaussian'."
            )

        template = _normalize_alignment_template_name(template)
        if method == 'com' and template != 'disk':
            raise ValueError(
                "template is only used with method='disk' or method='template'."
            )
        if (
            not np.isscalar(trench_width)
            or not np.isfinite(trench_width)
            or float(trench_width) <= 0
        ):
            raise ValueError("trench_width must be a positive finite scalar.")
        if not np.isscalar(kernel_amp) or not np.isfinite(kernel_amp):
            raise ValueError("kernel_amp must be a finite scalar.")
        if not np.isscalar(trench_amp) or not np.isfinite(trench_amp):
            raise ValueError("trench_amp must be a finite scalar.")
        if gaussian_sigma is not None and (
            not np.isscalar(gaussian_sigma)
            or not np.isfinite(gaussian_sigma)
            or float(gaussian_sigma) <= 0
        ):
            raise ValueError("gaussian_sigma must be positive, finite, or None.")

        def _normalize_template_kernel(kernel):
            kernel = np.asarray(kernel, dtype=float)
            kernel -= np.mean(kernel)
            norm = np.linalg.norm(kernel)
            if norm <= 0:
                raise ValueError(
                    "The selected alignment template has zero contrast after "
                    "normalization. Adjust r_center or template parameters."
                )
            return kernel / norm

        def _alignment_template(radius):
            radius = float(radius)
            if radius <= 0:
                raise ValueError("r_center must be positive.")
            half_size = int(np.ceil(radius))
            if template == 'negative_trench':
                half_size = int(np.ceil(radius + float(trench_width)))
            elif template == 'gaussian':
                sigma = (
                    float(gaussian_sigma)
                    if gaussian_sigma is not None
                    else max(float(radius) / 2.0, 1e-6)
                )
                half_size = max(1, int(np.ceil(3 * sigma)))

            coords = np.arange(-half_size, half_size + 1)
            yy, xx = np.meshgrid(coords, coords, indexing='ij')
            rr = np.hypot(yy, xx)

            if template == 'disk':
                kernel = (rr <= radius).astype(float)
            elif template == 'negative_trench':
                kernel = np.zeros_like(rr, dtype=float)
                kernel[rr <= radius] = float(kernel_amp)
                trench_mask = (rr > radius) & (
                    rr <= radius + float(trench_width)
                )
                kernel[trench_mask] = float(trench_amp)
            else:
                sigma = (
                    float(gaussian_sigma)
                    if gaussian_sigma is not None
                    else max(float(radius) / 2.0, 1e-6)
                )
                kernel = np.exp(-(rr**2) / (2 * sigma**2))

            return _normalize_template_kernel(kernel)

        def _candidate_template_radii():
            step = float(radius_step)
            if step <= 0:
                raise ValueError("radius_step must be positive.")

            if radius_range is None:
                span = max(2.0, 0.25 * float(r_center))
                r_min = max(1.0, float(r_center) - span)
                r_max = float(r_center) + span
                return np.arange(r_min, r_max + 0.5 * step, step)

            radii = np.asarray(radius_range, dtype=float).ravel()
            if radii.size == 2:
                r_min, r_max = radii
                if r_min > r_max:
                    r_min, r_max = r_max, r_min
                radii = np.arange(r_min, r_max + 0.5 * step, step)

            radii = radii[np.isfinite(radii) & (radii > 0)]
            if radii.size == 0:
                raise ValueError("radius_range must contain positive radii.")
            return radii

        def _search_bounds(center_value, search_radius_value, radius, shape=None):
            if shape is None:
                shape = (ky, kx)
            shape_y, shape_x = tuple(int(v) for v in shape)
            if search_radius_value is None:
                return 0, shape_y, 0, shape_x

            cy, cx = tuple(float(v) for v in center_value)
            search_extent = float(search_radius_value) + int(np.ceil(radius)) + 2
            y0 = max(0, int(np.floor(cy - search_extent)))
            y1 = min(shape_y, int(np.ceil(cy + search_extent)) + 1)
            x0 = max(0, int(np.floor(cx - search_extent)))
            x1 = min(shape_x, int(np.ceil(cx + search_extent)) + 1)

            if y0 >= y1 or x0 >= x1:
                raise ValueError(
                    "search_radius neighborhood does not overlap the "
                    "diffraction pattern."
                )
            return y0, y1, x0, x1

        def _normalized_template_correlation(dp_region, template):
            corr = fftconvolve(dp_region, template[::-1, ::-1], mode='same')
            n_pix = template.size
            # A box filter gives the same zero-padded support sums without two FFTs.
            local_sum = uniform_filter(
                dp_region, size=template.shape, mode='constant', cval=0.0
            ) * n_pix
            local_sum_sq = uniform_filter(
                dp_region**2, size=template.shape, mode='constant', cval=0.0
            ) * n_pix
            local_energy = local_sum_sq - (local_sum**2 / n_pix)
            local_energy = np.maximum(local_energy, 0)
            denom = np.sqrt(local_energy)
            return np.divide(
                corr,
                denom,
                out=np.zeros_like(corr, dtype=float),
                where=denom > 0,
            )

        def _integer_template_match(dp, radius, search_radius_value=None,
                                    center_value=None, template=None):
            if center_value is None:
                center_value = (center_y, center_x)
            dp = np.asarray(dp)
            if dp.ndim != 2:
                raise ValueError("Template matching requires a 2D diffraction pattern.")
            pattern_ky, pattern_kx = dp.shape

            if template is None:
                template = _alignment_template(radius)
            template_extent = (max(template.shape) - 1) / 2.0

            y0, y1, x0, x1 = _search_bounds(
                center_value,
                search_radius_value,
                template_extent,
                shape=(pattern_ky, pattern_kx),
            )
            dp_region = np.asarray(dp[y0:y1, x0:x1], dtype=float)
            corr_region = _normalized_template_correlation(dp_region, template)

            if search_radius_value is not None:
                search_mask = make_mask(
                    (center_value[0] - y0, center_value[1] - x0),
                    float(search_radius_value),
                    mask_dim=corr_region.shape,
                )
                if not np.any(search_mask):
                    raise ValueError(
                        "search_radius neighborhood contains no valid "
                        "diffraction-pattern coordinates."
                    )
                corr_region = np.where(search_mask, corr_region, -np.inf)

            local_max_idx = np.unravel_index(
                np.nanargmax(corr_region),
                corr_region.shape,
            )
            max_idx = (local_max_idx[0] + y0, local_max_idx[1] + x0)

            return corr_region, max_idx, corr_region[local_max_idx], (y0, x0)

        def _resolve_radius_reference_dp():
            if radius_reference_dp is None:
                representative = self.get_dp(operation=radius_operation)
                source_label = f"operation='{radius_operation}'"
            else:
                representative = radius_reference_dp
                source_label = 'radius_reference_dp'

            representative_dp = (
                representative.array
                if hasattr(representative, 'array')
                else np.asarray(representative)
            )
            representative_dp = np.asarray(representative_dp)
            if representative_dp.shape != (ky, kx):
                raise ValueError(
                    f"The radius-fitting reference from {source_label} must "
                    "be a single 2D diffraction "
                    f"pattern with shape {(ky, kx)}; got "
                    f"{representative_dp.shape}."
                )
            if np.iscomplexobj(representative_dp):
                raise TypeError(
                    f"The radius-fitting reference from {source_label} must "
                    "be real-valued; complex diffraction patterns are not "
                    "supported for radius fitting."
                )
            if not np.all(np.isfinite(representative_dp)):
                raise ValueError(
                    f"The radius-fitting reference from {source_label} must "
                    "contain only finite values."
                )
            return representative_dp.astype(float, copy=False), source_label

        def _fit_template_radius():
            representative_dp, source_label = _resolve_radius_reference_dp()
            radii = _candidate_template_radii()

            best_radius = float(radii[0])
            best_center = (center_y, center_x)
            best_score = -np.inf
            for candidate_radius in radii:
                _, match_center, score, _ = _integer_template_match(
                    representative_dp,
                    candidate_radius,
                    search_radius_value=search_radius,
                    center_value=(center_y, center_x),
                )
                if score > best_score:
                    best_score = score
                    best_radius = float(candidate_radius)
                    best_center = match_center

            return best_radius, best_center, best_score, source_label

        def _fit_disk_centers(array, reference_center):
            fit_y = np.zeros((y, x), dtype=float)
            fit_x = np.zeros_like(fit_y)
            pattern_ky, pattern_kx = array.shape[-2:]

            pattern_search_radius = search_radius
            center_template = _alignment_template(effective_r_center)

            for i in tqdm(range(y), desc=f"Template-matching {template} centers"):
                for j in range(x):
                    corr, max_idx, _, corr_origin = _integer_template_match(
                        array[i, j],
                        effective_r_center,
                        search_radius_value=pattern_search_radius,
                        center_value=reference_center,
                        template=center_template,
                    )

                    refine_radius = max(2, int(np.ceil(effective_r_center / 4)))
                    y0 = max(0, max_idx[0] - refine_radius - corr_origin[0])
                    y1 = min(
                        corr.shape[0], max_idx[0] + refine_radius + 1 - corr_origin[0]
                    )
                    x0 = max(0, max_idx[1] - refine_radius - corr_origin[1])
                    x1 = min(
                        corr.shape[1], max_idx[1] + refine_radius + 1 - corr_origin[1]
                    )
                    patch = corr[y0:y1, x0:x1]
                    finite_mask = np.isfinite(patch)
                    if not np.any(finite_mask):
                        fit_y[i, j] = np.clip(reference_center[0], 0, pattern_ky - 1)
                        fit_x[i, j] = np.clip(reference_center[1], 0, pattern_kx - 1)
                        continue

                    finite_patch = np.where(finite_mask, patch, np.nan)
                    finite_min = np.min(patch[finite_mask])
                    weights = finite_patch - finite_min
                    weights = np.nan_to_num(weights, nan=0.0)

                    total = np.sum(weights)
                    if total > 0:
                        yy, xx = np.indices(weights.shape)
                        fit_y[i, j] = (
                            corr_origin[0] + y0 + np.sum(yy * weights) / total
                        )
                        fit_x[i, j] = (
                            corr_origin[1] + x0 + np.sum(xx * weights) / total
                        )
                    else:
                        fit_y[i, j], fit_x[i, j] = max_idx

            return fit_y, fit_x

        def _common_centered_crop_shape(fit_y, fit_x):
            max_heights = np.zeros_like(fit_y, dtype=int)
            max_widths = np.zeros_like(fit_x, dtype=int)

            for i in range(y):
                for j in range(x):
                    cy = int(np.clip(round(fit_y[i, j]), 0, ky - 1))
                    cx = int(np.clip(round(fit_x[i, j]), 0, kx - 1))
                    max_heights[i, j] = 2 * min(cy, ky - 1 - cy) + 1
                    max_widths[i, j] = 2 * min(cx, kx - 1 - cx) + 1

            out_ky = int(np.min(max_heights))
            out_kx = int(np.min(max_widths))
            if enforce_square:
                out_ky = out_kx = min(out_ky, out_kx)
            if out_ky < 2 * effective_r_center + 1 or out_kx < 2 * effective_r_center + 1:
                print(
                    "Warning: aligned crop is smaller than the fitted disk "
                    "diameter for at least one diffraction pattern."
                )
            if out_ky <= 0 or out_kx <= 0:
                raise ValueError("Could not determine a valid common crop shape.")
            return out_ky, out_kx

        def _crop_bounds(center_value, output_size, axis_size):
            center_idx = int(np.clip(round(center_value), 0, axis_size - 1))
            start = center_idx - output_size // 2
            end = start + output_size
            if start < 0:
                start = 0
                end = output_size
            if end > axis_size:
                end = axis_size
                start = axis_size - output_size
            return int(start), int(end)

        def _shift_disk_centers_to_target(array, fit_y, fit_x, target_center,
                                          desc):
            target_y, target_x = tuple(float(v) for v in target_center)
            pattern_ky, pattern_kx = array.shape[-2:]
            shifted = np.zeros(
                array.shape,
                dtype=np.result_type(array.dtype, np.float64),
            )
            for i in tqdm(range(y), desc=desc):
                for j in range(x):
                    shift_y = target_y - fit_y[i, j]
                    shift_x = target_x - fit_x[i, j]
                    afine_tf = transform.AffineTransform(
                        translation=(shift_x, shift_y)
                    )
                    shifted[i, j] = transform.warp(
                        array[i, j],
                        inverse_map=afine_tf.inverse,
                        output_shape=(pattern_ky, pattern_kx),
                        preserve_range=True,
                    )
            return shifted

        radius_fit_requested = bool(
            fit_radius
            or radius_range is not None
            or radius_reference_dp is not None
        )
        effective_r_center = float(r_center)
        reference_center = (center_y, center_x)
        radius_score = None

        if method == 'disk':
            if radius_fit_requested:
                (
                    effective_r_center,
                    reference_center,
                    radius_score,
                    radius_source_label,
                ) = _fit_template_radius()
                print(
                    f"Selected {template} template reference from "
                    f"{radius_source_label}: radius={effective_r_center:.4f}, "
                    f"center=({reference_center[0]:.4f}, {reference_center[1]:.4f}) "
                    f"(score={radius_score:.4g})."
                )

            fit_y, fit_x = _fit_disk_centers(self.array, reference_center)
            initial_std_center = (np.std(fit_y), np.std(fit_x))
            initial_mean_center = (np.mean(fit_y), np.mean(fit_x))
            print(f'Initial disk-center standard deviation (ky, kx): ({initial_std_center[0]:.4f}, {initial_std_center[1]:.4f})')
            print(f'Initial disk center (ky, kx): ({initial_mean_center[0]:.4f}, {initial_mean_center[1]:.4f})')

            out_ky, out_kx = _common_centered_crop_shape(fit_y, fit_x)
            target_y, target_x = (out_ky - 1) / 2, (out_kx - 1) / 2
            aligned = np.zeros(
                (y, x, out_ky, out_kx),
                dtype=np.result_type(self.dtype, np.float64),
            )

            print(f'Cropping diffraction patterns to common k-space shape ({out_ky}, {out_kx}).')
            for i in tqdm(range(y), desc='Cropping and affine-aligning disks'):
                for j in range(x):
                    y0, y1 = _crop_bounds(fit_y[i, j], out_ky, ky)
                    x0, x1 = _crop_bounds(fit_x[i, j], out_kx, kx)
                    cropped = self.array[i, j, y0:y1, x0:x1]
                    center_rel_y = fit_y[i, j] - y0
                    center_rel_x = fit_x[i, j] - x0
                    shift_y = target_y - center_rel_y
                    shift_x = target_x - center_rel_x
                    afine_tf = transform.AffineTransform(
                        translation=(shift_x, shift_y)
                    )
                    aligned[i, j] = transform.warp(
                        cropped,
                        inverse_map=afine_tf.inverse,
                        output_shape=(out_ky, out_kx),
                        preserve_range=True,
                    )

            target_center = (target_y, target_x)
            for idx in range(1, iterations):
                fit_y, fit_x = _fit_disk_centers(aligned, target_center)
                iter_std_center = (np.std(fit_y), np.std(fit_x))
                iter_mean_center = (np.mean(fit_y), np.mean(fit_x))
                print(
                    f'Disk-center residual before iteration ({idx+1}/{iterations}) '
                    f'standard deviation (ky, kx): '
                    f'({iter_std_center[0]:.4f}, {iter_std_center[1]:.4f})'
                )
                print(
                    f'Disk center before iteration ({idx+1}/{iterations}) '
                    f'(ky, kx): ({iter_mean_center[0]:.4f}, '
                    f'{iter_mean_center[1]:.4f})'
                )
                aligned = _shift_disk_centers_to_target(
                    aligned,
                    fit_y,
                    fit_x,
                    target_center,
                    desc=f'Affine-aligning disk residuals ({idx+1}/{iterations})',
                )

            final_fit_y, final_fit_x = _fit_disk_centers(aligned, target_center)
            final_std_center = (np.std(final_fit_y), np.std(final_fit_x))
            final_mean_center = (np.mean(final_fit_y), np.mean(final_fit_x))
            print(f'Final disk-center standard deviation (ky, kx): ({final_std_center[0]:.4f}, {final_std_center[1]:.4f})')
            print(f'Final disk center (ky, kx): ({final_mean_center[0]:.4f}, {final_mean_center[1]:.4f})')

            aligned_obj = self._spawn(
                aligned,
                reciprocal_units=self.reciprocal_units,
                reciprocal_conv_factor=self.reciprocal_conv_factor,
            )
            aligned_obj.center_beam_metadata = _center_beam_metadata_from_pixels(
                effective_r_center,
                final_mean_center,
                (out_ky, out_kx),
                units=self.reciprocal_units,
                conv_factor=self.reciprocal_conv_factor,
                source='alignment',
                alignment_method='disk',
                template=template,
                iterations=iterations,
                radius_fit_requested=radius_fit_requested,
                radius_score=radius_score,
                reference_center_px=reference_center,
                target_center_px=target_center,
                initial_mean_fit_center_px=initial_mean_center,
                initial_std_fit_center_px=initial_std_center,
                mean_fit_center_px=final_mean_center,
                std_fit_center_px=final_std_center,
                original_reciprocal_shape=(ky, kx),
                output_reciprocal_shape=(out_ky, out_kx),
                search_radius_px=search_radius,
                radius_range=(
                    np.asarray(radius_range, dtype=float).tolist()
                    if radius_range is not None
                    else None
                ),
                radius_step=radius_step,
            )

            return aligned_obj

        com_y, com_x = self._quickCOM(r_mask=r_center, center=center) 
        cbed_tran = np.array(
            self.array, dtype=np.result_type(self.dtype, np.float32), copy=True
        )
        cbed_tran_Obj = self._spawn(cbed_tran)
        initial_std_com = (np.std(com_y), np.std(com_x))
        initial_mean_com = (np.mean(com_y), np.mean(com_x))
        std_com = initial_std_com
        mean_com = initial_mean_com
        
        print(f'Initial COM standard deviation (ky, kx): ({initial_std_com[0]:.4f}, {initial_std_com[1]:.4f})')
        print(f'Initial COM (ky, kx): ({initial_mean_com[0]:.4f}, {initial_mean_com[1]:.4f})')
        
        for idx in range(iterations):
            print()
            print(f'Processing {y} × {x} real-space positions. Iteration ({idx+1}/{iterations})...')        
            for i in tqdm(range(y), desc = 'Alignment Progress'):
                for j in range(x):
                    afine_tf = transform.AffineTransform(
                        translation=(
                            com_x[i, j] - center_x,
                            com_y[i, j] - center_y,
                        )
                    )
                    cbed_tran[i,j,:,:] = transform.warp(
                        cbed_tran[i,j,:,:],
                        inverse_map=afine_tf,
                        preserve_range=True,
                    )
        
            com_y, com_x = cbed_tran_Obj._quickCOM(
                r_mask=r_center,
                center=center,
            )
            std_com = (np.std(com_y), np.std(com_x))
            mean_com = (np.mean(com_y), np.mean(com_x))
            
            print(f'Iteration {idx+1} COM standard deviation (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
            print(f'Iteration {idx+1} COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')

        print(f'Final COM standard deviation (ky, kx): ({std_com[0]:.4f}, {std_com[1]:.4f})')
        print(f'Final COM (ky, kx): ({mean_com[0]:.4f}, {mean_com[1]:.4f})')

        cbed_tran_Obj.center_beam_metadata = _center_beam_metadata_from_pixels(
            r_center,
            (center_y, center_x),
            (ky, kx),
            units=self.reciprocal_units,
            conv_factor=self.reciprocal_conv_factor,
            source='alignment',
            alignment_method='com',
            iterations=iterations,
            initial_mean_fit_center_px=initial_mean_com,
            initial_std_fit_center_px=initial_std_com,
            mean_fit_center_px=mean_com,
            std_fit_center_px=std_com,
        )
        
        return cbed_tran_Obj
    
    def rotate_dps(self, angle, units='deg', order=3):
        """
        Rotate every diffraction pattern by a common angle.

        Cartesian diffraction patterns are spatially rotated on their last two
        axes. Polar patterns are rotated by periodically shifting the angular
        axis, so values crossing either edge re-enter at the opposite edge
        according to the 0/360-degree periodic boundary.

        Positive angles follow the SciPy convention and rotate Cartesian
        patterns counterclockwise. The polar transform stores increasing theta
        clockwise along the last array axis, so the equivalent positive
        rotation shifts polar columns toward lower indices.

        Parameters
        ----------
        angle : float
            Rotation angle shared by all diffraction patterns.
        units : {'deg', 'rad'}, optional
            Units of ``angle``. Common degree/radian name variants are accepted.
        order : int, optional
            Spline interpolation order from 0 to 5. This is used by Cartesian
            rotation and by non-integer polar column shifts. Integer polar
            shifts use an exact periodic roll and do not interpolate. For
            interpolated rotations, integer input is promoted to floating
            point so fractional intensities are not rounded away.

        Returns
        -------
        HyperData
            Rotated data with calibration and polar metadata preserved.
        """
        if self.ndim not in (3, 4):
            raise ValueError(
                "'rotate_dps' only supports 3D stacks of diffraction patterns "
                "or 4D-STEM datasets."
            )
        if (
            not np.isscalar(angle)
            or not np.isreal(angle)
            or not np.isfinite(angle)
        ):
            raise ValueError("angle must be a finite scalar.")
        if not isinstance(units, str):
            raise ValueError("units must be 'deg' or 'rad'.")
        if (
            isinstance(order, (bool, np.bool_))
            or not isinstance(order, (Integral, np.integer))
            or not 0 <= order <= 5
        ):
            raise ValueError("order must be an integer from 0 to 5.")

        normalized_units = units.strip().lower()
        if normalized_units in ('deg', 'degree', 'degrees'):
            angle_degrees = float(angle)
        elif normalized_units in ('rad', 'radian', 'radians'):
            angle_degrees = float(np.degrees(angle))
        else:
            raise ValueError("units must be 'deg' or 'rad'.")

        if self.is_polar:
            metadata = self.polar_metadata or {}
            axis_order = tuple(
                metadata.get('axis_order', ('radius', 'theta'))
            )
            if axis_order != ('radius', 'theta'):
                raise ValueError(
                    "Polar rotation requires the last two axes to be ordered "
                    "as ('radius', 'theta')."
                )

            n_theta = self.shape[-1]
            try:
                theta_range = np.asarray(
                    metadata.get('theta_range', (0.0, 360.0)),
                    dtype=float,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "polar_metadata['theta_range'] must contain two finite "
                    "numeric values."
                ) from exc
            if theta_range.shape != (2,) or not np.all(
                np.isfinite(theta_range)
            ):
                raise ValueError(
                    "polar_metadata['theta_range'] must contain two finite "
                    "numeric values."
                )
            theta_start, theta_stop = theta_range
            theta_span = theta_stop - theta_start
            if not np.isclose(theta_span, 360.0):
                raise ValueError(
                    "Periodic polar rotation requires a full 360-degree "
                    "angular range ordered from lower to higher angle."
                )

            theta_step = float(
                metadata.get('theta_step', theta_span / n_theta)
            )
            if not np.isfinite(theta_step) or theta_step <= 0:
                raise ValueError(
                    "polar_metadata['theta_step'] must be positive and finite."
                )
            expected_step = theta_span / n_theta
            if not np.isclose(theta_step, expected_step):
                raise ValueError(
                    "polar_metadata['theta_step'] is inconsistent with the "
                    "angular axis length and theta_range."
                )

            column_shift = -angle_degrees / theta_step
            integer_shift = int(np.rint(column_shift))
            if np.isclose(column_shift, integer_shift):
                new_data = np.roll(
                    self.array,
                    shift=integer_shift,
                    axis=-1,
                )
            else:
                working_dtype = np.result_type(self.dtype, np.float32)
                shift_vector = (0.0,) * (self.ndim - 1) + (column_shift,)
                new_data = ndimage.shift(
                    self.array.astype(working_dtype, copy=False),
                    shift=shift_vector,
                    order=int(order),
                    mode='grid-wrap',
                    prefilter=order > 1,
                )

            return self._spawn(
                new_data, center_beam_metadata=None,
                rq_calibration=_reframe_rq_calibration(
                    self.rq_calibration, reciprocal=_rq_rotation_matrix(angle_degrees),
                ),
            )

        output_dtype = (
            self.dtype if order == 0
            else np.result_type(self.dtype, np.float32)
        )
        if self.ndim == 4:
            y_size, x_size = self.shape[:2]
            output_shape = rotate(
                self.array[0, 0].astype(output_dtype, copy=False),
                angle_degrees,
                order=int(order),
            ).shape
            new_data = np.zeros(
                (y_size, x_size, *output_shape),
                dtype=output_dtype,
            )

            iterator = np.ndindex(y_size, x_size)
            iterator = tqdm(
                iterator,
                total=y_size * x_size,
                desc='Rotating Diffraction Patterns',
            )
            for y_idx, x_idx in iterator:
                new_data[y_idx, x_idx] = rotate(
                    self.array[y_idx, x_idx].astype(output_dtype, copy=False),
                    angle_degrees,
                    order=int(order),
                )
        else:
            n_patterns = self.shape[0]
            output_shape = rotate(
                self.array[0].astype(output_dtype, copy=False),
                angle_degrees,
                order=int(order),
            ).shape
            new_data = np.zeros(
                (n_patterns, *output_shape),
                dtype=output_dtype,
            )

            for idx in tqdm(
                range(n_patterns),
                desc='Rotating Diffraction Patterns',
            ):
                new_data[idx] = rotate(
                    self.array[idx].astype(output_dtype, copy=False),
                    angle_degrees,
                    order=int(order),
                )

        return self._spawn(
            new_data, center_beam_metadata=None,
            rq_calibration=_reframe_rq_calibration(
                self.rq_calibration, reciprocal=_rq_rotation_matrix(angle_degrees),
            ),
        )
    
    def standardize(self, method='local'):
        """
        Standardize the dataset using NumPy.
    
        Supports both 4D data (A, B, C, D) and 3D data (N, A, B):
    
        - For 4D data, the last two axes (C, D) are treated as the diffraction
          pattern dimensions, and the first two (A, B) as the real-space grid.
        - For 3D data, the last two axes (A, B) are treated as the diffraction
          pattern dimensions, and the first axis (N) indexes patterns.
    
        Parameters
        ----------
        method : str, optional
            The standardization method to use. Options are:
            - 'global': Standardize the dataset globally
              (entire dataset has zero mean and unit variance).
            - 'local': Standardize each individual image in the dataset
              (each pattern on the last two axes has zero mean and unit variance).
            Default is 'global'.
    
        Returns
        -------
        HyperData
            The standardized dataset with zero mean and unit variance, either
            globally or locally.
    
        Raises
        ------
        ValueError
            If an invalid method is specified, or if local standardization is
            applied to data with fewer than 2 dimensions.
        """
        methods = ['global', 'local']
        if method not in methods:
            raise ValueError(f"Valid methods are: {methods}")
    
        arr = self.array
    
        if method == 'global':
            # Global standardization: calculate mean and std over the entire dataset
            mean = np.mean(arr)
            std = np.std(arr)
            standardized_tensor = (arr - mean) / (std + 1)
    
        elif method == 'local':
            # Local standardization: standardize each pattern defined by the last two axes
            if arr.ndim < 2:
                raise ValueError("Local standardization requires at least 2 dimensions")
    
            # Compute mean and std for each image on the last two dimensions
            axes = tuple(range(arr.ndim - 2, arr.ndim))  # last two axes
            mean = np.mean(arr, axis=axes, keepdims=True)
            std = np.std(arr, axis=axes, keepdims=True)
    
            # Standardize each image independently
            standardized_tensor = (arr - mean) / (std + 1)
    
        return self._spawn(standardized_tensor, preserve_unfold=True)

    def normalize(self, method='global'):
        """
        Normalize the dataset to the range [0, 1] using NumPy.
    
        Supports both 4D data (A, B, C, D) and 3D data (N, A, B):
    
        - For 4D data, the last two axes (C, D) are treated as the diffraction
          pattern dimensions, and the first two (A, B) as the real-space grid.
        - For 3D data, the last two axes (A, B) are treated as the diffraction
          pattern dimensions, and the first axis (N) indexes patterns.
    
        Parameters
        ----------
        method : str, optional
            The normalization method to use. Options are:
            - 'global': Normalize using the global minimum and maximum of the
              entire dataset (one min/max for the whole array).
            - 'local': Normalize each individual pattern independently, using
              min/max over the last two axes (each pattern maps to [0, 1]).
            Default is 'global'.
    
        Returns
        -------
        HyperData
            New HyperData instance containing the normalized dataset with
            values in the range [0, 1] (up to numerical precision).
    
        Raises
        ------
        ValueError
            If an invalid method is specified, or if local normalization is
            applied to data with fewer than 2 dimensions.
        """
        methods = ['global', 'local']
        if method not in methods:
            raise ValueError(f"Valid methods are: {methods}")
    
        arr = self.array
    
        if method == 'global':
            # Global normalization: single min/max over entire dataset
            min_val = np.min(arr)
            max_val = np.max(arr)
            denom = max_val - min_val
    
            if denom == 0:
                # Constant array: return zeros
                normalized_tensor = np.zeros_like(arr, dtype=np.float32)
            else:
                normalized_tensor = (arr - min_val) / denom
    
        elif method == 'local':
            # Local normalization: per-pattern min/max on the last two axes
            if arr.ndim < 2:
                raise ValueError("Local normalization requires at least 2 dimensions")
    
            axes = tuple(range(arr.ndim - 2, arr.ndim))  # last two axes
    
            min_val = np.min(arr, axis=axes, keepdims=True)
            max_val = np.max(arr, axis=axes, keepdims=True)
            denom = max_val - min_val
    
            # Avoid division by zero: where denom == 0, set denom to 1 (pattern is constant)
            denom_safe = np.where(denom == 0, 1, denom)
    
            normalized_tensor = (arr - min_val) / denom_safe
    
        return self._spawn(normalized_tensor, preserve_unfold=True)

    
    def _resolve_clip_mask(self, mask):
        """
        Expand a Boolean clipping mask to the complete data shape.

        For data with diffraction dimensions on the last two axes, a mask
        matching the leading dimensions selects complete diffraction patterns,
        while a mask matching the final two dimensions selects reciprocal-space
        pixels in every pattern. Other masks must broadcast directly to the
        complete array shape.
        """
        if hasattr(mask, 'array'):
            mask = mask.array
        mask = np.asarray(mask)

        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("mask must contain Boolean values.")
        if mask.shape == self.shape:
            return mask

        candidates = []
        if self.ndim >= 3:
            leading_shape = self.shape[:-2]
            reciprocal_shape = self.shape[-2:]

            if mask.shape == leading_shape:
                expanded = mask.reshape(leading_shape + (1, 1))
                candidates.append(
                    ('leading', np.broadcast_to(expanded, self.shape))
                )
            if mask.shape == reciprocal_shape:
                expanded = mask.reshape(
                    (1,) * (self.ndim - 2) + reciprocal_shape
                )
                candidates.append(
                    ('reciprocal', np.broadcast_to(expanded, self.shape))
                )

        if len(candidates) > 1:
            raise ValueError(
                f"mask shape {mask.shape} matches both the leading and "
                "reciprocal dimensions. Make the intended domain explicit "
                "with singleton axes, for example mask[..., None, None] for "
                "scan positions or mask[None, None, ...] for reciprocal "
                "pixels in 4D data."
            )
        if candidates:
            return candidates[0][1]

        try:
            return np.broadcast_to(mask, self.shape)
        except ValueError as exc:
            raise ValueError(
                f"mask with shape {mask.shape} cannot be applied to data with "
                f"shape {self.shape}. Provide a full-shape mask, a mask for "
                "the leading scan/stack dimensions, a reciprocal-space mask, "
                "or an explicitly broadcastable Boolean mask."
            ) from exc

    def clip(self, a_min=1, a_max=None, mask=None):
        """
        Clip all or selected data values to the interval [a_min, a_max].
    
        Values smaller than `a_min` are set to `a_min`; values larger than `a_max`
        are set to `a_max`. If `a_min` or `a_max` is None, clipping on that side
        is skipped. When ``mask`` is provided, values outside the mask are
        copied without modification.
    
        Parameters
        ----------
        a_min : float or None, optional
            Lower bound for clipping. If None, no lower clipping is applied.
            Default is 1.
        a_max : float or None, optional
            Upper bound for clipping. If None, no upper clipping is applied.
            Default is None.
        mask : array-like of bool or data object, optional
            Region where clipping is applied. A full-shape mask selects
            individual values. For 4D data, a ``(Ry, Rx)`` mask selects whole
            diffraction patterns and a ``(Ky, Kx)`` mask selects reciprocal
            pixels across every pattern. For 3D data, a ``(N,)`` mask selects
            complete patterns. Explicitly broadcastable masks are also
            accepted. If omitted, the complete array is clipped.
    
        Returns
        -------
        HyperData
            New HyperData instance with clipped data and preserved metadata.

        Examples
        --------
        Clip the complete dataset:

        >>> clipped = data.clip(a_min=0, a_max=100)

        Clip only selected scan positions in a 4D dataset:

        >>> scan_mask = np.zeros(data.shape[:2], dtype=bool)
        >>> scan_mask[10:20, 15:25] = True
        >>> clipped = data.clip(a_min=1, mask=scan_mask)

        Clip selected reciprocal-space pixels in every diffraction pattern:

        >>> k_mask = np.zeros(data.shape[-2:], dtype=bool)
        >>> k_mask[30:90, 30:90] = True
        >>> clipped = data.clip(a_max=500, mask=k_mask)

        Notes
        -----
        This method does not mutate ``self.array``.
        """
        if mask is None:
            return self._spawn(
                clip_values(self.array, a_min, a_max), preserve_unfold=True,
            )

        resolved_mask = self._resolve_clip_mask(mask)
        selected_values = clip_values(
            self.array[resolved_mask],
            a_min,
            a_max,
        )
        output_dtype = np.result_type(
            self.array.dtype,
            selected_values.dtype,
        )
        clipped = self.array.astype(output_dtype, copy=True)
        clipped[resolved_mask] = selected_values
        return self._spawn(clipped, preserve_unfold=True)

    def block_direct_beam(self, radius=None, center=None, beam_units='auto',
                          fill_value=1, return_mask=False):
        """
        Block the central/direct beam in every diffraction pattern.

        This method masks a circular disk on the last two reciprocal-space
        axes and replaces the selected direct-beam pixels with ``fill_value``.
        It supports 3D stacks ``(N, Ky, Kx)`` and 4D-STEM datasets
        ``(Ry, Rx, Ky, Kx)``. The operation never mutates ``self.array``.

        Parameters
        ----------
        radius : float or None, optional
            Direct-beam radius. Explicit values take precedence over stored
            ``center_beam_metadata``. If omitted, the method uses
            ``center_beam_metadata['radius_px']`` from alignment.
        center : array-like of two floats or None, optional
            Direct-beam center as ``(ky, kx)``. Explicit values take precedence
            over metadata. If omitted, the method uses metadata when available
            and otherwise falls back to the geometric diffraction-pattern
            center.
        beam_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system for explicit ``radius`` and ``center``. ``'auto'``
            uses reciprocal-space calibration when available and otherwise
            falls back to pixels. Calibrated centers are reciprocal coordinates
            relative to the diffraction origin, with positive ``ky`` upward
            and positive ``kx`` to the right.
        fill_value : scalar, optional
            Value assigned to blocked direct-beam pixels. Defaults to 1.
        return_mask : bool, optional
            If True, return ``(blocked_data, beam_mask)`` where
            ``beam_mask`` is True for blocked pixels.

        Returns
        -------
        HyperData or tuple
            New HyperData object with direct-beam pixels blocked, optionally
            followed by the 2D beam mask.

        Notes
        -----
        ``block_direct_beam`` is the preferred name. ``remove_center_beam`` is
        kept as an alias for older notebook wording.
        """
        if self.ndim not in (3, 4):
            raise ValueError(
                "block_direct_beam requires a 3D stack of diffraction "
                "patterns or a 4D-STEM dataset."
            )
        if self.is_polar:
            raise ValueError(
                "block_direct_beam currently expects Cartesian reciprocal "
                "axes. Apply it before to_polar(), or use an explicit polar "
                "mask on polar data."
            )
        if not np.isscalar(fill_value):
            raise ValueError("fill_value must be a scalar.")

        ky, kx = self.shape[-2:]
        center_px, radius_px, resolved_metadata = _resolve_center_beam_parameters(
            (ky, kx),
            metadata=self.center_beam_metadata,
            radius=radius,
            center=center,
            units=self.reciprocal_units,
            conv_factor=self.reciprocal_conv_factor,
            beam_units=beam_units,
        )
        beam_mask = make_mask(center_px, radius_px, mask_dim=(ky, kx))
        if not np.any(beam_mask):
            raise ValueError(
                "The resolved direct-beam mask selects no pixels. Check "
                "radius, center, and beam_units."
            )

        output_dtype = np.result_type(self.array.dtype, np.asarray(fill_value).dtype)
        blocked = self.array.astype(output_dtype, copy=True)
        blocked[..., beam_mask] = fill_value

        result = self._spawn(blocked, preserve_unfold=True)
        resolved_metadata.update({
            'source': 'block_direct_beam',
            'fill_value': fill_value,
            'previous_metadata': deepcopy(self.center_beam_metadata),
        })
        result.center_beam_metadata = resolved_metadata

        if return_mask:
            return result, beam_mask
        return result

    remove_center_beam = block_direct_beam

    
    def crop(self, ylim=None, xlim=None, kylim=None, kxlim=None,
             kshape=None, rshape=None, real_limit_units='auto',
             reciprocal_limit_units='auto'):
        """
        Crop a 4D dataset either in the real or the reciprocal domain,
        with optional subpixel cropping and enforced resizing in both
        k-space (kshape) and real space (rshape).
    
        Parameters
        ----------
        ylim : int or tuple, optional
            Real-space vertical limits. Interpreted according to
            ``real_limit_units``.
        xlim : int or tuple, optional
            Real-space horizontal limits. Interpreted according to
            ``real_limit_units``.
        kylim : int or tuple of (float or int), optional
            Reciprocal-space vertical limits. Interpreted according to
            ``reciprocal_limit_units``. Pixel-space values can be floats for
            subpixel cropping. Calibrated reciprocal limits use coordinates
            relative to the diffraction origin, with positive ky upward.
        kxlim : int or tuple of (float or int), optional
            Reciprocal-space horizontal limits. Interpreted according to
            ``reciprocal_limit_units``. Pixel-space values can be floats for
            subpixel cropping. Calibrated reciprocal limits use coordinates
            relative to the diffraction origin, with positive kx to the right.
        kshape : tuple of (int, int), optional
            Shape (A, B) to resize cropped diffraction patterns to.
            If not provided, it is inferred from the cropped reciprocal region
            when subpixel cropping or explicit k-limits are used.
            Interpolating integer data returns floating-point output so detector
            counts retain their original range and subpixel values.
        rshape : tuple of (int, int), optional
            Shape (Ny_out, Nx_out) for real-space resizing (generalized binning).
            When ``rshape`` divides the current real-space shape, block
            averaging is used. Otherwise, the scan axes are interpolated.
            Resampling integer data produces floating-point output so fractional
            counts are not truncated.
        real_limit_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for ``ylim`` and ``xlim``. ``'auto'`` uses stored
            real-space calibration when available and otherwise falls back to
            pixels. Calibrated limits account for the image's real-space origin.
        reciprocal_limit_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for ``kylim`` and ``kxlim``. ``'auto'`` uses
            stored reciprocal-space calibration when available and otherwise
            falls back to pixels.

        Examples
        --------
        Crop with pixel limits:

        >>> cropped = data.crop(
        ...     ylim=(10, 40),
        ...     xlim=(20, 60),
        ...     kylim=(30, 90),
        ...     kxlim=(28, 92),
        ...     real_limit_units='pixels',
        ...     reciprocal_limit_units='pixels',
        ... )

        Crop using stored calibrated units:

        >>> cropped = data.crop(
        ...     ylim=(0.0, 5.0),
        ...     xlim=(0.0, 5.0),
        ...     kylim=(-0.6, 0.6),
        ...     kxlim=(-0.6, 0.6),
        ...     real_limit_units='calibrated',
        ...     reciprocal_limit_units='calibrated',
        ... )
        """

        if self.ndim != 4:
            raise ValueError("crop currently requires a 4D HyperData object.")

        _, real_factor, real_mode = _resolve_unit_mode(
            real_limit_units,
            self.real_units,
            self.real_conv_factor,
            label='real_limit_units',
        )
        _, reciprocal_factor, reciprocal_mode = _resolve_unit_mode(
            reciprocal_limit_units,
            self.reciprocal_units,
            self.reciprocal_conv_factor,
            label='reciprocal_limit_units',
        )
    
        def parse_limits(limits, max_length, allow_float=False, name="limits",
                         unit_mode='pixels', conv_factor=None, axis='real',
                         origin=0.0):
            """Return (start, end) within [0, max_length]."""
            if limits is None:
                return (0, max_length)

            def calibrated_to_pixel(value):
                if unit_mode != 'calibrated':
                    return float(value)
                if axis == 'real':
                    return (float(value) - origin) / conv_factor
                axis_center = (max_length - 1) / 2.0
                if axis == 'reciprocal_y':
                    return axis_center - float(value) / conv_factor
                if axis == 'reciprocal_x':
                    return axis_center + float(value) / conv_factor
                raise ValueError(f"Unknown crop axis {axis!r}.")
    
            # Single index
            if np.isscalar(limits):
                value = float(limits)
                if not np.isfinite(value):
                    raise ValueError(f"{name}: index must be finite.")
                if unit_mode == 'calibrated':
                    idx = int(np.rint(calibrated_to_pixel(value)))
                else:
                    if not value.is_integer():
                        raise ValueError(
                            f"{name}: single-index limits must be integer "
                            "when using pixel units."
                        )
                    idx = int(value)
                if idx < 0 or idx >= max_length:
                    raise ValueError(f"{name}: index {idx} out of bounds for axis length {max_length}")
                return (idx, idx + 1)
    
            # Two-element sequence
            is_pair = (
                isinstance(limits, (tuple, list, np.ndarray))
                and np.asarray(limits).shape == (2,)
            )
            if is_pair:
                start, end = np.asarray(limits, dtype=float)
                if not np.all(np.isfinite((start, end))):
                    raise ValueError(f"{name}: range limits must be finite.")
    
                if unit_mode == 'calibrated':
                    start_px = calibrated_to_pixel(start)
                    end_px = calibrated_to_pixel(end)
                    if axis.startswith('reciprocal_'):
                        start, end = sorted((start_px, end_px))
                    else:
                        start, end = start_px, end_px
                    if not allow_float:
                        start = int(np.floor(start))
                        end = int(np.ceil(end))
                elif not allow_float:
                    # Real-space indices must be integers
                    if not float(start).is_integer() or not float(end).is_integer():
                        raise ValueError(f"{name}: real-space limits must be integers")
                    start = int(start)
                    end = int(end)
    
                if start < 0 or end > max_length:
                    raise ValueError(
                        f"{name}: range ({start}, {end}) out of bounds for axis length {max_length}"
                    )
                if end <= start:
                    raise ValueError(f"{name}: end ({end}) must be greater than start ({start})")
                if allow_float and np.isclose(start, np.rint(start)) and np.isclose(end, np.rint(end)):
                    start = int(np.rint(start))
                    end = int(np.rint(end))
                return (start, end)
    
            raise ValueError(f"{name}: limits must be int, tuple, list, or None")
    
        Ny, Nx, Ky, Kx = self.shape
    
        # --- Real-space limits ---
        ylim_range = _parse_real_selection(
            ylim,
            Ny,
            'ylim',
            real_mode,
            _real_spacing_pair(real_factor)[0],
            self.real_origin[0],
        )[:2]
        xlim_range = _parse_real_selection(
            xlim,
            Nx,
            'xlim',
            real_mode,
            _real_spacing_pair(real_factor)[1],
            self.real_origin[1],
        )[:2]
    
        # Fast path: only real-space crop, no k-space crop or resizing, no r-resize
        if kylim is None and kxlim is None and kshape is None and rshape is None:
            y0r, y1r = ylim_range
            x0r, x1r = xlim_range
            real_step = _real_spacing_pair(self.real_conv_factor)
            new_origin = (
                self.real_origin[0] + y0r * real_step[0],
                self.real_origin[1] + x0r * real_step[1],
            )
            return self._spawn(
                self.array[y0r:y1r, x0r:x1r], real_origin=new_origin,
            )
    
        # --- Reciprocal-space limits (allow floats for subpixel cropping) ---
        kylim_range = parse_limits(
            kylim,
            Ky,
            allow_float=True,
            name="kylim",
            unit_mode=reciprocal_mode,
            conv_factor=reciprocal_factor,
            axis='reciprocal_y',
        )
        kxlim_range = parse_limits(
            kxlim,
            Kx,
            allow_float=True,
            name="kxlim",
            unit_mode=reciprocal_mode,
            conv_factor=reciprocal_factor,
            axis='reciprocal_x',
        )
    
        # Integer boundaries for actual array slicing (pad around subpixel ROI)
        y0 = int(np.floor(kylim_range[0]))
        y1 = int(np.ceil(kylim_range[1]))
        x0 = int(np.floor(kxlim_range[0]))
        x1 = int(np.ceil(kxlim_range[1]))
    
        if y0 < 0 or y1 > Ky or x0 < 0 or x1 > Kx:
            raise ValueError(
                f"Reciprocal-space crop out of bounds: "
                f"y in [{y0}, {y1}), x in [{x0}, {x1}), shape ({Ky}, {Kx})"
            )
    
        # Determine whether subpixel cropping is needed (any float in k-limits)
        def is_int_like(v):
            return isinstance(v, (Integral, np.integer))
    
        subpixel = not (
            is_int_like(kylim_range[0])
            and is_int_like(kylim_range[1])
            and is_int_like(kxlim_range[0])
            and is_int_like(kxlim_range[1])
        )
    
        # Natural reciprocal-space shape from the slice
        natural_kshape = (y1 - y0, x1 - x0)
    
        # Normalize kshape (if provided)
        if kshape is not None:
            if len(kshape) != 2:
                raise ValueError("kshape must be a tuple (A, B)")
            kshape = (int(kshape[0]), int(kshape[1]))
            if kshape[0] <= 0 or kshape[1] <= 0:
                raise ValueError("kshape must contain positive integers")
        else:
            # If user provided k-limits or we are doing subpixel cropping,
            # default kshape to the natural cropped size
            if kylim is not None or kxlim is not None or subpixel:
                kshape = natural_kshape
    
        if kshape is not None and kshape[0] != kshape[1]:
            print(f"Warning: Non-square diffraction pattern shape {kshape}. Proceeding anyway.")
    
        # --- Crop in real space first ---
        y0r, y1r = ylim_range
        x0r, x1r = xlim_range
        subarray = self.array[y0r:y1r, x0r:x1r]
    
        # --- Crop reciprocal region ---
        cropped = subarray[:, :, y0:y1, x0:x1]
    
        # --- k-space resize (subpixel or enforced kshape) ---
        needs_k_resize = subpixel or (kshape is not None and kshape != natural_kshape)
    
        if needs_k_resize:
            # If kshape is None here, fall back to natural_kshape
            if kshape is None:
                kshape = natural_kshape

            resize_dtype = np.result_type(subarray.dtype, np.float32)
            new_data = np.empty(
                (subarray.shape[0], subarray.shape[1], kshape[0], kshape[1]),
                dtype=resize_dtype,
            )
    
            for i in range(subarray.shape[0]):
                for j in range(subarray.shape[1]):
                    dp = cropped[i, j]
                    new_data[i, j] = transform.resize(
                        dp.astype(resize_dtype, copy=False),
                        kshape,
                        order=1,          # bilinear
                        mode="reflect",
                        anti_aliasing=True,
                        preserve_range=True,
                    )
    
            cropped = new_data
    
        # --- Real-space resize (rshape) ---
        if rshape is not None:
            if len(rshape) != 2:
                raise ValueError("rshape must be a tuple (Ny_out, Nx_out)")
    
            Ny_out, Nx_out = int(rshape[0]), int(rshape[1])
            if Ny_out <= 0 or Nx_out <= 0:
                raise ValueError("rshape must contain positive integers")
    
            Ny_c, Nx_c, Ky_c, Kx_c = cropped.shape
    
            # If already at requested size, nothing to do
            if (Ny_out, Nx_out) != (Ny_c, Nx_c):
    
                # Case 1: exact divisors -> block-averaging / binning
                if Ny_c % Ny_out == 0 and Nx_c % Nx_out == 0:
                    by = Ny_c // Ny_out
                    bx = Nx_c // Nx_out
                    # Reshape and average over bin axes
                    reshaped = cropped.reshape(
                        Ny_out, by,
                        Nx_out, bx,
                        Ky_c, Kx_c
                    )
                    # Average over the binning dimensions (1 and 3)
                    binned = reshaped.mean(axis=(1, 3))
                    cropped = binned
    
                else:
                    # Case 2: non-divisors -> interpolation along real-space axes
                    # Treat each diffraction pattern (Ky*Kx) as "channels"
                    channels = Ky_c * Kx_c
                    tmp = cropped.reshape(Ny_c, Nx_c, channels)
    
                    resized = transform.resize(
                        tmp.astype(np.result_type(tmp.dtype, np.float32), copy=False),
                        (Ny_out, Nx_out, channels),
                        order=1,          # bilinear in real space
                        mode="reflect",
                        anti_aliasing=True,
                        preserve_range=True,
                    )
                    cropped = resized.reshape(Ny_out, Nx_out, Ky_c, Kx_c)

        new_real_conv = self.real_conv_factor
        new_reciprocal_conv = self.reciprocal_conv_factor
        real_step = _real_spacing_pair(self.real_conv_factor)
        new_real_origin = (
            self.real_origin[0] + y0r * real_step[0],
            self.real_origin[1] + x0r * real_step[1],
        )

        if kshape is not None and self.reciprocal_conv_factor is not None:
            y_scale = (kylim_range[1] - kylim_range[0]) / kshape[0]
            x_scale = (kxlim_range[1] - kxlim_range[0]) / kshape[1]
            if np.isclose(y_scale, x_scale):
                new_reciprocal_conv = self.reciprocal_conv_factor * x_scale
            else:
                print("Warning: anisotropic reciprocal-space resizing cleared the stored reciprocal-space calibration.")
                new_reciprocal_conv = None

        if rshape is not None and self.real_conv_factor is not None:
            y_scale = (ylim_range[1] - ylim_range[0]) / rshape[0]
            x_scale = (xlim_range[1] - xlim_range[0]) / rshape[1]
            new_real_conv = _scaled_real_spacing(
                self.real_conv_factor, (y_scale, x_scale),
            )
            new_real_origin = (
                new_real_origin[0] + 0.5 * (y_scale - 1) * real_step[0],
                new_real_origin[1] + 0.5 * (x_scale - 1) * real_step[1],
            )

        new_real_units = self.real_units if new_real_conv is not None else None
        new_reciprocal_units = (
            self.reciprocal_units if new_reciprocal_conv is not None else None
        )

        k_geometry_changed = (
            (y0, y1, x0, x1) != (0, Ky, 0, Kx) or needs_k_resize
        )
        beam_metadata = self.center_beam_metadata
        if k_geometry_changed:
            beam_metadata = None if subpixel or self.polar_metadata is not None else (
                _resampled_center_beam_metadata(
                    self.center_beam_metadata, (Ky, Kx), cropped.shape[-2:],
                    crop_origin=(y0, x0), crop_shape=natural_kshape,
                    units=new_reciprocal_units,
                    conv_factor=new_reciprocal_conv,
                )
            )

        return self._spawn(
            cropped,
            real_units=new_real_units,
            real_conv_factor=new_real_conv,
            real_origin=new_real_origin,
            reciprocal_units=new_reciprocal_units,
            reciprocal_conv_factor=new_reciprocal_conv,
            center_beam_metadata=beam_metadata,
        )

    # Private method: helper function for the 'alignment' method
    def _quickCOM(self, r_mask=5, center=None):
        """Return masked (ky, kx) centers of mass for every diffraction pattern.

        A circular radius or ``(inner, outer)`` ring is accepted. Empty or
        zero-intensity regions retain the historical center value of zero.
        """

        y, x, ky, kx = np.shape(self.array)
        if center is None:
            center_y, center_x = (ky - 1) / 2, (kx - 1) / 2
        else:
            center_y, center_x = tuple(float(v) for v in center)

        if type(r_mask) == tuple:
            inner_mask = make_mask((center_y, center_x), r_mask[0], mask_dim=(ky, kx), invert=True)
            outer_mask = make_mask((center_y, center_x), r_mask[1], mask_dim=(ky, kx))
            mask = np.logical_and(inner_mask, outer_mask)
            
        else:
            mask = make_mask((center_y, center_x), r_mask, mask_dim=(ky, kx))
        
        ap2_y = np.zeros((y, x), dtype=float)
        ap2_x = np.zeros_like(ap2_y)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return ap2_y, ap2_x

        y0, y1 = rows.min(), rows.max() + 1
        x0, x1 = cols.min(), cols.max() + 1
        region = self.array[:, :, y0:y1, x0:x1]
        local_mask = mask[y0:y1, x0:x1]
        mass = np.einsum('ijab,ab->ij', region, local_mask, dtype=float)
        y_moment = np.einsum(
            'ijab,ab->ij', region,
            local_mask * np.arange(y0, y1)[:, None], dtype=float,
        )
        x_moment = np.einsum(
            'ijab,ab->ij', region,
            local_mask * np.arange(x0, x1)[None, :], dtype=float,
        )
        np.divide(y_moment, mass, out=ap2_y, where=mass != 0)
        np.divide(x_moment, mass, out=ap2_x, where=mass != 0)
        return ap2_y, ap2_x
    
    def fix_elliptical_distortions(self, r=None, R=None, interp_method='linear', 
                                   show_ellipse=False, return_fix=True, **kwargs):
        """
        Corrects elliptical distortions across all diffraction patterns in the dataset.
    
        This function computes the average diffraction pattern, fits an ellipse to this average,
        and applies an affine transformation to each diffraction pattern (by transforming the 
        ellipse into a circle) to correct the elliptical distortion.
        
        Parameters
        ----------
        r : int, optional
            Inner radius of the annular mask. Defaults to one fifth of the diffraction pattern's horizontal dimension if not specified.
        R : int, optional
            Outer radius of the annular mask. Defaults to two fifths of the diffraction pattern's horizontal dimension if not specified.
        interp_method : str, optional
            Interpolation method to use during affine transformations. Can be 'linear' or 'cubic'.
        show_ellipse : bool, optional
            Plots the mean diffraction pattern in the dataset with an overlaid ring of inner radius r and outer radius R. If return_fix 
            is False, it plots the ring in the non-corrected pattern. Otherwise, it plots it in the corrected pattern.
        
        Returns
        -------
        HyperData
            A new instance of HyperData containing the corrected data.
    
        Notes
        -----
        This method must be applied after applying the 'alignment' method for better accuracy.
        """
        
        A, B, C, D = self.shape
        
        if not r:
            r = C//5 
        if not R:
            R = 2*C//5
        
        mean_pattern = self.get_dp('mean').array
        params = self._extract_ellipse(mean_pattern, C//2, D//2, r, R)
        corrected_data = np.empty_like(self.array)
        
        Ang, a, b = params 
        print("Ellptical statistics before correction:")
        print(f"Ellipse rotation = {np.degrees(Ang)} degrees \nMajor axis 'a' = {a} px \nMinor axis 'b' = {b} px \n")
        
        if show_ellipse:
            self._plot_with_ring(np.log(mean_pattern), r, R, **kwargs)
        
        # Perform Correction
        for i in tqdm(range(A), desc="Fixing elliptical distortions"):
            for j in range(B):
                corrected_data[i, j] = self._apply_affine_transformation(self.array[i, j], *params, interp_method)
        
        corrected_data = HyperData(corrected_data)
        
        mean_pattern = corrected_data.get_dp('mean').array
        params = corrected_data._extract_ellipse(mean_pattern, C//2, D//2, r, R)
        
        Ang, a, b = params 
        print("Ellptical statistics after correction:")
        print(f"Ellipse rotation = {np.degrees(Ang)} degrees \nMajor axis 'a' = {a} px \nMinor axis 'b' = {b} px \n")
        
        if show_ellipse:
            self._plot_with_ring(np.log(mean_pattern), r, R, **kwargs)
        
        return corrected_data

    #TODO: verify that the cost function is good
    def _extract_ellipse(self, image, cy, cx, r, R):
        """
        Fit an ellipse to an annular region of an image (diffraction pattern).
    
        The function calculates parameters for the ellipse that best fits the specified annular region
        of the mean diffraction pattern. These parameters are optimized to minimize the defined cost function.
    
        Parameters
        ----------
        image : numpy.ndarray
            The mean diffraction pattern as a 2D numpy array.
        cy : int
            The y-coordinate of the center of the diffraction pattern.
        cx : int
            The x-coordinate of the center of the diffraction pattern.
        r : int
            Inner radius for the annular mask.
        R : int
            Outer radius for the annular mask.
    
        Returns
        -------
        tuple
            A tuple of fitted ellipse parameters (A, a, b), where A is the rotation angle,
            a is the length of the major axis, and b is the length of the minor axis.
        """
        y, x = np.indices(image.shape)  # Get array indices for the entire image
        mask = ((x - cx)**2 + (y - cy)**2 >= r**2) & ((x - cx)**2 + (y - cy)**2 <= R**2)
        x_vals, y_vals = x[mask], y[mask]  # Get the coordinates within the mask
        intensities = image[mask]  # Get the intensities at these coordinates

        def ellipse_cost(params):
            A, a, b = params
            cost = intensities**4 * (((x_vals - cx) * np.cos(A) + (y_vals - cy) * np.sin(A))**2 / a**2 +
                                  ((y_vals - cy) * np.cos(A) - (x_vals - cx) * np.sin(A))**2 / b**2 - 1)**2
            return np.sum(cost)

        initial_guess = [0, (R-r)/2 + 1, (R-r)/2 - 1]  # Initial guess for A, a, b
        result = minimize(ellipse_cost, initial_guess, method='L-BFGS-B')
        return result.x


    def _apply_affine_transformation(self, image, A, a, b, interp_method):
        """
        Apply an affine transformation to make the fitted ellipse a circle.

        Parameters
        ----------
        image : numpy.ndarray
            The 2D diffraction pattern to correct.
        A : float
            The angle of rotation of the ellipse.
        a : float
            The length of the major axis.
        b : float
            The length of the minor axis.

        Returns
        -------
        numpy.ndarray
            The corrected diffraction pattern.
        """
        # NumPy shape is (rows, cols); OpenCV points/sizes use (x, y)/(width, height).
        rows, cols = image.shape
        
        # Points in the original image
        p1 = np.float32([
            [a * np.cos(A) + cols//2, a * np.sin(A) + rows//2],
            [-a * np.cos(A) + cols//2, -a * np.sin(A) + rows//2],
            [b * np.sin(A) + cols//2, -b * np.cos(A) + rows//2]
        ])
        
        # Depending on whether a or b is larger, set the corresponding points in the corrected image
        # This ensures that the affine transformation results in expansion rather than shrinkage of the
        # transformed diffraction pattern.
        if a < b:
            p2 = np.float32([
                [b * np.cos(A) + cols//2, b * np.sin(A) + rows//2],
                [-b * np.cos(A) + cols//2, -b * np.sin(A) + rows//2],
                [b * np.sin(A) + cols//2, -b * np.cos(A) + rows//2]
            ])
        else:
            p2 = np.float32([
                [a * np.cos(A) + cols//2, a * np.sin(A) + rows//2],
                [-a * np.cos(A) + cols//2, -a * np.sin(A) + rows//2],
                [a * np.sin(A) + cols//2, -a * np.cos(A) + rows//2]
            ])
            
        M = cv2.getAffineTransform(p1, p2)
        
        if interp_method == 'cubic':
            transformed_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_CUBIC)
            
        if interp_method == 'linear':
            transformed_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

        return transformed_image
    
    
    def _plot_with_ring(self, data, inner_radius, outer_radius, **kwargs):
        """
        Plots a 2D array with an overlaid red ring directly on the same figure.
        
        Parameters
        ----------
        data : numpy.ndarray
            The 2D array to plot.
        inner_radius : float
            The inner radius of the ring.
        outer_radius : float
            The outer radius of the ring.
        """
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Generate a grid of points
        y, x = np.indices(data.shape)
        center_x, center_y = np.array(data.shape) // 2
        
        # Calculate the radius for each point in the grid
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create a mask for the ring
        ring_mask = (radius >= inner_radius) & (radius <= outer_radius)
        
        # Create a colored overlay where the ring is red
        ring_overlay = np.zeros(data.shape + (4,), dtype=np.float32)  # Adding a new dimension for RGBA
        ring_overlay[..., 0] = 1  # Red channel
        ring_overlay[..., 3] = ring_mask * 1.0  # Alpha channel only set where the ring is
        
        # Display the image
        ax.imshow(data, **kwargs)
        ax.imshow(ring_overlay, cmap='hot', alpha=0.5) 
        
        # Set plot details
        ax.set_title('Mean diffraciton with annular overlay')
        ax.axis('off')
        
        plt.show()
    
    def centerBeam_Stats(self, square_side=8):
        """
        Analyze diffraction patterns to find the mean and standard deviation of 
        the center of mass.
    
        Parameters:
        data (numpy.ndarray): 4D dataset of shape (A, B, C, D).
        square_side (int): Side length of the square used to calculate the 
        center of mass.
    
        Returns:
        tuple: Mean and standard deviation of the center of mass coordinates.
        """
        A, B, C, D = self.shape
        com_coordinates = []
    
        # Define the region for center of mass calculation
        half_side = square_side // 2
        center = C//2
        min_coord, max_coord = center - half_side, center + half_side
    
        # Calculate center of mass for each BxB image
        for i in range(A):
            for j in range(B):
                region = self.array[i, j, min_coord:max_coord, min_coord:max_coord]
                com = center_of_mass(region)
                com_coordinates.append((com[0] + min_coord, com[1] + min_coord))
    
        # Convert to numpy array for ease of calculation
        com_coordinates = np.array(com_coordinates)
    
        # Calculate mean and standard deviation
        mean_com = np.mean(com_coordinates, axis=0)
        std_dev_com = np.std(com_coordinates, axis=0)
    
        print(f"Mean CoM Coordinate (ky, kx): {mean_com}")
        print(f"Standard Deviation of CoM Coordinates (ky, kx): {std_dev_com} px")
    
        return mean_com, std_dev_com
    
    #TODO: Is there a faster way to get this?
    def get_ewpc(self):
        """
        Transform dataset to Exit Wave Power Cepstrum (EWPC)
        """
    
        return self._spawn(np.abs(fftshift(
                                       fft2(
                                            np.log(self.clip().array),
                                            axes=(-2, -1)
                                            ),
                                       axes=(-2, -1)
                                      )), center_beam_metadata=None)
    
    
    def get_stdDev(self, domain='reciprocal'):
        """
        Calculate a 2D standard-deviation image over the other coordinate grid.

        Parameters:
        domain (string): the domain for which the standard deviation will be
        computer for each position.

        Returns
        -------
        RealSpace or ReciprocalSpace
            Calibrated 2D image for the requested domain.
        """
        # Validate the shape of the data
        if len(self.array.shape) != 4:
            raise ValueError("Data must be a 4D array")
                
        if domain == 'reciprocal':
            # Calculate the standard deviation for each pixel across all diffraction patterns
            std_dev = np.std(self.array, axis=(0, 1))
            return self._spawn_reciprocal(std_dev)
        elif domain == 'real':
            # Calculate the standard deviation for each pixel across all scanning positions
            std_dev = np.std(self.array, axis=(2, 3))
            return self._spawn_real(std_dev, quantity='Standard deviation')
        else:
            raise ValueError("'domain' must be 'reciprocal' or 'real'")
    
    def visualize(self, grid_shape=None, padding=2, reduction='mean',
                  power=1, title='Diffraction-Pattern Montage',
                  log_scale=True, axes=True, vmin=None, vmax=None,
                  figsize=None, aspect=None, cmap='turbo',
                  axis_units='auto'):
        """
        Display a reduced 4D dataset as a two-dimensional diffraction montage.

        Neighboring scan positions are combined into ``grid_shape`` real-space
        bins. Each resulting diffraction pattern is placed at the corresponding
        scan position in one image, separated by white padding. A shared
        intensity scale is used so brightness remains comparable across tiles.

        Parameters
        ----------
        grid_shape : tuple of int or None, optional
            Number of diffraction patterns to display as ``(Ny, Nx)``. Values
            cannot exceed the corresponding real-space dimensions. By default,
            each dimension is 10 percent of its original size, rounded down,
            with a minimum of 1 and a maximum of 25.
        padding : int, optional
            Number of white display pixels between adjacent patterns.
        reduction : {'mean', 'sum'}, optional
            How scan positions contributing to each displayed tile are
            combined. ``'mean'`` uses area-weighted averaging. ``'sum'``
            approximates an integrated diffraction pattern for each bin.
        power : float, optional
            Display parameter matching :meth:`ReciprocalSpace.show`. With
            logarithmic scaling the values are ``power * log(intensity)``;
            otherwise they are ``intensity ** power``.
        title : str, optional
            Figure title.
        log_scale : bool, optional
            Apply logarithmic intensity scaling.
        axes : bool, optional
            Show real-space scan-position axes and a shared colorbar.
        vmin, vmax : float or None, optional
            Shared display limits after the intensity transformation.
        figsize : tuple or None, optional
            Matplotlib figure size. It is inferred from the montage if omitted.
        aspect : float, str, or None, optional
            Aspect ratio forwarded to the montage axes.
        cmap : str, optional
            Matplotlib colormap used for every diffraction pattern.
        axis_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for the displayed real-space scan axes and the
            per-tile reciprocal-space calibration note. ``'auto'`` uses stored
            calibration when available and otherwise falls back to pixels.

        Returns
        -------
        tuple
            ``(fig, ax)`` for further customization or saving.

        Notes
        -----
        The outer axes describe real-space scan position. Reciprocal-space axes
        repeat inside every tile and cannot be represented by one continuous
        montage axis, so their calibration is reported in the title.
        """
        if self.ndim != 4:
            raise ValueError(
                "visualize requires 4D data with shape (Ry, Rx, Ky, Kx)."
            )

        ry, rx, ky, kx = self.shape
        if grid_shape is None:
            grid_shape = (
                min(25, max(1, int(np.floor(0.1 * ry)))),
                min(25, max(1, int(np.floor(0.1 * rx)))),
            )
        elif isinstance(grid_shape, np.ndarray):
            grid_shape = grid_shape.tolist()

        grid_shape = self._normalize_resize_shape(
            grid_shape, 2, 'grid_shape'
        )
        ny, nx = grid_shape
        if ny > ry or nx > rx:
            raise ValueError(
                "grid_shape cannot exceed the real-space shape "
                f"{(ry, rx)}; received {grid_shape}."
            )

        if not isinstance(padding, (Integral, np.integer)) or padding < 0:
            raise ValueError("padding must be a non-negative integer.")
        padding = int(padding)

        if not isinstance(reduction, str):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        reduction = reduction.lower()
        if reduction not in ('mean', 'sum'):
            raise ValueError("reduction must be 'mean' or 'sum'.")

        if not np.isscalar(power) or not np.isfinite(power):
            raise ValueError("power must be a finite scalar.")
        if not isinstance(log_scale, (bool, np.bool_)):
            raise ValueError("log_scale must be a boolean.")

        reduced = self.resize(
            grid_shape,
            domain='real',
            method='area',
        ).array
        if reduction == 'sum':
            reduced = reduced * ((ry / ny) * (rx / nx))

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if log_scale:
                displayed = power * np.log(reduced)
            else:
                displayed = reduced ** power

        montage_shape = (
            ny * ky + (ny - 1) * padding,
            nx * kx + (nx - 1) * padding,
        )
        montage = np.ma.masked_all(
            montage_shape,
            dtype=np.result_type(displayed.dtype, np.float32),
        )
        for y_idx in range(ny):
            y_start = y_idx * (ky + padding)
            for x_idx in range(nx):
                x_start = x_idx * (kx + padding)
                tile = np.ma.masked_invalid(displayed[y_idx, x_idx])
                montage[
                    y_start:y_start + ky,
                    x_start:x_start + kx,
                ] = tile

        finite_values = montage.compressed()
        if finite_values.size == 0:
            raise ValueError(
                "No finite values remain after applying the display transform."
            )
        if vmin is None:
            vmin = float(np.min(finite_values))
        if vmax is None:
            vmax = float(np.max(finite_values))
        if vmin > vmax:
            raise ValueError("vmin must be less than or equal to vmax.")

        if figsize is None:
            ratio = montage_shape[1] / montage_shape[0]
            if ratio >= 1:
                figsize = (min(16, 10 * ratio), 10)
            else:
                figsize = (10, min(16, 10 / ratio))

        cmap_object = deepcopy(plt.get_cmap(cmap))
        cmap_object.set_bad('white')

        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(
            montage,
            cmap=cmap_object,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            origin='upper',
        )
        if aspect is not None:
            ax.set_aspect(aspect)

        representative_dp = self._spawn_reciprocal(reduced[0, 0])
        if representative_dp.is_polar:
            _, theta_units, radius_units, _, _ = representative_dp._polar_axis_info(
                axis_units=axis_units
            )
            tile_description = (
                f"Each tile: {ky} radial x {kx} angular samples "
                f"({radius_units}, {theta_units})"
            )
        else:
            reciprocal_units, reciprocal_factor, _ = (
                _resolve_unit_mode(
                    axis_units,
                    representative_dp.units,
                    representative_dp.conv_factor,
                    label='axis_units',
                )
            )
            unit_text = representative_dp._format_unit_text(reciprocal_units)
            if reciprocal_factor is None:
                tile_description = f"Each tile: {ky} x {kx} reciprocal pixels"
            else:
                tile_description = (
                    f"Each tile: {ky} x {kx} pixels; "
                    f"{reciprocal_factor:g} {unit_text}/pixel"
                )

        if axes:
            max_ticks = 8

            def _scan_ticks(count, tile_size, source_size):
                indices = np.unique(
                    np.rint(
                        np.linspace(0, count - 1, min(count, max_ticks))
                    ).astype(int)
                )
                positions = (
                    indices * (tile_size + padding)
                    + (tile_size - 1) / 2
                )
                source_centers = (
                    (indices + 0.5) * source_size / count - 0.5
                )
                return positions, source_centers

            x_positions, x_centers = _scan_ticks(nx, kx, rx)
            y_positions, y_centers = _scan_ticks(ny, ky, ry)

            real_units, real_scale, _ = _resolve_unit_mode(
                axis_units,
                self.real_units,
                self.real_conv_factor,
                label='axis_units',
            )
            real_scale = 1.0 if real_scale is None else real_scale
            y_scale, x_scale = _real_spacing_pair(real_scale)
            x_origin = self.real_origin[1] if real_units is not None else 0.0
            y_origin = self.real_origin[0] if real_units is not None else 0.0
            x_labels = x_origin + x_centers * x_scale
            y_labels = y_origin + y_centers * y_scale
            real_units = (
                'scan px' if real_units is None
                else self._spawn_real(
                    np.empty((1, 1))
                )._format_unit_text(real_units)
            )

            def _format_axis_tick(value):
                label = f"{value:.1f}"
                if label in {"-0.0", "+0.0"}:
                    label = "0.0"
                if label.endswith(".0"):
                    return label[:-2]
                return label

            ax.set_xticks(x_positions)
            ax.set_yticks(y_positions)
            ax.set_xticklabels([_format_axis_tick(value) for value in x_labels])
            ax.set_yticklabels([_format_axis_tick(value) for value in y_labels])
            ax.set_xlabel(f"Real-space x ({real_units})", fontsize=14)
            ax.set_ylabel(f"Real-space y ({real_units})", fontsize=14)
            ax.set_title(f"{title}\n{tile_description}", fontsize=16)

            divider = make_axes_locatable(ax)
            colorbar_axis = divider.append_axes("right", size="3%", pad=0.08)
            colorbar = fig.colorbar(image, cax=colorbar_axis)
            if log_scale:
                colorbar_label = (
                    "log(Intensity)"
                    if power == 1
                    else f"log(Intensity) (Power = {power:g})"
                )
            else:
                colorbar_label = (
                    "Intensity"
                    if power == 1
                    else f"Intensity (Power = {power:g})"
                )
            colorbar.set_label(colorbar_label, fontsize=12)
        else:
            ax.set_axis_off()

        fig.tight_layout()
        plt.show()
        return fig, ax

    def get_dp(self, y=None, x=None, mask=None, operation=None,
               selection_units='pixels', **flat_kwargs):
        """
        Obtain a diffraction pattern from a 3D/4D dataset, either at a
        specific point, averaged over a specified region, or via a special operation.
    
        Parameters
        ----------
        y : int, float, tuple, list, ndarray, or None
            If tuple (ymin, ymax), averages over rows [ymin:ymax];
            if scalar, selects a single row; if None, ignored.
        x : int, float, tuple, list, ndarray, or None
            If tuple (xmin, xmax), averages over cols [xmin:xmax];
            if scalar, selects a single column; if None, ignored.
        mask : ndarray[bool] | None
            Shape (A, B). If provided and operation in
            {'mean','median','max','min','std'}, returns aggregation over all
            True pixels. If operation == 'random', picks a random True pixel
            and returns its DP.
        operation : {'mean','median','max','min','std','random','flat_mean'} | None
            Operation to apply. Defaults to 'mean' if None. The reduction
            operations aggregate over the selected real-space region (or the
            full scan when no region is given). ``random`` selects one pattern;
            ``flat_mean`` uses radial masking and a flat-field threshold.
        selection_units : {'pixels', 'calibrated', 'auto'}, optional
            Unit system used to interpret ``y`` and ``x`` for 4D data.
            Defaults to ``'pixels'`` so scalar calls such as ``get_dp(i, j)``
            always select scan indices, even when real-space calibration is
            stored. Use ``'calibrated'`` to select using real-space units.
    
        Returns
        -------
        ReciprocalSpace or ndarray
        """
    
        operations = {
            'mean':   np.mean,
            'median': np.median,
            'max':    np.max,
            'min':    np.min,
            'std':    np.std,
            'random': 'random',
            'flat_mean': None,
        }

        if (
            isinstance(y, str)
            and x is None
            and operation is None
            and y in operations
        ):
            operation = y
            y = None
    
        # Default operation is mean
        if operation is None:
            operation = 'mean'
    
        if operation not in operations:
            valid_ops = ', '.join(f"'{op}'" for op in operations.keys())
            raise ValueError(f"'operation' must be one of: {valid_ops}.")

        selection_mode = _normalize_unit_mode(
            selection_units,
            label='selection_units',
        )
        selection_factor = None
        if self.ndim == 4:
            _, selection_factor, selection_mode = self._resolve_real_selection_units(
                selection_units
            )
        elif selection_mode == 'calibrated':
            raise ValueError(
                "selection_units='calibrated' is only supported for 4D data, "
                "where y and x are real-space scan coordinates."
            )
    
        # Selecting a random diffraction pattern
        if operation == 'random':
            rng = np.random
            if self.ndim not in (3, 4):
                raise ValueError("Random selection supported only for 3D or 4D arrays.")
    
            # (2) mask constraint
            if mask is not None:
                if self.ndim != 4:
                    raise ValueError("mask requires a 4D dataset (A,B,C,D).")
                if mask.shape != (self.shape[0], self.shape[1]):
                    raise ValueError(f"'mask' must be shape {(self.shape[0], self.shape[1])}.")
                ys, xs = np.where(mask)
                if ys.size == 0:
                    raise ValueError("mask has no True pixels.")
                idx = rng.randint(0, ys.size)
                y_pos, x_pos = int(ys[idx]), int(xs[idx])
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos}) using mask...\n")
                return self._spawn_reciprocal(self.array[y_pos, x_pos])
    
            # Region via y/x. Units are resolved before random sampling.
            if self.ndim == 4 and (y is not None or x is not None):
                A, B = self.shape[0], self.shape[1]
                y0, y1, _ = _parse_real_selection(
                    y,
                    A,
                    'y',
                    selection_mode,
                    _real_spacing_pair(selection_factor)[0],
                    self.real_origin[0],
                )
                x0, x1, _ = _parse_real_selection(
                    x,
                    B,
                    'x',
                    selection_mode,
                    _real_spacing_pair(selection_factor)[1],
                    self.real_origin[1],
                )
                y_pos = rng.randint(y0, y1)
                x_pos = rng.randint(x0, x1)
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos}) within specified region...\n")
                return self._spawn_reciprocal(self.array[y_pos, x_pos])
    
            # Global random selection
            if self.ndim == 4:
                y_pos = int(self.shape[0] * rng.random())
                x_pos = int(self.shape[1] * rng.random())
                print(f"Acquired diffraction pattern at position ({y_pos}, {x_pos})...\n")
                return self._spawn_reciprocal(self.array[y_pos, x_pos])
            else:  # self.ndim == 3
                idx = int(self.shape[0] * rng.random())
                print(f"Acquired diffraction pattern of index {idx}...\n")
                return self._spawn_reciprocal(self.array[idx])
    
        # Automatically finding the mean of non-tilt regions
        #TODO: verify generalizability
        if operation == 'flat_mean':
            r = (self.shape[-2] + self.shape[-1]) / 4
            r_min = 0.6 * r
            r_max = 0.8 * r
            reduced_data = self.apply_mask(r_min, r_max,)
            mean, flat_mask = mask_and_average(
                reduced_data.array,
                return_mask=True,
                function='sum_2d',
                threshold='upper',
                **flat_kwargs
            )
            result = np.mean(self.apply_mask(mask=flat_mask, domain='real',).array, axis=0)
            return self._spawn_reciprocal(result)
    
        # From here on: mean/median/max/min
        agg_func = operations[operation]
    
        # Operating on a provided real-space mask
        if mask is not None:
            if self.ndim != 4:
                raise ValueError("mask requires a 4D dataset (A,B,C,D).")
            if mask.shape != (self.shape[0], self.shape[1]):
                raise ValueError(f"'mask' must be shape {(self.shape[0], self.shape[1])}.")
            sub = self.array[mask]   # shape (N, C, D)
            result = agg_func(sub, axis=0)
            return self._spawn_reciprocal(result)
    
        # Operating based on the (y, x) inputs
        if y is not None or x is not None:
            if self.ndim == 4:
                A, B = self.shape[0], self.shape[1]
                y0, y1, y_kind = _parse_real_selection(
                    y,
                    A,
                    'y',
                    selection_mode,
                    _real_spacing_pair(selection_factor)[0],
                    self.real_origin[0],
                )
                x0, x1, x_kind = _parse_real_selection(
                    x,
                    B,
                    'x',
                    selection_mode,
                    _real_spacing_pair(selection_factor)[1],
                    self.real_origin[1],
                )

                if y_kind == 'index' and x_kind == 'index':
                    return self._spawn_reciprocal(self.array[y0, x0])

                sub = self.array[y0:y1, x0:x1, :, :]
                result = agg_func(sub, axis=(0, 1))
                return self._spawn_reciprocal(result)
    
            elif self.ndim == 3:
                # For 3D: only support picking by index, as before
                if isinstance(y, (Integral, np.integer)) and x is None:
                    return self._spawn_reciprocal(self.array[int(y), :, :])
                if y is None and isinstance(x, (Integral, np.integer)):
                    return self._spawn_reciprocal(self.array[int(x), :, :])
                raise ValueError("Region-based (tuple) selection currently only supported for 4D arrays.")
    
        # Global Operation
        # If no y/x or masks are given, apply global aggregation over (A,B)
        if self.ndim == 4:
            result = agg_func(self.array, axis=(0, 1))
            return self._spawn_reciprocal(result)
        elif self.ndim == 3:
            result = agg_func(self.array, axis=0)
            return self._spawn_reciprocal(result)
    
        raise ValueError("No parameters specified.")


    @staticmethod
    def _normalize_resize_shape(shape, expected_ndim, label):
        """Normalize a requested resize shape to a tuple of positive integers."""
        if expected_ndim == 1 and isinstance(shape, (Integral, np.integer)):
            shape = (int(shape),)
        elif isinstance(shape, (tuple, list)):
            if not all(isinstance(v, (Integral, np.integer)) for v in shape):
                raise ValueError(f"{label} must contain integers.")
            shape = tuple(int(v) for v in shape)
        else:
            raise ValueError(f"{label} must be an integer or tuple of integers.")

        if len(shape) != expected_ndim:
            raise ValueError(f"{label} must have {expected_ndim} dimension(s).")
        if any(v <= 0 for v in shape):
            raise ValueError(f"{label} must contain positive integers.")
        return shape

    @staticmethod
    def _resize_axis_area(array, new_size, axis):
        """
        Resize one axis by area-weighted averaging.

        Exact integer factors reduce to standard block averaging. Non-integer
        factors use fractional overlap weights so each output bin integrates the
        corresponding input interval.
        """
        old_size = array.shape[axis]
        if new_size > old_size:
            raise ValueError("resize currently supports downsampling only.")

        out_dtype = np.result_type(array.dtype, np.float64)
        if new_size == old_size:
            return array.astype(out_dtype, copy=True)

        moved = np.moveaxis(array, axis, 0).astype(out_dtype, copy=False)
        resized = np.zeros((new_size,) + moved.shape[1:], dtype=out_dtype)
        scale = old_size / new_size

        for out_idx in range(new_size):
            start = out_idx * scale
            stop = (out_idx + 1) * scale
            first = max(0, int(np.floor(start)))
            last = min(old_size, int(np.ceil(stop)))

            for in_idx in range(first, last):
                overlap = min(stop, in_idx + 1) - max(start, in_idx)
                if overlap > 0:
                    resized[out_idx] += moved[in_idx] * overlap

            resized[out_idx] /= scale

        return np.moveaxis(resized, 0, axis)

    @classmethod
    def _resize_area(cls, array, output_shape, axes):
        """Apply area-weighted resizing over one or more axes."""
        resized = array
        for axis, new_size in zip(axes, output_shape):
            resized = cls._resize_axis_area(resized, new_size, axis)
        return resized

    @staticmethod
    def _resize_interpolation(array, output_shape, axes, method='linear'):
        """Resize selected axes with interpolation, supporting up/downsampling."""
        target_shape = list(array.shape)
        for axis, new_size in zip(axes, output_shape):
            target_shape[axis] = int(new_size)

        order = 0 if method == 'nearest' else 1
        downsampling = any(new < old for new, old in zip(output_shape, [array.shape[a] for a in axes]))
        resized = transform.resize(
            array,
            tuple(target_shape),
            order=order,
            mode='reflect',
            anti_aliasing=(order > 0 and downsampling),
            preserve_range=True,
        )
        if order == 0:
            return resized.astype(array.dtype, copy=False)
        return resized

    @staticmethod
    def _resized_calibration(units, conv_factor, scale_factors, label):
        """
        Update units-per-pixel calibration after resizing.

        Real-space spacing may be anisotropic. Reciprocal-space calibration is
        still scalar and is cleared if the resize makes its pixels anisotropic.
        """
        if conv_factor is None:
            return units, conv_factor

        scale_factors = tuple(float(v) for v in scale_factors)
        if label == 'real' and len(scale_factors) == 2:
            return units, _scaled_real_spacing(conv_factor, scale_factors)
        if label == 'real' and len(scale_factors) == 1:
            return units, _scaled_real_spacing(
                conv_factor, (scale_factors[0], scale_factors[0]),
            )
        if len(scale_factors) == 1 or np.allclose(scale_factors, scale_factors[0]):
            return units, conv_factor * scale_factors[0]

        print(
            f"Warning: anisotropic {label}-space resizing cleared the stored "
            f"{label}-space calibration."
        )
        return None, None

    def resize(self, shape, domain, method='area'):
        """
        Resize real- or reciprocal-space dimensions.

        Parameters
        ----------
        shape : int or tuple
            Requested output shape for the selected domain. For 4D data,
            ``domain='real'`` expects ``(Ny, Nx)`` and ``domain='reciprocal'``
            expects ``(Ky, Kx)``. For 3D stacks, ``domain='real'`` expects the
            number of diffraction patterns and ``domain='reciprocal'`` expects
            ``(Ky, Kx)``.
        domain : {'real', 'reciprocal'}
            Which axes to resize.
        method : {'area', 'linear', 'nearest'}, optional
            ``'area'`` performs weighted area averaging and supports
            downsampling only. ``'linear'`` and ``'nearest'`` use interpolation
            and support both downsampling and upsampling.

        Real-space resizing updates each pixel spacing independently and
        shifts ``real_origin`` to the center of the first output pixel.
        """
        if domain is None:
            raise ValueError("domain must be 'real' or 'reciprocal'.")
        if not isinstance(domain, str):
            raise ValueError("domain must be 'real' or 'reciprocal'.")
        if not isinstance(method, str):
            raise ValueError("method must be a string.")

        domain = domain.lower()
        method = method.lower()
        if domain not in ('real', 'reciprocal'):
            raise ValueError("domain must be 'real' or 'reciprocal'.")
        if method not in ('area', 'linear', 'nearest'):
            raise ValueError("method must be 'area', 'linear', or 'nearest'.")
        if self.ndim not in (3, 4):
            raise ValueError("resize currently supports 3D or 4D datasets.")

        if self.ndim == 4 and domain == 'real':
            output_shape = self._normalize_resize_shape(shape, 2, 'shape')
            current_shape = self.shape[:2]
            axes = (0, 1)
        elif self.ndim == 4 and domain == 'reciprocal':
            output_shape = self._normalize_resize_shape(shape, 2, 'shape')
            current_shape = self.shape[2:4]
            axes = (2, 3)
        elif self.ndim == 3 and domain == 'real':
            output_shape = self._normalize_resize_shape(shape, 1, 'shape')
            current_shape = (self.shape[0],)
            axes = (0,)
        else:
            output_shape = self._normalize_resize_shape(shape, 2, 'shape')
            current_shape = self.shape[1:3]
            axes = (1, 2)

        if method == 'area' and any(new > old for new, old in zip(output_shape, current_shape)):
            raise ValueError(
                "resize with method='area' supports downsampling only: requested "
                f"{output_shape} from {current_shape}."
            )

        if method == 'area':
            resized = self._resize_area(self.array, output_shape, axes)
        else:
            resized = self._resize_interpolation(
                self.array,
                output_shape,
                axes,
                method=method,
            )

        real_units = self.real_units
        real_conv = self.real_conv_factor
        real_origin = self.real_origin
        reciprocal_units = self.reciprocal_units
        reciprocal_conv = self.reciprocal_conv_factor
        scale_factors = tuple(
            old / new for old, new in zip(current_shape, output_shape)
        )

        if domain == 'real':
            real_units, real_conv = self._resized_calibration(
                real_units, real_conv, scale_factors, 'real'
            )
            if self.ndim == 4:
                old_step = _real_spacing_pair(self.real_conv_factor)
                real_origin = tuple(
                    old_origin + 0.5 * (factor - 1) * step
                    for old_origin, factor, step in zip(
                        self.real_origin, scale_factors, old_step,
                    )
                )
        else:
            reciprocal_units, reciprocal_conv = self._resized_calibration(
                reciprocal_units, reciprocal_conv, scale_factors, 'reciprocal'
            )

        beam_metadata = self.center_beam_metadata
        if domain == 'reciprocal' and output_shape != current_shape:
            beam_metadata = _resampled_center_beam_metadata(
                self.center_beam_metadata, current_shape, output_shape,
                units=reciprocal_units, conv_factor=reciprocal_conv,
            ) if self.polar_metadata is None else None

        return self._spawn(
            resized,
            real_units=real_units,
            real_conv_factor=real_conv,
            real_origin=real_origin,
            reciprocal_units=reciprocal_units,
            reciprocal_conv_factor=reciprocal_conv,
            center_beam_metadata=beam_metadata,
        )

    def bin_data(self, domain=None, iterations=1, *, bin_domain=None):
        """
        Bin data by powers of two through :meth:`resize`.

        ``domain`` must be provided explicitly as ``'real'`` or
        ``'reciprocal'``. ``bin_domain`` is accepted as a backwards-compatible
        keyword alias.
        """
        if domain is None:
            domain = bin_domain
        elif bin_domain is not None and domain != bin_domain:
            raise ValueError("domain and bin_domain refer to different domains.")
        if domain is None:
            raise ValueError("domain must be provided as 'real' or 'reciprocal'.")
        if not isinstance(iterations, (Integral, np.integer)) or iterations < 1:
            raise ValueError("iterations must be an integer >= 1.")
        if not isinstance(domain, str):
            raise ValueError("domain must be 'real' or 'reciprocal'.")

        domain = domain.lower()
        if domain not in ('real', 'reciprocal'):
            raise ValueError("domain must be 'real' or 'reciprocal'.")

        bin_factor = 2 ** int(iterations)

        if self.ndim == 4 and domain == 'real':
            output_shape = (
                self.shape[0] // bin_factor,
                self.shape[1] // bin_factor,
            )
        elif self.ndim == 4 and domain == 'reciprocal':
            output_shape = (
                self.shape[2] // bin_factor,
                self.shape[3] // bin_factor,
            )
        elif self.ndim == 3 and domain == 'real':
            output_shape = self.shape[0] // bin_factor
        elif self.ndim == 3 and domain == 'reciprocal':
            output_shape = (
                self.shape[1] // bin_factor,
                self.shape[2] // bin_factor,
            )
        else:
            raise ValueError("bin_data currently supports 3D or 4D datasets.")

        if isinstance(output_shape, tuple):
            too_small = any(v < 1 for v in output_shape)
        else:
            too_small = output_shape < 1
        if too_small:
            raise ValueError("iterations are too large for the selected domain.")

        return self.resize(output_shape, domain=domain, method='area')
    
    def virtual_image(self, annulus=None, *, radius=None, centers=None,
                      mask=None, theta_range=None, detector_units='auto',
                      axis_units='auto', vmin=None, vmax=None,
                      grid=True, num_div=10, plot_mask=False,
                      grid_color='black', axes=True,
                      return_detector=False, title='Virtual Detector Image',
                      units='auto', mask_cmap='turbo', mask_vmin=None,
                      mask_vmax=None, mask_power=1, mask_log_scale=True,
                      mask_show_kwargs=None, ring_color='green',
                      ring_alpha=0.3, show=True):
        """
        Form a real-space virtual image using a reciprocal-space detector.

        The detector type is inferred from exactly one of these specifications:

        - ``annulus=(inner_radius, outer_radius)`` for a centered annulus.
        - ``radius`` and ``centers`` for one or more Cartesian disks.
        - ``mask`` for an arbitrary Boolean or weighted detector.

        Parameters
        ----------
        annulus : array-like of two floats or None, optional
            Inner and outer detector radii. Interpreted according to
            ``detector_units``. For Cartesian data the annulus is centered at
            the diffraction origin. For polar data it selects a radial ``kr``
            interval.
        radius : float or array-like of floats or None, optional
            Radius of each Cartesian circular detector. Interpreted according
            to ``detector_units``. A scalar is applied to every center;
            otherwise provide one radius per center.
        centers : array-like or None, optional
            One ``(ky, kx)`` center or an ``(N, 2)`` array of centers. Multiple
            circles are combined as a union. Pixel centers are array
            coordinates. Calibrated centers are reciprocal coordinates relative
            to the diffraction origin, with positive ``ky`` upward and positive
            ``kx`` to the right. Only valid for Cartesian data.
        mask : ndarray or None, optional
            Arbitrary detector with shape matching the last two data axes.
            Boolean masks select pixels. Numeric masks act as detector weights
            and may contain positive, negative, or fractional values.
        theta_range : array-like of two floats or None, optional
            Angular interval in degrees for a polar annular detector. A wrapped
            interval such as ``(330, 30)`` is supported. This parameter is only
            valid with ``annulus`` on polar data.
        detector_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for ``annulus``, ``radius``, and ``centers``.
            ``'auto'`` uses reciprocal calibration when available and otherwise
            falls back to pixels.
        axis_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for the displayed real-space virtual image axes.
            ``'auto'`` uses real-space calibration when available and otherwise
            falls back to pixels.
        vmin, vmax : float or None, optional
            Display range passed to the real-space visualization.
        title : str, optional
            Title of the real-space virtual image.
        grid : bool, optional
            Whether to overlay a scan grid on the virtual image.
        num_div : int or tuple, optional
            Number of grid divisions along each real-space axis.
        plot_mask : bool, optional
            If True, show the unmasked mean diffraction pattern with the
            detector region overlaid. For ``annulus``, the overlay is a ring.
        units : {'auto', 'pixels', 'calibrated'}, optional
            Axis units for the diffraction pattern shown by ``plot_mask``.
            ``'auto'`` uses the stored reciprocal-space calibration.
        mask_cmap : str, optional
            Diffraction-pattern colormap used by ``plot_mask``.
        mask_vmin, mask_vmax : float or None, optional
            Display limits for the diffraction pattern shown by ``plot_mask``.
            These are independent of the virtual image's ``vmin`` and ``vmax``.
        mask_power : float, optional
            Intensity power passed to ``ReciprocalSpace.show``.
        mask_log_scale : bool, optional
            Whether the mean diffraction pattern uses logarithmic display.
        mask_show_kwargs : dict or None, optional
            Additional ``ReciprocalSpace.show`` options for the diffraction
            pattern, such as ``figsize``, ``aspect``, and ``axes``.
        ring_color : matplotlib color, optional
            Color of the detector overlay. Defaults to green.
        ring_alpha : float, optional
            Overlay opacity between 0 and 1. The default 0.3 is 70% transparent.
        grid_color : str, optional
            Grid color for the real-space visualization.
        axes : bool, optional
            Whether to show axes in the real-space visualization.
        return_detector : bool, optional
            If True, also return the masked mean diffraction pattern.
        show : bool, optional
            Display the virtual image when True. Set False to return a
            ``RealSpace`` image without creating its figure.

        Returns
        -------
        RealSpace or tuple
            The virtual image, optionally followed by the detector image. The
            detector image is ``ReciprocalSpace`` for Cartesian data and
            ``HyperData`` with polar metadata for polar data.

        Examples
        --------
        >>> image = data.virtual_image(annulus=(10, 35), detector_units='pixels')
        >>> image = data.virtual_image(radius=5, centers=(62, 74), detector_units='pixels')
        >>> image = data.virtual_image(
        ...     radius=[4, 5],
        ...     centers=[(62, 74), (51, 43)],
        ... )
        >>> image = data.virtual_image(mask=detector_weights)
        """
        if self.ndim != 4:
            raise ValueError("virtual_image requires a 4D HyperData object.")
        if mask_show_kwargs is not None and not isinstance(mask_show_kwargs, dict):
            raise TypeError("mask_show_kwargs must be a dictionary or None.")
        _normalize_unit_mode(units, label='units')
        if not np.isfinite(ring_alpha) or not 0 <= ring_alpha <= 1:
            raise ValueError("ring_alpha must be between 0 and 1.")
        to_rgba(ring_color)

        has_annulus = annulus is not None
        has_mask = mask is not None
        has_disk_input = radius is not None or centers is not None

        if (radius is None) != (centers is None):
            raise ValueError(
                "radius and centers must be provided together for circular "
                "detectors."
            )

        detector_count = sum((has_annulus, has_mask, has_disk_input))
        if detector_count != 1:
            raise ValueError(
                "Specify exactly one detector: annulus, radius with centers, "
                "or mask."
            )
        if has_disk_input and self.is_polar:
            raise ValueError(
                "radius and centers define Cartesian (ky, kx) circles and "
                "cannot be applied after to_polar(). Use annulus or mask."
            )

        ky, kx = self.shape[-2:]
        if has_annulus or has_disk_input:
            if self.is_polar:
                _, detector_factor, detector_mode = self._resolve_polar_radius_units(
                    detector_units
                )
            else:
                _, detector_factor, detector_mode = self._resolve_reciprocal_detector_units(
                    detector_units
                )
        else:
            _normalize_unit_mode(detector_units, label='detector_units')
            detector_factor = None
            detector_mode = 'pixels'

        if has_mask:
            if theta_range is not None:
                raise ValueError(
                    "theta_range is only valid with annulus on polar data."
                )
            detector_weights = np.asarray(mask)
            if detector_weights.shape != (ky, kx):
                raise ValueError(
                    "mask must match the reciprocal-space shape "
                    f"{(ky, kx)}; got {detector_weights.shape}."
                )
            if not (
                np.issubdtype(detector_weights.dtype, np.bool_)
                or np.issubdtype(detector_weights.dtype, np.number)
            ):
                raise TypeError("mask must contain Boolean or numeric values.")
            if np.issubdtype(detector_weights.dtype, np.complexfloating):
                raise TypeError("mask weights must be real-valued.")
            if (
                np.issubdtype(detector_weights.dtype, np.number)
                and not np.all(np.isfinite(detector_weights))
            ):
                raise ValueError("mask must contain only finite values.")
            detector_weights = detector_weights.astype(
                np.result_type(detector_weights.dtype, np.float32),
                copy=False,
            )

        elif has_annulus:
            annulus_values = np.asarray(annulus, dtype=float)
            if annulus_values.shape != (2,):
                raise ValueError(
                    "annulus must contain exactly (inner_radius, outer_radius)."
                )
            if not np.all(np.isfinite(annulus_values)):
                raise ValueError("annulus radii must be finite.")
            if detector_mode == 'calibrated':
                annulus_values = self._scale_values_to_pixels(
                    annulus_values,
                    detector_factor,
                )
            inner_radius, outer_radius = annulus_values
            if inner_radius < 0 or outer_radius <= inner_radius:
                raise ValueError(
                    "annulus must satisfy 0 <= inner_radius < outer_radius."
                )

            if self.is_polar:
                metadata = self.polar_metadata or {}
                radius_min, radius_max = metadata.get(
                    'radius_range_pixels',
                    (0.0, float(ky - 1)),
                )
                radius_values = np.linspace(radius_min, radius_max, ky)
                radial_mask = (
                    (radius_values >= inner_radius)
                    & (radius_values <= outer_radius)
                )

                angular_mask = np.ones(kx, dtype=bool)
                if theta_range is not None:
                    angle_values = np.asarray(theta_range, dtype=float)
                    if angle_values.shape != (2,):
                        raise ValueError(
                            "theta_range must contain exactly two angles."
                        )
                    if not np.all(np.isfinite(angle_values)):
                        raise ValueError("theta_range angles must be finite.")
                    theta_min, theta_max = angle_values
                    theta_start, theta_stop = metadata.get(
                        'theta_range',
                        (0.0, 360.0),
                    )
                    theta_values = np.linspace(
                        theta_start,
                        theta_stop,
                        kx,
                        endpoint=False,
                    )
                    if theta_min <= theta_max:
                        angular_mask = (
                            (theta_values >= theta_min)
                            & (theta_values <= theta_max)
                        )
                    else:
                        angular_mask = (
                            (theta_values >= theta_min)
                            | (theta_values <= theta_max)
                        )

                detector_weights = (
                    radial_mask[:, None] & angular_mask[None, :]
                ).astype(np.float32)
            else:
                if theta_range is not None:
                    raise ValueError(
                        "theta_range is only valid for polar data."
                    )
                detector_weights = make_mask(
                    ((ky - 1) / 2, (kx - 1) / 2),
                    (float(inner_radius), float(outer_radius)),
                    mask_dim=(ky, kx),
                ).astype(np.float32)

        else:
            if theta_range is not None:
                raise ValueError(
                    "theta_range is only valid with annulus on polar data."
                )

            center_values = np.asarray(centers, dtype=float)
            if center_values.shape == (2,):
                center_values = center_values.reshape(1, 2)
            elif center_values.ndim != 2 or center_values.shape[1] != 2:
                raise ValueError(
                    "centers must be one (ky, kx) pair or an (N, 2) array."
                )
            if center_values.shape[0] == 0:
                raise ValueError("centers must contain at least one center.")
            if not np.all(np.isfinite(center_values)):
                raise ValueError("centers must contain only finite values.")
            if detector_mode == 'calibrated':
                center_values = self._calibrated_centers_to_pixels(
                    center_values,
                    detector_factor,
                    (ky, kx),
                )

            radius_values = np.asarray(radius, dtype=float)
            if radius_values.ndim == 0:
                radius_values = np.full(
                    center_values.shape[0],
                    float(radius_values),
                )
            elif (
                radius_values.ndim == 1
                and radius_values.size == center_values.shape[0]
            ):
                pass
            else:
                raise ValueError(
                    "radius must be a scalar or contain one value per center."
                )
            if (
                not np.all(np.isfinite(radius_values))
                or np.any(radius_values <= 0)
            ):
                raise ValueError("All detector radii must be positive and finite.")
            if detector_mode == 'calibrated':
                radius_values = self._scale_values_to_pixels(
                    radius_values,
                    detector_factor,
                )

            detector_weights = np.zeros((ky, kx), dtype=np.float32)
            for center, disk_radius in zip(center_values, radius_values):
                detector_weights[
                    make_mask(
                        center,
                        float(disk_radius),
                        mask_dim=(ky, kx),
                    )
                ] = 1.0

        if not np.any(detector_weights != 0):
            raise ValueError("The detector selects no reciprocal-space pixels.")

        mean_detector = np.mean(self.array, axis=(0, 1))
        masked_image = np.sum(
            self.array * detector_weights[None, None, :, :],
            axis=(2, 3),
        )
        virtual_image = self._spawn_real(
            masked_image, quantity='Virtual detector signal',
        )

        detector_array = mean_detector * detector_weights
        if self.is_polar:
            detector_image = HyperData(
                detector_array,
                polar_metadata=deepcopy(self.polar_metadata),
            )
        else:
            detector_image = self._spawn_reciprocal(detector_array)

        has_negative_weights = np.any(detector_weights < 0)
        if vmin is None:
            if has_negative_weights:
                vmin = np.min(masked_image)
            else:
                positive_values = masked_image[masked_image > 0]
                vmin = (
                    np.min(positive_values)
                    if positive_values.size
                    else np.min(masked_image)
                )
        if vmax is None:
            vmax = np.max(masked_image)

        if show:
            virtual_image.show(
                title=title,
                cmap='gray',
                vmin=vmin,
                vmax=vmax,
                axes=axes,
                grid=grid,
                num_div=num_div,
                grid_color=grid_color,
                axis_units=axis_units,
            )

        if plot_mask:
            mask_options = {
                'title': 'Mean Diffraction Pattern and Virtual Detector',
                'cmap': mask_cmap,
                'vmin': mask_vmin,
                'vmax': mask_vmax,
                'power': mask_power,
                'logScale': mask_log_scale,
                'axis_units': units,
                'overlay_mask': detector_weights != 0,
                'overlay_color': ring_color,
                'overlay_alpha': ring_alpha,
            }
            if self.is_polar:
                mask_options['aspect'] = 'auto'
            if mask_show_kwargs:
                mask_options.update(mask_show_kwargs)
            self._spawn_reciprocal(mean_detector).show(**mask_options)

        if return_detector:
            return virtual_image, detector_image
        return virtual_image


    def get_peaks(self,
                  radius: float,
                  min_distance: int,
                  trench_width: float = 1.0,
                  kernel_amp: float = 1.0,
                  trench_amp: float = -0.5,
                  threshold_abs: float = 1.0,
                  threshold_rel: float = None,
                  r_range: tuple = None,
                  n_fold: int | None = None,
                  sym_mode: str = 'none',
                  sym_tolerance_px: float = 2.0,
                  center_tolerance_px: float = 2.0,
                  orbit_min_fraction: float = 0.5,
                  reorder: bool = False,
                  real_mask=None,
                  center=None,
                  return_details: bool = False):
        """Detect peaks independently in every diffraction pattern.

        This dataset-level method applies :meth:`ReciprocalSpace.get_peaks`
        to each diffraction pattern without requiring reference coordinates.
        Because the number of detected peaks may differ between patterns, the
        result is returned as a ragged collection of ``(n_peaks, 2)`` arrays.

        Parameters
        ----------
        radius : float
            Radius, in reciprocal-space pixels, of the positive disk in the
            peak-detection template.
        min_distance : int
            Minimum pixel separation between detected peaks.
        trench_width : float, optional
            Width of the negative ring surrounding the positive template.
        kernel_amp : float, optional
            Amplitude of the positive disk in the template.
        trench_amp : float, optional
            Amplitude of the negative surrounding trench.
        threshold_abs : float or None, optional
            Absolute correlation threshold. Set to None to use
            ``threshold_rel`` instead.
        threshold_rel : float or None, optional
            Correlation threshold relative to the maximum correlation.
        r_range : tuple or None, optional
            Radial search interval ``(r_min, r_max)`` in pixels.
        n_fold : int or None, optional
            Rotational-symmetry order used by the underlying peak detector.
        sym_mode : {'none', 'repair', 'prune', 'both'}, optional
            Rotational-symmetry handling mode.
        sym_tolerance_px : float, optional
            Matching tolerance for symmetry-related peaks, in pixels.
        center_tolerance_px : float, optional
            Radius used to identify central peaks, in pixels.
        orbit_min_fraction : float, optional
            Minimum detected fraction required for a symmetry orbit.
        reorder : bool, optional
            If True, reorder detected peaks into symmetry-related shells when
            supported by ``n_fold``.
        real_mask : ndarray of bool or None, optional
            For 4D data, only detect peaks where this ``(Ry, Rx)`` mask is
            True. Masked-out positions contain empty ``(0, 2)`` arrays. This
            parameter is not supported for 3D data.
        center : (ky, kx) or None, optional
            Reciprocal-space search center in pixels. If omitted, use valid
            center-beam metadata and then the pattern midpoint. This center
            controls both the radial search and symmetry grouping.
        return_details : bool, optional
            If True, return :class:`PeakDetectionResult` entries with
            correlation scores and measured/synthetic provenance.

        Returns
        -------
        peaks : list
            For 4D data, a nested list shaped ``[Ry][Rx]``. For 3D data, a
            list shaped ``[N]``. Every entry is an integer array with shape
            ``(n_peaks, 2)`` containing ``(y, x)`` peak coordinates, or a
            :class:`PeakDetectionResult` when ``return_details=True``.

        Notes
        -----
        Use :meth:`get_centers` afterward when the detected coordinates should
        be refined independently within each diffraction pattern.
        """
        if self.ndim not in (3, 4):
            raise ValueError("HyperData.get_peaks requires a 3D or 4D dataset.")
        if self.is_polar:
            raise ValueError(
                "get_peaks requires Cartesian diffraction patterns; "
                "use to_cartesian() before peak detection."
            )

        peak_kwargs = {
            'radius': radius,
            'min_distance': min_distance,
            'trench_width': trench_width,
            'kernel_amp': kernel_amp,
            'trench_amp': trench_amp,
            'threshold_abs': threshold_abs,
            'threshold_rel': threshold_rel,
            'r_range': r_range,
            'n_fold': n_fold,
            'sym_mode': sym_mode,
            'sym_tolerance_px': sym_tolerance_px,
            'center_tolerance_px': center_tolerance_px,
            'orbit_min_fraction': orbit_min_fraction,
            'reorder': reorder,
            'center': center,
            'return_details': return_details,
        }

        # The detector carries shared calibration and geometry; each call
        # supplies its own pattern without constructing another wrapper.
        detector = self._spawn_reciprocal(
            np.empty(self.shape[-2:], dtype=self.array.dtype),
        )

        def empty_result():
            coords = np.empty((0, 2), dtype=int)
            if not return_details:
                return coords
            return PeakDetectionResult(
                coords, np.empty(0, dtype=float),
                np.empty(0, dtype=bool), np.empty(0, dtype=int),
            )

        if self.ndim == 4:
            Ry, Rx = self.shape[:2]
            if real_mask is None:
                valid_positions = np.ones((Ry, Rx), dtype=bool)
            else:
                valid_positions = np.asarray(real_mask)
                if valid_positions.shape != (Ry, Rx):
                    raise ValueError(
                        f"real_mask must have shape {(Ry, Rx)}, got "
                        f"{valid_positions.shape}."
                    )
                if not np.issubdtype(valid_positions.dtype, np.bool_):
                    raise TypeError("real_mask must be a boolean array.")

            all_peaks = [[None for _ in range(Rx)] for _ in range(Ry)]
            positions = np.ndindex(Ry, Rx)
            for y, x in tqdm(
                positions,
                total=Ry * Rx,
                desc='Detecting peaks',
            ):
                if not valid_positions[y, x]:
                    all_peaks[y][x] = empty_result()
                    continue

                all_peaks[y][x] = detector.get_peaks(
                    **peak_kwargs, _array=self.array[y, x],
                )

            return all_peaks

        if real_mask is not None:
            raise ValueError("real_mask is only supported for 4D datasets.")

        all_peaks = []
        for index in tqdm(
            range(self.shape[0]),
            desc='Detecting peaks',
        ):
            all_peaks.append(detector.get_peaks(
                **peak_kwargs, _array=self.array[index],
            ))

        return all_peaks
        
        
    def get_centers(self, r, ref_coords, method='CoM', real_mask=None):
        """
        Compute Bragg spot centers for each diffraction pattern.
    
        Supports both 4D (Ny×Nx×C×D) and 3D (B×C×D) HyperData.
    
        Parameters
        ----------
        r : float
            Radius (in pixels) of the local window used by the center-finding
            algorithm in each diffaction pattern.
        ref_coords : array-like or nested list
            For 4D array input, shape ``(n_peaks, 2)`` supplies shared peaks.
            For a 4D ragged list, ``ref_coords[i][j]`` holds the peaks at
            scan position ``(i, j)``. For 3D array input, shape
            ``(n_peaks, 2)`` supplies shared peaks; a 3D ragged list uses
            ``ref_coords[i]`` for pattern ``i``.
    
        method : {'CoM', 'gaussian', 'elliptical_gaussian'}, optional
            Local center-refinement method for each diffraction pattern.
        real_mask : ndarray[bool] or None, optional
            Only used for 4D datasets. Boolean mask of shape (Ny, Nx) defining
            which real-space positions have centers computed. False positions
            are skipped and filled with zeros, retaining the peak count for
            array or ragged input. A 3D stack does not accept ``real_mask``.
    
        Returns
        -------
        centers : ndarray or list
            Fixed-peak input returns ``(Ny, Nx, n_peaks, 2)`` for 4D or
            ``(B, n_peaks, 2)`` for 3D. Ragged input returns a matching
            nested list of per-pattern coordinate arrays.
        """
        assert 2 < self.ndim < 5, "HyperData must be 3D or 4D"
    
        # 4D case
        if self.ndim == 4:
            Ny, Nx, _, _ = self.shape
    
            if real_mask is not None:
                real_mask = np.asarray(real_mask)
                if real_mask.shape != (Ny, Nx):
                    raise ValueError(
                        f"real_mask must have shape (Ny, Nx) = {(Ny, Nx)}, "
                        f"got {real_mask.shape}."
                    )
    
            # ragged list-of-lists input
            if isinstance(ref_coords, list):
                centers = []
                for i in tqdm(range(Ny), desc="Computing centers (4D)"):
                    row = []
                    for j in range(Nx):
                        coords_ij = np.asarray(ref_coords[i][j])
    
                        # If masked out, skip computation and fill zeros
                        if real_mask is not None and not real_mask[i, j]:
                            if coords_ij.size == 0:
                                c = coords_ij.reshape(0, 2)
                            else:
                                c = np.zeros_like(coords_ij, dtype=float)
                            row.append(c)
                            continue
    
                        c = _peak_centers_from_array(
                            self.array[i, j], r, coords_ij, method=method
                        )
                        row.append(c)
                    centers.append(row)
                return centers
    
            # fixed array input
            coords = np.asarray(ref_coords)
            n_peaks = coords.shape[0]
            all_centers = np.zeros((Ny, Nx, n_peaks, 2), dtype=float)
    
            for i in tqdm(range(Ny), desc="Computing centers (4D)"):
                for j in range(Nx):
                    # If masked out, leave zeros and skip
                    if real_mask is not None and not real_mask[i, j]:
                        continue
                    all_centers[i, j] = _peak_centers_from_array(
                        self.array[i, j], r, coords, method=method
                    )
            return all_centers
    
        # 3D case
        else:
            if real_mask is not None:
                raise ValueError("real_mask is only supported for 4D datasets.")
    
            B, _, _ = self.shape
    
            # ragged list input
            if isinstance(ref_coords, list):
                centers = []
                for i in tqdm(range(B), desc="Computing centers (3D)"):
                    coords_i = np.asarray(ref_coords[i])
                    c = _peak_centers_from_array(
                        self.array[i], r, coords_i, method=method
                    )
                    centers.append(c)
                return centers
    
            # fixed array input
            coords = np.asarray(ref_coords)
            n_peaks = coords.shape[0]
            all_centers = np.zeros((B, n_peaks, 2), dtype=float)
            for i in tqdm(range(B), desc="Computing centers (3D)"):
                all_centers[i] = _peak_centers_from_array(
                    self.array[i], r, coords, method=method
                )
            return all_centers



    def get_intensities(self,
                        r=6,
                        centers=None,
                        ref_coords=None,
                        method='CoM',
                        compute_resBg=False,
                        residual_frac=0.9,
                        real_mask=None,
                        **resBg_kwargs):
        """
        Extract Bragg peak intensities from each diffraction pattern slice.
    
        Supports both 4D (Ny×Nx×C×D) and 3D (B×C×D) HyperData.
    
        Parameters
        ----------
        r : float, optional
            Integration radius in pixels.
        centers : array-like or list, optional
            Fixed-peak arrays have shape ``(Ny, Nx, n_peaks, 2)`` for 4D or
            ``(B, n_peaks, 2)`` for 3D. Ragged input holds a coordinate
            array per pattern, nested by scan row for 4D. If None, centers
            are refined from ``ref_coords`` within each
            diffraction pattern before its intensities are integrated.
        ref_coords : array-like or nested list, optional
            Reference peak coordinates used when `centers is None`.
            See `get_centers` for allowed formats.
        method : {'CoM', 'gaussian', 'elliptical_gaussian'}, optional
            Center-finding method used if `centers` is None.
        compute_resBg : bool, optional
            If True, subtract the estimated residual background from each
            integrated peak using its actual number of included pixels.
        residual_frac : float, optional
            Fraction of the estimated background to subtract per included pixel.
        real_mask : ndarray[bool] or None, optional
            Only used for 4D datasets. Boolean mask of shape (Ny, Nx) defining
            which real-space positions have intensities computed. False
            positions are skipped and filled with zeros, retaining the peak
            count for array or ragged input. A 3D stack does not accept
            ``real_mask``.
        **resBg_kwargs :
            Additional keyword arguments forwarded to `dp.get_intensities(...)`.
    
        Returns
        -------
        all_ints : np.ndarray or list
            Fixed-peak input returns ``(Ny, Nx, n_peaks)`` for 4D or
            ``(B, n_peaks)`` for 3D. Ragged input returns a matching list of
            per-pattern intensity arrays.
        """
        assert 2 < self.ndim < 5, "HyperData must be 3D or 4D"
    
        infer_centers = centers is None
        if infer_centers and ref_coords is None:
            raise ValueError("ref_coords is required when centers is None.")

        def compute_dp_int(dp_array, dp_centers=None, dp_refs=None):
            if dp_centers is None:
                dp_centers = _peak_centers_from_array(
                    dp_array, r, dp_refs, method=method
                )
            if compute_resBg:
                dp = self._spawn_reciprocal(dp_array)
                return dp.get_intensities(
                    r=r,
                    centers=dp_centers,
                    compute_resBg=True,
                    residual_frac=residual_frac,
                    **resBg_kwargs,
                )
            return _peak_intensities_from_array(dp_array, r, dp_centers)
    
        # --------------------------- 4D case ---------------------------- #
        if self.ndim == 4:
            Ny, Nx, _, _ = self.shape
    
            if real_mask is not None:
                real_mask = np.asarray(real_mask)
                if real_mask.shape != (Ny, Nx):
                    raise ValueError(
                        f"real_mask must have shape (Ny, Nx) = {(Ny, Nx)}, "
                        f"got {real_mask.shape}."
                    )
    
            # ragged (list-of-lists) branch
            source = ref_coords if infer_centers else centers
            if isinstance(source, list):
                all_ints = []
                for i in tqdm(range(Ny), desc="Row"):
                    row_ints = []
                    for j in range(Nx):
                        dp_coords = np.asarray(source[i][j])

                        # If masked out, skip computation and fill zeros
                        if real_mask is not None and not real_mask[i, j]:
                            row_ints.append(np.zeros(len(dp_coords), dtype=float))
                            continue

                        if infer_centers:
                            row_ints.append(compute_dp_int(
                                self.array[i, j], dp_refs=dp_coords
                            ))
                        else:
                            row_ints.append(compute_dp_int(
                                self.array[i, j], dp_centers=dp_coords
                            ))
                    all_ints.append(row_ints)
                return all_ints

            # fixed-shape array branch
            else:
                source_arr = np.asarray(source)
                n_peaks = source_arr.shape[-2]
                all_ints = np.zeros((Ny, Nx, n_peaks), dtype=float)
                for i in tqdm(range(Ny), desc="Calculating intensities"):
                    for j in range(Nx):
                        # If masked out, leave zeros and skip
                        if real_mask is not None and not real_mask[i, j]:
                            continue
                        if infer_centers:
                            all_ints[i, j, :] = compute_dp_int(
                                self.array[i, j], dp_refs=source_arr
                            )
                        else:
                            all_ints[i, j, :] = compute_dp_int(
                                self.array[i, j], dp_centers=source_arr[i, j]
                            )
                return all_ints
    
        # --------------------------- 3D case ---------------------------- #
        else:
            if real_mask is not None:
                raise ValueError("real_mask is only supported for 4D datasets.")
    
            B, _, _ = self.shape
    
            # ragged (list) branch
            source = ref_coords if infer_centers else centers
            if isinstance(source, list):
                all_ints = []
                for i in tqdm(range(B), desc="DP"):
                    dp_coords = np.asarray(source[i])
                    if infer_centers:
                        all_ints.append(compute_dp_int(
                            self.array[i], dp_refs=dp_coords
                        ))
                    else:
                        all_ints.append(compute_dp_int(
                            self.array[i], dp_centers=dp_coords
                        ))
                return all_ints
    
            # fixed-shape array branch
            else:
                source_arr = np.asarray(source)
                n_peaks = source_arr.shape[-2]
                all_ints = np.zeros((B, n_peaks), dtype=float)
                for i in tqdm(range(B), desc="Calculating intensities"):
                    if infer_centers:
                        all_ints[i, :] = compute_dp_int(
                            self.array[i], dp_refs=source_arr
                        )
                    else:
                        all_ints[i, :] = compute_dp_int(
                            self.array[i], dp_centers=source_arr[i]
                        )
                return all_ints

    
    def get_residualBg(self, r=6, centers=None, ref_coords=None,
                       method='CoM', **resBg_kwargs):
        """Measure residual Bragg backgrounds for a 3D stack or 4D scan.

        ``centers`` may be a shared ``(n_peaks, 2)`` coordinate array, a
        scan-shaped array, or a ragged list following the scan layout. If it
        is omitted, ``ref_coords`` is required for center refinement.

        ``bg_method='rings'`` returns one value per peak: a scan-shaped
        ndarray when counts agree, or a nested list when they differ.
        ``'rings_mean'`` and ``'grimms_ring'`` return one scalar per pattern,
        with shape ``(N,)`` or ``(Ry, Rx)`` respectively.
        Additional keyword arguments go to ``ReciprocalSpace.get_residualBg``.
        """
        if self.ndim not in (3, 4):
            raise ValueError("get_residualBg requires a 3D stack or 4D scan.")
        if centers is None:
            if ref_coords is None:
                raise ValueError("Provide centers or ref_coords for background estimation.")
            centers = self.get_centers(r, ref_coords=ref_coords, method=method)

        bg_method = resBg_kwargs.get('bg_method', 'rings')
        if bg_method not in ('rings', 'rings_mean', 'grimms_ring'):
            raise ValueError(
                "bg_method must be 'rings', 'rings_mean', or 'grimms_ring'."
            )
        scan_shape = self.shape[:-2]
        try:
            center_array = np.asarray(centers, dtype=float)
        except (TypeError, ValueError):
            center_array = None
        shared_centers = (
            center_array if center_array is not None
            and center_array.ndim == 2 and center_array.shape[-1] == 2
            else None
        )
        if shared_centers is None and center_array is not None:
            if center_array.shape[:len(scan_shape)] != scan_shape:
                raise ValueError(
                    f"centers must follow scan shape {scan_shape}."
                )

        backgrounds = np.empty(scan_shape, dtype=object if bg_method == 'rings' else float)
        for index in tqdm(
            np.ndindex(*scan_shape), total=int(np.prod(scan_shape)),
            desc='Computing residual backgrounds',
        ):
            local_centers = shared_centers if shared_centers is not None else centers
            if shared_centers is None:
                for axis_index in index:
                    local_centers = local_centers[axis_index]
            local_centers = np.asarray(local_centers, dtype=float)
            if local_centers.ndim != 2 or local_centers.shape[1] != 2:
                raise ValueError(
                    f"centers at scan index {index} must have shape (n_peaks, 2)."
                )
            value = self.get_dp(*index, selection_units='pixels').get_residualBg(
                centers=local_centers, **resBg_kwargs,
            )
            backgrounds[index] = np.asarray(value, dtype=float) if bg_method == 'rings' else float(value)

        if bg_method != 'rings':
            return backgrounds
        if backgrounds.size == 0:
            return np.empty(scan_shape + (0,), dtype=float)
        vectors = list(backgrounds.flat)
        if all(vector.shape == vectors[0].shape for vector in vectors):
            return np.stack(vectors).reshape(scan_shape + vectors[0].shape)
        return backgrounds.tolist()

    def get_strains(self, centers=None, ref_centers=None, ang=0, g_vector=None,
                    r_CoM=None, r_inner=None, r_outer=None, intensity_array=None,
                    intensity_percentile=None, intensity_clip='both',
                    real_mask=None, ewpc=False, match_peaks='auto',
                    fit_translation=False, min_peak_pairs=2,
                    return_transform=False, center=None, peak_units='pixels',
                    max_condition_number=1e8,
                    reject_peak_outliers=False, outlier_threshold=3.5,
                    outlier_min_peak_pairs=None, outlier_min_peak_fraction=None,
                    outlier_max_iterations=2,
                    outlier_space='radial_angular',
                    outlier_method='auto_mixture',
                    outlier_bic_delta=0.0):
        """
        Calculate strain and rotation maps from Bragg peak centers.

        This method fits the best 2D linear transform from reference peak
        coordinates to measured peak coordinates at each diffraction pattern,
        then extracts strain from the polar stretch matrix. Peak sets are not
        required to have the same size or order. By default, reference and
        measured peaks are paired by a one-to-one, centroid-aligned spatial
        assignment before fitting.

        Parameters
        ----------
        centers : ndarray or nested list, optional
            Measured peak centers. Common fixed-size shape is
            ``(Ry, Rx, n_peaks, 2)``. Nested lists may be used when each
            diffraction pattern has a different number of peaks.
        ref_centers : ndarray or nested list, optional
            Reference peak centers. Supported forms are a global
            ``(n_ref_peaks, 2)`` array or a local reference with the same scan
            layout as ``centers``.
        ang : float, optional
            In-plane angle in degrees used to rotate the strain basis.
        g_vector : optional
            Retained for compatibility with older calls. The least-squares
            implementation uses all matched peaks rather than selecting a
            hexagonal g-vector pair.
        r_CoM : float, optional
            Radius passed to ``get_centers`` when measured centers are not
            supplied.
        intensity_array : ndarray or nested list, optional
            Optional peak weights. If supplied, weights are matched to measured
            peaks and used in the least-squares fit. Each local weight vector
            must match either all measured peaks or their finite subset.
        intensity_percentile : float or None, optional
            If supplied with ``intensity_array``, discard measured peaks with
            local outlier intensities before matching and fitting. With the
            default ``intensity_clip='both'``, the percentile is split evenly
            between tails, so ``intensity_percentile=10`` removes the lowest
            5% and highest 5% of finite measured peaks in each diffraction
            pattern.
        intensity_clip : {'both', 'lower', 'upper'}, optional
            Which intensity tail(s) to clip when ``intensity_percentile`` is
            supplied. ``'lower'`` clips only weak peaks, ``'upper'`` clips only
            bright outliers, and ``'both'`` clips both tails.
        real_mask : ndarray[bool] or None, optional
            Only used for 4D scan-shaped centers. Boolean mask of shape
            ``(Ry, Rx)`` defining which real-space positions should have strain
            computed. Masked-out positions remain NaN in strain/transform maps
            and have zero match counts.
        ewpc : bool, optional
            If True, invert the fitted reciprocal-space transform before
            extracting strain.
        match_peaks : {'auto', 'ordered', 'nearest'}, optional
            ``'auto'`` (the default) and ``'nearest'`` use one-to-one spatial
            assignment regardless of peak count or input order. This assumes
            the peak sets are approximately aligned before strain fitting.
            Use ``'ordered'`` only when corresponding peaks are already at
            matching indices in both inputs, including after intensity
            filtering.
        fit_translation : bool, optional
            If True, fit and remove an additional translation term after
            subtracting ``center`` so detector shifts are not interpreted as
            strain.
        min_peak_pairs : int, optional
            Minimum number of matched peak pairs required to fit a transform.
            With ``fit_translation=True``, at least three non-collinear pairs
            are required even if this value is smaller.
        return_transform : bool, optional
            If True, attach detailed transform and peak-rejection diagnostics
            to the result. For existing tuple-style code, these remain the
            fifth item when unpacking or indexing the result.
        center : array-like of shape (2,) or None, optional
            Reciprocal-space origin ``(ky, kx)`` about which peak displacements
            are measured before fitting strain. If None, this defaults to
            ``self.center_beam_metadata['mean_fit_center_px']`` when available.
            If that metadata is unavailable, it falls back to the finite mean
            coordinate of ``ref_centers`` when a reference is supplied, or to
            the finite absolute mean coordinate of all peaks in ``centers``
            when ``ref_centers`` is None.
        peak_units : {'pixels', 'calibrated'}, optional
            Units of supplied ``centers``, ``ref_centers``, and explicit
            ``center``. Pixel ``(ky, kx)`` coordinates are the default.
            Calibrated reciprocal ``(ky, kx)`` coordinates use positive ky
            upward and require stored reciprocal calibration. If the beam
            center comes from pixel metadata, it is converted automatically.
            Automatic center finding is pixel-only.
        max_condition_number : float, optional
            Reject peak configurations whose weighted reference design matrix
            exceeds this condition number. Such patterns remain NaN in the
            strain maps rather than producing unstable strain estimates.
        reject_peak_outliers : bool, optional
            If True, reject matched measured peaks that are poorly explained
            by the fitted strain transform. Rejection is performed after an
            initial transform fit and before the final strain fit.
        outlier_threshold : float, optional
            Robust z-score threshold used only when
            ``outlier_method='robust_zscore'``. With
            ``outlier_space='radial_angular'``, radial and angular residual
            z-scores are combined as a Euclidean score.
        outlier_min_peak_pairs : int or None, optional
            Minimum number of matched peak pairs required to attempt outlier
            rejection and the minimum number of pairs that must remain after
            rejection. If rejecting a candidate set would leave fewer pairs,
            only the most extreme candidates are rejected. The default of 4
            lets a six-peak pattern drop up to two outlier peaks while keeping
            enough peaks for a stable 2D transform fit. Mutually exclusive
            with ``outlier_min_peak_fraction``.
        outlier_min_peak_fraction : float or None, optional
            Alternative to ``outlier_min_peak_pairs``. If supplied, the local
            outlier minimum is ``ceil(fraction * n_reference_peaks)`` for each
            diffraction pattern, with a floor of 2 pairs. For example, ``0.5``
            requires half the local reference peak count to remain.
        outlier_max_iterations : int, optional
            Maximum number of initial-fit/reject cycles before the final fit.
        outlier_space : {'radial_angular', 'cartesian', 'distance'}, optional
            Residual coordinate system used for robust z-score outlier
            detection. The
            ``'radial_angular'`` mode resolves residuals into radial and
            tangential components relative to ``center`` and normalizes by
            peak radius.
        outlier_method : {'auto_mixture', 'robust_zscore'}, optional
            Peak-rejection strategy. ``'auto_mixture'`` fits the strain
            transform first, models the post-fit residual distances as either
            one population or two populations using BIC, and rejects the
            high-residual population only when the two-population model is
            preferred. ``'robust_zscore'`` uses ``outlier_threshold``.
        outlier_bic_delta : float, optional
            Extra BIC margin required before the automatic mixture method
            accepts a two-population residual model. The default ``0`` uses
            the standard BIC preference. Larger values make automatic
            rejection more conservative.

        Returns
        -------
        StrainResult
            Named strain and rotation maps, fit-quality maps, optional
            diagnostics, and real-space scan calibration. Strains are
            dimensionless; rotation is in radians. The result also supports
            four-value unpacking and indexed access used by older notebooks.

        Examples
        --------
        >>> result = data.get_strains(centers=measured, ref_centers=reference)
        >>> exx_image = result.as_real_space('exx')
        >>> fit_error = result.relative_fit_rmse
        >>> exx, eyy, exy, erot = result
        """
        def _is_sequence_of_sequences(value):
            return (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], list)
            )

        def _reference_for_center_finding(ref_centers):
            if _is_sequence_of_sequences(ref_centers):
                return ref_centers

            ref_array = np.asarray(ref_centers, dtype=float)
            if ref_array.ndim == 4 and ref_array.shape[-1] == 2:
                return [
                    [ref_array[i, j] for j in range(ref_array.shape[1])]
                    for i in range(ref_array.shape[0])
                ]
            return ref_centers

        def _infer_scan_shape(peak_data):
            if _is_sequence_of_sequences(peak_data):
                return len(peak_data), len(peak_data[0]), False
            if isinstance(peak_data, list):
                return len(peak_data), 1, True

            peak_array = np.asarray(peak_data, dtype=float)
            if peak_array.ndim == 4 and peak_array.shape[-1] == 2:
                return peak_array.shape[0], peak_array.shape[1], False
            if peak_array.ndim == 3 and peak_array.shape[-1] == 2:
                return peak_array.shape[0], 1, True
            raise ValueError(
                "centers must have shape (Ry, Rx, n_peaks, 2), "
                "shape (B, n_peaks, 2), or an equivalent nested list."
            )

        def _local_peaks(peak_data, i, j, squeezed_scan=False):
            if _is_sequence_of_sequences(peak_data):
                return np.asarray(peak_data[i][j], dtype=float)
            if isinstance(peak_data, list):
                return np.asarray(peak_data[i], dtype=float)

            peak_array = np.asarray(peak_data, dtype=float)
            if peak_array.ndim == 2:
                return peak_array
            if peak_array.ndim == 4:
                return peak_array[i, j]
            if peak_array.ndim == 3 and squeezed_scan:
                return peak_array[i]
            raise ValueError("Unsupported peak-center layout.")

        def _local_weights(weights, i, j, squeezed_scan=False):
            if weights is None:
                return None
            if _is_sequence_of_sequences(weights):
                return np.asarray(weights[i][j], dtype=float)
            if isinstance(weights, list):
                return np.asarray(weights[i], dtype=float)

            weight_array = np.asarray(weights, dtype=float)
            if weight_array.ndim == 1:
                return weight_array
            if weight_array.ndim == 3:
                return weight_array[i, j]
            if weight_array.ndim == 2 and squeezed_scan:
                return weight_array[i]
            raise ValueError(
                "intensity_array must have the same scan layout as centers "
                "and one weight per measured peak."
            )

        def _clean_peak_set(peaks):
            peaks = np.asarray(peaks, dtype=float)
            if peaks.size == 0:
                return peaks.reshape(0, 2), np.array([], dtype=int)
            peaks = peaks.reshape(-1, 2)
            valid = np.isfinite(peaks).all(axis=1)
            return peaks[valid], np.flatnonzero(valid)

        def _mean_peak_coordinate(peak_data, mask=None):
            coord_sets = []
            mask_array = None if mask is None else np.asarray(mask)

            if _is_sequence_of_sequences(peak_data):
                for i, row in enumerate(peak_data):
                    for j, peaks in enumerate(row):
                        if (
                            mask_array is not None
                            and mask_array.shape == (len(peak_data), len(row))
                            and not mask_array[i, j]
                        ):
                            continue
                        clean_peaks, _ = _clean_peak_set(peaks)
                        if clean_peaks.size:
                            coord_sets.append(clean_peaks)
            elif isinstance(peak_data, list):
                for i, peaks in enumerate(peak_data):
                    if (
                        mask_array is not None
                        and mask_array.shape == (len(peak_data),)
                        and not mask_array[i]
                    ):
                        continue
                    clean_peaks, _ = _clean_peak_set(peaks)
                    if clean_peaks.size:
                        coord_sets.append(clean_peaks)
            else:
                peak_array = np.asarray(peak_data, dtype=float)
                if peak_array.size == 0:
                    raise ValueError("Cannot infer center from an empty peak set.")
                if peak_array.shape[-1] != 2:
                    raise ValueError("Peak coordinates must end with a length-2 axis.")
                if (
                    mask_array is not None
                    and peak_array.ndim > mask_array.ndim
                    and peak_array.shape[:mask_array.ndim] == mask_array.shape
                ):
                    peak_array = peak_array[mask_array]
                clean_peaks, _ = _clean_peak_set(peak_array)
                if clean_peaks.size:
                    coord_sets.append(clean_peaks)

            if not coord_sets:
                raise ValueError("Cannot infer center from non-finite or empty peaks.")

            return np.mean(np.vstack(coord_sets), axis=0)

        def _validate_strain_center(center_value):
            center_array = np.asarray(center_value, dtype=float)
            if center_array.shape != (2,) or not np.isfinite(center_array).all():
                raise ValueError("center must be a finite coordinate pair (ky, kx).")
            return center_array

        def _metadata_strain_center():
            metadata = self.center_beam_metadata
            if not isinstance(metadata, dict) or 'mean_fit_center_px' not in metadata:
                return None
            try:
                center_px = _validate_strain_center(
                    metadata['mean_fit_center_px']
                )
                if peak_units == 'calibrated':
                    return np.asarray(_center_to_calibrated(
                        center_px, self.k_shape, self.reciprocal_conv_factor,
                    ))
                return center_px
            except ValueError as exc:
                raise ValueError(
                    "center_beam_metadata['mean_fit_center_px'] must be a "
                    "finite coordinate pair (ky, kx)."
                ) from exc

        def _weights_for_valid_measured(weights, meas_valid_idx, raw_count):
            if weights is None:
                return None

            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1:
                raise ValueError("Each intensity_array peak vector must be 1D.")
            if weights.shape[0] == raw_count:
                return weights[meas_valid_idx]
            if weights.shape[0] == meas_valid_idx.shape[0]:
                return weights
            raise ValueError(
                "intensity_array has a peak count that does not match "
                "the measured peaks in centers."
            )

        def _intensity_keep_mask(weights):
            finite_weights = np.isfinite(weights)
            if not np.any(finite_weights):
                return finite_weights

            finite_values = weights[finite_weights]
            if intensity_clip == 'both':
                tail_percentile = intensity_percentile / 2.0
                lower = np.percentile(finite_values, tail_percentile)
                upper = np.percentile(finite_values, 100.0 - tail_percentile)
                return finite_weights & (weights >= lower) & (weights <= upper)
            if intensity_clip == 'lower':
                lower = np.percentile(finite_values, intensity_percentile)
                return finite_weights & (weights >= lower)
            if intensity_clip == 'upper':
                upper = np.percentile(finite_values, 100.0 - intensity_percentile)
                return finite_weights & (weights <= upper)

            raise ValueError("intensity_clip must be 'both', 'lower', or 'upper'.")

        def _match_peak_sets(reference, measured, weights=None):
            raw_measured_count = np.asarray(measured).reshape(-1, 2).shape[0]
            reference, ref_valid_idx = _clean_peak_set(reference)
            measured, meas_valid_idx = _clean_peak_set(measured)
            measured_weights = _weights_for_valid_measured(
                weights, meas_valid_idx, raw_measured_count,
            )

            if intensity_percentile is not None:
                if measured_weights is None:
                    raise ValueError(
                        "intensity_percentile requires intensity_array values "
                        "that match the measured peaks in centers."
                    )
                keep = _intensity_keep_mask(measured_weights)
                if not np.any(keep):
                    return None, None, None, None, None

                measured = measured[keep]
                meas_valid_idx = meas_valid_idx[keep]
                measured_weights = measured_weights[keep]

            if reference.shape[0] < min_peak_pairs or measured.shape[0] < min_peak_pairs:
                return None, None, None, None, None

            match_mode = match_peaks.lower()
            if match_mode == 'auto':
                match_mode = 'nearest'

            if match_mode == 'ordered':
                n_pairs = min(reference.shape[0], measured.shape[0])
                ref_idx = np.arange(n_pairs)
                meas_idx = np.arange(n_pairs)
            elif match_mode in ('nearest', 'hungarian'):
                ref_centered = reference - np.mean(reference, axis=0)
                measured_centered = measured - np.mean(measured, axis=0)
                distances = cdist(ref_centered, measured_centered)
                ref_idx, meas_idx = linear_sum_assignment(distances)
            else:
                raise ValueError("match_peaks must be 'auto', 'ordered', or 'nearest'.")

            matched_weights = None
            if measured_weights is not None:
                matched_weights = measured_weights[meas_idx]

            return (
                reference[ref_idx],
                measured[meas_idx],
                matched_weights,
                ref_valid_idx[ref_idx],
                meas_valid_idx[meas_idx],
            )

        def _fit_peak_transform(reference, measured, strain_center, weights=None):
            reference = reference - strain_center
            measured = measured - strain_center

            if weights is not None:
                weights = np.asarray(weights, dtype=float)
                weights = np.where(np.isfinite(weights), weights, 0)
                weights = np.clip(weights, 0, None)
                if np.sum(weights) <= 0:
                    weights = None

            if fit_translation:
                if weights is None:
                    ref_origin = np.mean(reference, axis=0)
                    measured_origin = np.mean(measured, axis=0)
                else:
                    ref_origin = np.average(reference, axis=0, weights=weights)
                    measured_origin = np.average(measured, axis=0, weights=weights)
                X = reference - ref_origin
                Y = measured - measured_origin
            else:
                ref_origin = np.zeros(2)
                measured_origin = np.zeros(2)
                X = reference
                Y = measured

            if weights is None:
                X_fit = X
                Y_fit = Y
            else:
                sqrt_weights = np.sqrt(weights)[:, np.newaxis]
                X_fit = X * sqrt_weights
                Y_fit = Y * sqrt_weights

            coeffs, _, rank, singular_values = np.linalg.lstsq(
                X_fit, Y_fit, rcond=None,
            )
            if (
                rank < 2 or singular_values[-1] <= 0
                or singular_values[0] / singular_values[-1] > max_condition_number
            ):
                raise np.linalg.LinAlgError(
                    "Matched reference peaks are collinear or too ill-conditioned "
                    "for a reliable 2D strain fit."
                )
            transform_matrix = coeffs.T
            translation = measured_origin - ref_origin @ coeffs
            return transform_matrix, translation

        def _robust_zscore(values):
            values = np.asarray(values, dtype=float)
            z = np.full(values.shape, np.inf, dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                return z

            finite_values = values[finite]
            median = np.median(finite_values)
            deviation = np.abs(finite_values - median)
            scale = 1.4826 * np.median(deviation)
            if scale <= np.finfo(float).eps:
                z[finite] = 0.0
                z[np.flatnonzero(finite)[deviation > np.finfo(float).eps]] = np.inf
                return z

            z[finite] = np.abs(values[finite] - median) / scale
            return z

        def _peak_residual_vectors(reference, measured, transform_matrix,
                                   translation, strain_center):
            reference_centered = reference - strain_center
            measured_centered = measured - strain_center
            predicted = reference_centered @ transform_matrix.T + translation
            return measured_centered - predicted

        def _peak_residual_distances(reference, measured, transform_matrix,
                                     translation, strain_center):
            residual = _peak_residual_vectors(
                reference,
                measured,
                transform_matrix,
                translation,
                strain_center,
            )
            return np.linalg.norm(residual, axis=1)

        def _fit_residual_quality(reference, measured, transform_matrix,
                                  translation, strain_center, weights):
            residual = _peak_residual_vectors(
                reference, measured, transform_matrix, translation, strain_center,
            )
            residual_sq = np.einsum('ij,ij->i', residual, residual)
            reference_centered = reference - strain_center
            reference_radius_sq = np.einsum(
                'ij,ij->i', reference_centered, reference_centered,
            )
            if weights is not None:
                weights = np.asarray(weights, dtype=float)
                weights = np.where(np.isfinite(weights), weights, 0.0)
                weights = np.clip(weights, 0.0, None)
                if np.sum(weights) <= 0:
                    weights = None

            if weights is None:
                mean_residual_sq = np.mean(residual_sq)
                mean_reference_radius_sq = np.mean(reference_radius_sq)
            else:
                mean_residual_sq = np.average(residual_sq, weights=weights)
                mean_reference_radius_sq = np.average(
                    reference_radius_sq, weights=weights,
                )

            rmse = float(np.sqrt(mean_residual_sq))
            reference_rms_radius = float(np.sqrt(mean_reference_radius_sq))
            relative_rmse = (
                rmse / reference_rms_radius
                if reference_rms_radius > np.finfo(float).eps
                else np.nan
            )
            return rmse, relative_rmse

        def _peak_outlier_scores(reference, measured, transform_matrix,
                                 translation, strain_center):
            reference_centered = reference - strain_center
            residual = _peak_residual_vectors(
                reference,
                measured,
                transform_matrix,
                translation,
                strain_center,
            )

            if outlier_space == 'radial_angular':
                radii = np.linalg.norm(reference_centered, axis=1)
                safe_radii = np.where(radii > np.finfo(float).eps, radii, 1.0)
                radial_unit = np.divide(
                    reference_centered,
                    safe_radii[:, np.newaxis],
                    out=np.zeros_like(reference_centered, dtype=float),
                    where=safe_radii[:, np.newaxis] > 0,
                )
                tangential_unit = np.column_stack(
                    (-radial_unit[:, 1], radial_unit[:, 0])
                )
                radial_residual = np.sum(residual * radial_unit, axis=1) / safe_radii
                angular_residual = np.sum(residual * tangential_unit, axis=1) / safe_radii
                radial_z = _robust_zscore(radial_residual)
                angular_z = _robust_zscore(angular_residual)
                score = np.hypot(radial_z, angular_z)
            elif outlier_space == 'cartesian':
                y_z = _robust_zscore(residual[:, 0])
                x_z = _robust_zscore(residual[:, 1])
                score = np.hypot(y_z, x_z)
            elif outlier_space == 'distance':
                score = _robust_zscore(np.linalg.norm(residual, axis=1))
            else:
                raise ValueError(
                    "outlier_space must be 'radial_angular', 'cartesian', "
                    "or 'distance'."
                )

            return score

        def _auto_mixture_outlier_mask(reference, measured, transform_matrix,
                                       translation, strain_center):
            residual_distances = _peak_residual_distances(
                reference,
                measured,
                transform_matrix,
                translation,
                strain_center,
            )
            probabilities = np.full(residual_distances.shape, np.nan, dtype=float)
            model_info = {
                'method': 'auto_mixture',
                'accepted_two_component_model': False,
                'bic_one_component': np.nan,
                'bic_two_component': np.nan,
                'low_residual_log_mean': np.nan,
                'high_residual_log_mean': np.nan,
            }
            reject = np.zeros(residual_distances.shape, dtype=bool)

            finite = np.isfinite(residual_distances)
            if np.count_nonzero(finite) < 2:
                return reject, residual_distances, probabilities, model_info

            finite_distances = residual_distances[finite]
            if np.ptp(finite_distances) <= np.finfo(float).eps:
                return reject, residual_distances, probabilities, model_info

            eps = max(np.finfo(float).eps, np.nanmedian(finite_distances) * 1e-12)
            log_distances = np.log10(np.maximum(finite_distances, eps))
            if np.unique(log_distances).size < 2:
                return reject, residual_distances, probabilities, model_info

            values = log_distances.reshape(-1, 1)
            low_start, high_start = np.percentile(log_distances, [25, 90])
            if high_start <= low_start:
                low_start, high_start = np.min(log_distances), np.max(log_distances)

            one_component = GaussianMixture(
                n_components=1,
                covariance_type='full',
                reg_covar=1e-9,
                random_state=0,
                init_params='random',
                n_init=1,
            )
            two_component = GaussianMixture(
                n_components=2,
                covariance_type='full',
                reg_covar=1e-9,
                random_state=0,
                init_params='random',
                means_init=np.array([[low_start], [high_start]]),
                weights_init=np.array([0.8, 0.2]),
                n_init=1,
            )
            try:
                one_component.fit(values)
                two_component.fit(values)
            except ValueError:
                return reject, residual_distances, probabilities, model_info

            bic_one = one_component.bic(values)
            bic_two = two_component.bic(values)
            model_info['bic_one_component'] = float(bic_one)
            model_info['bic_two_component'] = float(bic_two)
            if not np.isfinite(bic_one) or not np.isfinite(bic_two):
                return reject, residual_distances, probabilities, model_info
            if bic_two + outlier_bic_delta >= bic_one:
                return reject, residual_distances, probabilities, model_info

            means = two_component.means_.reshape(-1)
            high_component = int(np.argmax(means))
            low_component = int(np.argmin(means))
            model_info['low_residual_log_mean'] = float(means[low_component])
            model_info['high_residual_log_mean'] = float(means[high_component])

            labels = two_component.predict(values)
            high_probabilities = two_component.predict_proba(values)[:, high_component]
            finite_indices = np.flatnonzero(finite)
            probabilities[finite_indices] = high_probabilities
            reject[finite_indices] = labels == high_component
            model_info['accepted_two_component_model'] = bool(np.any(reject))
            return reject, residual_distances, probabilities, model_info

        def _resolve_outlier_min_peak_pairs(reference_peak_count):
            if outlier_min_peak_fraction is None:
                return outlier_min_peak_pairs
            return max(
                2,
                int(np.ceil(float(reference_peak_count) * outlier_min_peak_fraction)),
            )

        def _reject_peak_outliers(reference, measured, weights, strain_center,
                                  local_outlier_min_peak_pairs):
            keep = np.ones(reference.shape[0], dtype=bool)
            residual_distances = np.full(reference.shape[0], np.nan, dtype=float)
            probabilities = np.full(reference.shape[0], np.nan, dtype=float)
            model_info = {
                'method': outlier_method,
                'accepted_two_component_model': False,
                'bic_one_component': np.nan,
                'bic_two_component': np.nan,
            }
            if (
                not reject_peak_outliers
                or reference.shape[0] < local_outlier_min_peak_pairs
                or outlier_max_iterations <= 0
            ):
                return keep, residual_distances, probabilities, model_info

            min_remaining = max(min_peak_pairs, local_outlier_min_peak_pairs)
            for _ in range(outlier_max_iterations):
                if np.count_nonzero(keep) < min_remaining:
                    break

                current_weights = weights[keep] if weights is not None else None
                transform_matrix, translation = _fit_peak_transform(
                    reference[keep],
                    measured[keep],
                    strain_center,
                    weights=current_weights,
                )
                kept_indices = np.flatnonzero(keep)
                if outlier_method == 'auto_mixture':
                    (
                        local_reject,
                        local_residual_distances,
                        local_probabilities,
                        local_model_info,
                    ) = _auto_mixture_outlier_mask(
                        reference[keep],
                        measured[keep],
                        transform_matrix,
                        translation,
                        strain_center,
                    )
                    residual_distances[kept_indices] = local_residual_distances
                    probabilities[kept_indices] = local_probabilities
                    model_info = local_model_info
                else:
                    scores = _peak_outlier_scores(
                        reference[keep],
                        measured[keep],
                        transform_matrix,
                        translation,
                        strain_center,
                    )
                    local_residual_distances = _peak_residual_distances(
                        reference[keep],
                        measured[keep],
                        transform_matrix,
                        translation,
                        strain_center,
                    )
                    residual_distances[kept_indices] = local_residual_distances
                    local_reject = (~np.isfinite(scores)) | (scores > outlier_threshold)

                if not np.any(local_reject):
                    break

                available_rejections = np.count_nonzero(keep) - min_remaining
                if available_rejections <= 0:
                    break

                reject_candidates = np.flatnonzero(local_reject)
                if reject_candidates.size > available_rejections:
                    if outlier_method == 'auto_mixture':
                        candidate_scores = local_residual_distances[reject_candidates]
                    else:
                        candidate_scores = scores[reject_candidates]
                    candidate_scores = np.where(
                        np.isfinite(candidate_scores),
                        candidate_scores,
                        np.inf,
                    )
                    reject_candidates = reject_candidates[
                        np.argsort(candidate_scores)[-available_rejections:]
                    ]

                candidate_keep = keep.copy()
                candidate_keep[np.flatnonzero(keep)[reject_candidates]] = False
                if np.array_equal(candidate_keep, keep):
                    break
                keep = candidate_keep

            return keep, residual_distances, probabilities, model_info

        ref_centers_was_none = ref_centers is None
        if ref_centers is None and centers is None:
            raise ValueError("Either 'ref_centers' or 'centers' must be defined.")

        if peak_units not in ('pixels', 'calibrated'):
            raise ValueError("peak_units must be 'pixels' or 'calibrated'.")
        if peak_units == 'calibrated':
            if self.reciprocal_units is None or self.reciprocal_conv_factor is None:
                raise ValueError(
                    "peak_units='calibrated' requires reciprocal_units and "
                    "reciprocal_conv_factor."
                )
            if centers is None:
                raise ValueError(
                    "peak_units='calibrated' requires supplied centers; "
                    "automatic get_centers returns pixel coordinates."
                )
        try:
            max_condition_number = float(max_condition_number)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_condition_number must be finite and > 1.") from exc
        if not np.isfinite(max_condition_number) or max_condition_number <= 1:
            raise ValueError("max_condition_number must be finite and > 1.")

        if intensity_percentile is not None:
            if intensity_array is None:
                raise ValueError("intensity_percentile requires intensity_array.")
            try:
                intensity_percentile = float(intensity_percentile)
            except (TypeError, ValueError):
                raise ValueError("intensity_percentile must be a number.")
            if not 0 <= intensity_percentile <= 100:
                raise ValueError("intensity_percentile must be between 0 and 100.")

        if not isinstance(intensity_clip, str):
            raise ValueError("intensity_clip must be 'both', 'lower', or 'upper'.")
        intensity_clip = intensity_clip.strip().lower()
        if intensity_clip not in ('both', 'lower', 'upper'):
            raise ValueError("intensity_clip must be 'both', 'lower', or 'upper'.")

        reject_peak_outliers = bool(reject_peak_outliers)
        if not isinstance(outlier_method, str):
            raise ValueError(
                "outlier_method must be 'auto_mixture' or 'robust_zscore'."
            )
        outlier_method = outlier_method.strip().lower().replace('-', '_')
        if outlier_method in ('mixture', 'gmm', 'auto_gmm'):
            outlier_method = 'auto_mixture'
        if outlier_method in ('zscore', 'z_score', 'robust_z_score'):
            outlier_method = 'robust_zscore'
        if outlier_method not in ('auto_mixture', 'robust_zscore'):
            raise ValueError(
                "outlier_method must be 'auto_mixture' or 'robust_zscore'."
            )

        try:
            outlier_bic_delta = float(outlier_bic_delta)
        except (TypeError, ValueError):
            raise ValueError("outlier_bic_delta must be a number.")
        if not np.isfinite(outlier_bic_delta) or outlier_bic_delta < 0:
            raise ValueError("outlier_bic_delta must be non-negative and finite.")

        if outlier_method == 'robust_zscore':
            try:
                outlier_threshold = float(outlier_threshold)
            except (TypeError, ValueError):
                raise ValueError("outlier_threshold must be a number.")
            if not np.isfinite(outlier_threshold) or outlier_threshold <= 0:
                raise ValueError("outlier_threshold must be positive and finite.")

        if (
            outlier_min_peak_pairs is not None
            and outlier_min_peak_fraction is not None
        ):
            raise ValueError(
                "Define either outlier_min_peak_pairs or "
                "outlier_min_peak_fraction, not both."
            )

        if outlier_min_peak_fraction is not None:
            try:
                outlier_min_peak_fraction = float(outlier_min_peak_fraction)
            except (TypeError, ValueError):
                raise ValueError("outlier_min_peak_fraction must be a number.")
            if not 0 < outlier_min_peak_fraction <= 1:
                raise ValueError(
                    "outlier_min_peak_fraction must be greater than 0 and "
                    "less than or equal to 1."
                )
            outlier_min_peak_mode = 'fraction'
        else:
            if outlier_min_peak_pairs is None:
                outlier_min_peak_pairs = 4
            outlier_min_peak_pairs = int(outlier_min_peak_pairs)
            if outlier_min_peak_pairs < 2:
                raise ValueError("outlier_min_peak_pairs must be at least 2.")
            outlier_min_peak_mode = 'absolute'

        outlier_max_iterations = int(outlier_max_iterations)
        if outlier_max_iterations < 0:
            raise ValueError("outlier_max_iterations must be non-negative.")

        if not isinstance(outlier_space, str):
            raise ValueError(
                "outlier_space must be 'radial_angular', 'cartesian', or 'distance'."
            )
        outlier_space = outlier_space.strip().lower().replace('-', '_')
        if outlier_space not in ('radial_angular', 'cartesian', 'distance'):
            raise ValueError(
                "outlier_space must be 'radial_angular', 'cartesian', or 'distance'."
            )

        if centers is None:
            centers = self.get_centers(
                r=r_CoM,
                ref_coords=_reference_for_center_finding(ref_centers),
                method='CoM',
                real_mask=real_mask,
            )

        ydim, xdim, squeezed_scan = _infer_scan_shape(centers)
        if real_mask is not None:
            if self.ndim != 4 or squeezed_scan:
                raise ValueError(
                    "real_mask is only supported for 4D scan-shaped centers."
                )
            real_mask = np.asarray(real_mask)
            if real_mask.shape != (ydim, xdim):
                raise ValueError(
                    f"real_mask must have shape (Ry, Rx) = {(ydim, xdim)}, "
                    f"got {real_mask.shape}."
                )

        if center is None:
            center = _metadata_strain_center()
            if center is not None:
                center_source = 'center_beam_metadata.mean_fit_center_px'
            else:
                center_source = 'centers' if ref_centers_was_none else 'ref_centers'
                center_data = centers if ref_centers_was_none else ref_centers
                center = _mean_peak_coordinate(center_data, mask=real_mask)
        else:
            center_source = 'user'
        center = _validate_strain_center(center)

        if ref_centers is None:
            if isinstance(centers, list):
                raise ValueError(
                    "ref_centers cannot be inferred from ragged centers. "
                    "Provide a global or local reference peak set."
                )
            if r_inner is None or r_outer is None:
                r = (self.shape[-2] + self.shape[-1]) / 4
                reduced_data = self.apply_mask(r_inner=0.6 * r, r_outer=0.8 * r)
            else:
                reduced_data = self.apply_mask(r_inner, r_outer)
            mean, flat_mask = mask_and_average(reduced_data.array, return_mask=True, 
                                               function='sum_2d', threshold='upper', 
                                               percentile=5, show_mask=False)
            reference_mask = flat_mask
            if real_mask is not None:
                reference_mask = reference_mask & real_mask
            if not np.any(reference_mask):
                raise ValueError(
                    "No real-space positions remain for reference-center "
                    "inference after applying real_mask."
                )
            ref_centers = np.mean(centers[reference_mask], axis=0)

        output_shape = (ydim,) if squeezed_scan else (ydim, xdim)
        ang_rad = np.radians(ang)
        R1 = np.array([[np.cos(ang_rad), np.sin(ang_rad)],
                       [-np.sin(ang_rad), np.cos(ang_rad)]])

        exx = np.full(output_shape, np.nan)
        eyy = np.full(output_shape, np.nan)
        exy = np.full(output_shape, np.nan)
        erot = np.full(output_shape, np.nan)
        fit_rmse = np.full(output_shape, np.nan)
        relative_fit_rmse = np.full(output_shape, np.nan)
        transforms = np.full(output_shape + (2, 2), np.nan)
        translations = np.full(output_shape + (2,), np.nan)
        match_counts = np.zeros(output_shape, dtype=int)
        initial_match_counts = np.zeros(output_shape, dtype=int)
        outlier_counts = np.zeros(output_shape, dtype=int)
        resolved_outlier_min_peak_counts = np.zeros(output_shape, dtype=int)
        peak_outlier_masks = None
        outlier_ref_indices = None
        outlier_measured_indices = None
        peak_residuals = None
        outlier_probabilities = None
        outlier_model_info = None
        if reject_peak_outliers:
            peak_outlier_masks = np.empty(output_shape, dtype=object)
            outlier_ref_indices = np.empty(output_shape, dtype=object)
            outlier_measured_indices = np.empty(output_shape, dtype=object)
            peak_residuals = np.empty(output_shape, dtype=object)
            outlier_probabilities = np.empty(output_shape, dtype=object)
            outlier_model_info = np.empty(output_shape, dtype=object)
            peak_outlier_masks.fill(None)
            outlier_ref_indices.fill(None)
            outlier_measured_indices.fill(None)
            peak_residuals.fill(None)
            outlier_probabilities.fill(None)
            outlier_model_info.fill(None)

        for i in tqdm(range(ydim), desc='Computing strain matrices'):
            for j in range(xdim):
                if real_mask is not None and not real_mask[i, j]:
                    continue

                ref_ij = _local_peaks(ref_centers, i, j, squeezed_scan)
                centers_ij = _local_peaks(centers, i, j, squeezed_scan)
                weights_ij = _local_weights(intensity_array, i, j, squeezed_scan)
                reference_peak_count = _clean_peak_set(ref_ij)[0].shape[0]
                local_outlier_min_peak_pairs = _resolve_outlier_min_peak_pairs(
                    reference_peak_count
                )

                matched = _match_peak_sets(ref_ij, centers_ij, weights_ij)
                matched_ref, matched_centers, weights, ref_idx, meas_idx = matched
                if matched_ref is None or matched_ref.shape[0] < min_peak_pairs:
                    continue

                out_idx = i if squeezed_scan else (i, j)
                initial_match_counts[out_idx] = matched_ref.shape[0]
                resolved_outlier_min_peak_counts[out_idx] = (
                    local_outlier_min_peak_pairs
                )

                try:
                    (
                        peak_keep,
                        residual_distances,
                        outlier_probability,
                        local_outlier_model_info,
                    ) = _reject_peak_outliers(
                        matched_ref,
                        matched_centers,
                        weights,
                        center,
                        local_outlier_min_peak_pairs,
                    )
                    if np.count_nonzero(peak_keep) < min_peak_pairs:
                        continue

                    fit_ref = matched_ref[peak_keep]
                    fit_centers = matched_centers[peak_keep]
                    fit_weights = weights[peak_keep] if weights is not None else None

                    transform_matrix, translation = _fit_peak_transform(
                        fit_ref,
                        fit_centers,
                        center,
                        weights=fit_weights,
                    )

                    if ewpc:
                        transform_for_strain = np.linalg.inv(transform_matrix)
                    else:
                        transform_for_strain = transform_matrix

                    T = R1 @ transform_for_strain @ np.linalg.inv(R1)
                    R, U = polar(T)
                    rmse, relative_rmse = _fit_residual_quality(
                        fit_ref, fit_centers, transform_matrix, translation,
                        center, fit_weights,
                    )
                except np.linalg.LinAlgError:
                    continue

                rejected_mask = ~peak_keep
                rejected_count = int(np.count_nonzero(rejected_mask))
                eyy[out_idx] = 1 - U[0, 0]
                exx[out_idx] = 1 - U[1, 1]
                exy[out_idx] = U[1, 0]
                erot[out_idx] = np.arctan2(R[1, 0], R[0, 0])
                fit_rmse[out_idx] = rmse
                relative_fit_rmse[out_idx] = relative_rmse
                transforms[out_idx] = transform_matrix
                translations[out_idx] = translation
                match_counts[out_idx] = fit_ref.shape[0]
                outlier_counts[out_idx] = rejected_count
                if reject_peak_outliers:
                    peak_outlier_masks[out_idx] = rejected_mask
                    outlier_ref_indices[out_idx] = ref_idx[rejected_mask]
                    outlier_measured_indices[out_idx] = meas_idx[rejected_mask]
                    peak_residuals[out_idx] = residual_distances
                    outlier_probabilities[out_idx] = outlier_probability
                    outlier_model_info[out_idx] = local_outlier_model_info

        diagnostics = None
        if return_transform:
            diagnostics = {
                'transforms': transforms,
                'translations': translations,
                'match_counts': match_counts,
                'initial_match_counts': initial_match_counts,
                'kept_peak_counts': match_counts,
                'outlier_counts': outlier_counts,
                'fit_rmse': fit_rmse,
                'relative_fit_rmse': relative_fit_rmse,
                'match_peaks': match_peaks,
                'fit_translation': fit_translation,
                'center': np.array(center, copy=True),
                'center_source': center_source,
                'peak_units': peak_units,
                'max_condition_number': max_condition_number,
                'min_peak_pairs': min_peak_pairs,
                'intensity_percentile': intensity_percentile,
                'intensity_clip': intensity_clip,
                'reject_peak_outliers': reject_peak_outliers,
                'outlier_threshold': outlier_threshold,
                'outlier_min_peak_pairs': outlier_min_peak_pairs,
                'outlier_min_peak_fraction': outlier_min_peak_fraction,
                'outlier_min_peak_mode': outlier_min_peak_mode,
                'resolved_outlier_min_peak_counts': resolved_outlier_min_peak_counts,
                'outlier_max_iterations': outlier_max_iterations,
                'outlier_space': outlier_space,
                'outlier_method': outlier_method,
                'outlier_bic_delta': outlier_bic_delta,
                'peak_outlier_masks': peak_outlier_masks,
                'outlier_ref_indices': outlier_ref_indices,
                'outlier_measured_indices': outlier_measured_indices,
                'peak_residuals': peak_residuals,
                'outlier_probabilities': outlier_probabilities,
                'outlier_model_info': outlier_model_info,
                'real_mask': (
                    np.array(real_mask, copy=True)
                    if real_mask is not None
                    else None
                ),
                'g_vector': g_vector,
            }

        calibrated_scan = self.real_shape == output_shape
        return StrainResult(
            exx=exx,
            eyy=eyy,
            exy=exy,
            erot=erot,
            fit_rmse=fit_rmse,
            relative_fit_rmse=relative_fit_rmse,
            match_counts=match_counts,
            initial_match_counts=initial_match_counts,
            outlier_counts=outlier_counts,
            peak_origin=tuple(float(value) for value in center),
            peak_origin_source=center_source,
            basis_angle_deg=float(ang),
            ewpc=bool(ewpc),
            peak_units=(
                self.reciprocal_units if peak_units == 'calibrated' else 'pixels'
            ),
            real_units=self.real_units if calibrated_scan else None,
            real_conv_factor=(
                self.real_conv_factor if calibrated_scan else None
            ),
            real_origin=(
                self.real_origin if calibrated_scan else (0.0, 0.0)
            ),
            diagnostics=diagnostics,
        )
        
        
    def apply_mask(self, r_inner=None, r_outer=None, mask=None, domain=None):
        """
        Apply either a real-space selection mask or a reciprocal-space mask.

        Parameters
        ----------
        r_inner : float or None, optional
            Inner radius of a reciprocal-space mask centered on the diffraction
            pattern midpoint. If ``r_outer`` is omitted, ``r_inner`` defines a
            filled circular mask. If both are provided, they define an annulus.
        r_outer : float or None, optional
            Outer radius of an annular reciprocal-space mask.
        mask : ndarray of bool, optional
            Explicit mask to apply. For ``domain='real'`` this must match the
            real-space scan shape ``(Ny, Nx)`` of a 4D dataset and the selected
            scan positions are returned as a 3D stack. For
            ``domain='reciprocal'`` this must match the diffraction-pattern
            shape ``(ky, kx)`` and is broadcast across all scan positions.
        domain : {'real', 'reciprocal'} or None, optional
            Domain of an explicit ``mask``. Ignored when using radial masks.

        Returns
        -------
        HyperData
            A new ``HyperData`` instance containing the masked data. Real-space
            masking returns only the selected diffraction patterns, while
            reciprocal-space masking preserves the input dimensionality.

        Notes
        -----
        This method never mutates ``self.array`` in place.
        """

        if self.ndim < 3:
            raise ValueError("The data object must be 3-dimensional or greater.")

        data = self.array
        ky, kx = self.shape[-2], self.shape[-1]

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)

            if domain == 'real':
                if self.ndim != 4:
                    raise ValueError("Real-space masking requires a 4D dataset.")

                expected_shape = self.shape[:2]
                if mask.shape != expected_shape:
                    raise ValueError(
                        f"Real-space mask must have shape {expected_shape}, "
                        f"got {mask.shape}."
                    )
                if not np.any(mask):
                    raise ValueError("Real-space mask contains no True pixels.")

                return self._spawn(
                    data[mask],
                    real_units=None,
                    real_conv_factor=None,
                )

            if domain == 'reciprocal':
                expected_shape = (ky, kx)
                if mask.shape != expected_shape:
                    raise ValueError(
                        f"Reciprocal-space mask must have shape {expected_shape}, "
                        f"got {mask.shape}."
                    )

                return self._spawn(data * mask, preserve_unfold=True)

            raise ValueError(
                "When 'mask' is provided, 'domain' must be either 'real' or "
                "'reciprocal'."
            )

        if r_outer is not None:
            if r_inner is None:
                raise ValueError("'r_inner' must be provided when 'r_outer' is used.")
            bool_mask = make_mask(
                ((ky - 1) / 2, (kx - 1) / 2),
                (r_inner, r_outer),
                mask_dim=(ky, kx),
            )
        else:
            if r_inner is None:
                raise ValueError(
                    "Provide either an explicit 'mask' or a reciprocal-space "
                    "radius via 'r_inner'."
                )
            bool_mask = make_mask(
                ((ky - 1) / 2, (kx - 1) / 2),
                r_inner,
                mask_dim=(ky, kx),
            )

        return self._spawn(data * bool_mask, preserve_unfold=True)
    
    @staticmethod
    def _majority_filter_labels(labels, size):
        """Smooth categorical labels without treating their IDs as ordered values."""
        kernel = np.ones((size, size), dtype=np.int32)
        result = labels.copy()
        best_count = np.full(labels.shape, -1, dtype=np.int32)
        for label_value in np.unique(labels):
            counts = ndimage.convolve(
                (labels == label_value).astype(np.int32), kernel, mode='reflect',
            )
            # Preserve the original label at a tied pixel; otherwise use the
            # smallest label for a deterministic tie break.
            replace = (counts > best_count) | (
                (counts == best_count) & (labels == label_value)
            )
            result[replace] = label_value
            best_count[replace] = counts[replace]
        return result

    def get_clusters(self, n_PCAcomponents=None, n_clusters=None, r_centerBeam=None,
                     std_Threshold=0.2, power=1, clustering_method="k-means",
                     plotStdMask=False, plotScree=False, plotClusterMap=False,
                     plot3dClusterMap=False, filter_size=None, cluster_cmap=None,
                     filter_iterations=1, outer_ring=None, polar=None,
                     split_disconnected=False, *, detector_units='pixels',
                     detector_mask=None, intensity_transform='log',
                     normalize='none', pca_fit_samples=None, random_state=0,
                     batch_size=None, return_diagnostics=False,
                     include_mean_dps=False, split_connectivity=2,
                     max_plot_points=10000, n_components=None, fit_samples=None,
                     reduction_method='pca', feature_weighting='hard',
                     soft_weight_scale=None, gmm_covariance_type='diag',
                     hdbscan_min_cluster_size=20, hdbscan_min_samples=None,
                     nmf_epochs=3):
        """Cluster a 4D scan from weighted detector pixels and reduced features.

        The return value is a ``(Ry, Rx)`` integer label array. Set
        ``return_diagnostics=True`` for ``(labels, details)``. No intermediate
        arrays or model are retained on this object.

        Parameters
        ----------
        n_components : int or None
            Number of PCA or NMF components. Use this name for new calls;
            ``n_PCAcomponents`` remains accepted by existing notebooks.
        n_clusters : int or None
            Requested cluster count. Must be None for HDBSCAN, which selects
            its own number of clusters and may label outliers as -1.
        r_centerBeam, outer_ring : float or None
            Inner exclusion radius and optional outer limit. ``None`` means
            no inner/outer limit. Values use ``detector_units``.
        detector_units : {'pixels', 'auto', 'calibrated'}
            Units for both radii. Pixels remain the default for existing
            notebook calls. On polar data, pixels mean *source Cartesian*
            diffraction pixels, not radial row indices.
        detector_mask : ndarray of bool or None
            Optional ``(Ky, Kx)`` or ``(Kr, Ktheta)`` mask intersected with
            the radial selection.
        std_Threshold : float
            With ``feature_weighting='hard'``, keep pixels whose transformed-
            data standard deviation is at least this fraction of the maximum.
        feature_weighting : {'hard', 'soft', 'none'}
            Hard uses the previous Boolean threshold. Soft retains every
            varying pixel in the detector selection and assigns weight
            ``std / (std + soft_weight_scale)``. None retains all selected
            pixels with unit weight. Physical exclusions still use the hard
            ``detector_mask`` and radial limits.
        soft_weight_scale : positive float or None
            Scale in transformed-data standard-deviation units. None uses the
            75th percentile of positive pixel deviations, avoiding dependence
            on a single unusually variable detector pixel. Only for soft mode.
        intensity_transform : {'log', 'log1p', 'sqrt', 'raw'}
            Transform nonnegative detector intensities before feature selection
            and decomposition. ``'log'`` uses ``log(max(I, 1))``; ``'raw'`` leaves
            intensities unchanged. Negative inputs are clipped to zero.
        power : positive float
            Exponent applied *after* ``intensity_transform``. Thus ``power=2``
            genuinely emphasizes higher transformed intensities. Use
            ``intensity_transform='raw', power=2`` for squared raw counts.
        normalize : {'none', 'total'}
            ``'total'`` divides each pattern by its mean nonnegative signal
            over the radial detector region before transforming it. This
            removes overall brightness variation but preserves relative spots.
        reduction_method : {'pca', 'nmf'}
            PCA uses incremental fitting. NMF uses mini-batch nonnegative
            factorization and can expose its detector components in diagnostics.
            ``plotScree`` applies only to PCA.
        nmf_epochs : positive int
            Number of shuffled training passes for NMF. More passes may
            improve the factorization but read the data more times.
        clustering_method : {'k-means', 'mini-batch-k-means', 'hierarchical',
                             'gaussian-mixture', 'hdbscan'}
            Mini-batch k-means is useful for large scans. Hierarchical
            clustering remains limited to 5000 patterns due to quadratic RAM.
            Gaussian mixtures provide probabilistic assignments; HDBSCAN
            finds variable-density groups but may take longer on large scans.
        gmm_covariance_type : {'diag', 'spherical', 'tied', 'full'}
            Gaussian-mixture covariance model. Diagonal is the less costly
            default for high-dimensional PCA or NMF scores.
        hdbscan_min_cluster_size, hdbscan_min_samples : int or None
            HDBSCAN density controls. The former defaults to 20; None for the
            latter lets HDBSCAN use its own default.
        fit_samples : int, float in (0, 1], or None
            Number or fraction of randomly sampled patterns used to fit the
            reducer. ``pca_fit_samples`` is accepted for existing notebooks.
            Every pattern is still transformed and clustered. None fits all.
        random_state : int or None
            Seed for sampling, NMF, and clustering; 0 is reproducible.
        batch_size : int or None
            Patterns per data batch. None chooses a bounded size automatically.
        filter_size : odd positive int or None
            Opt-in categorical majority-filter window; None disables smoothing.
        filter_iterations : nonnegative int
            Number of majority-filter passes when ``filter_size`` is set.
        split_disconnected : bool
            Give disconnected islands distinct labels after smoothing.
        split_connectivity : {1, 2}
            4- or 8-connected neighborhoods for splitting; 8 is the default.
        plotStdMask, plotScree, plotClusterMap, plot3dClusterMap : bool
            Optional plots, disabled by default to avoid accumulating notebook
            outputs. ``max_plot_points`` caps the 3D scatter size.
        cluster_cmap : str or None
            Cluster-map colormap; defaults to ``'gnuplot'``.
        polar : bool or None
            Optional consistency check for existing calls. Geometry always
            follows ``self.is_polar`` and its metadata.
        return_diagnostics : bool
            Also return selected detector pixels, feature weights, cluster
            sizes, and method settings. NMF components are in weighted feature
            space. Gaussian-mixture confidence is before spatial filtering.
        include_mean_dps : bool
            With diagnostics, also compute raw mean diffraction patterns for
            the final clusters. This requires another pass over the 4D data.

        Notes
        -----
        Soft weights multiply detector intensities by ``sqrt(weight)`` before
        decomposition, so squared Euclidean distances weight each pixel by
        ``weight``. PCA and k-means remain the defaults for existing calls.
        """
        if self.ndim != 4:
            raise ValueError("get_clusters requires a 4D dataset.")
        if n_PCAcomponents is not None and n_components is not None:
            raise ValueError("Specify only one of n_components and n_PCAcomponents.")
        n_components = n_PCAcomponents if n_components is None else n_components
        if isinstance(n_components, (bool, np.bool_)) or not isinstance(
            n_components, Integral
        ) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if clustering_method not in (
            'k-means', 'mini-batch-k-means', 'hierarchical',
            'gaussian-mixture', 'hdbscan',
        ):
            raise ValueError("Unsupported clustering_method.")
        if clustering_method == 'hdbscan':
            if n_clusters is not None:
                raise ValueError("n_clusters must be None for HDBSCAN.")
        elif isinstance(n_clusters, (bool, np.bool_)) or not isinstance(
            n_clusters, Integral
        ) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if reduction_method not in ('pca', 'nmf'):
            raise ValueError("reduction_method must be 'pca' or 'nmf'.")
        if isinstance(nmf_epochs, (bool, np.bool_)) or not isinstance(
            nmf_epochs, Integral
        ) or nmf_epochs < 1:
            raise ValueError("nmf_epochs must be a positive integer.")
        if feature_weighting not in ('hard', 'soft', 'none'):
            raise ValueError("feature_weighting must be 'hard', 'soft', or 'none'.")
        if soft_weight_scale is not None and (
            feature_weighting != 'soft' or not np.isfinite(soft_weight_scale)
            or soft_weight_scale <= 0
        ):
            raise ValueError("soft_weight_scale requires soft weighting and must be positive.")
        if gmm_covariance_type not in ('diag', 'spherical', 'tied', 'full'):
            raise ValueError("Unsupported gmm_covariance_type.")
        if isinstance(hdbscan_min_cluster_size, (bool, np.bool_)) or not isinstance(
            hdbscan_min_cluster_size, Integral
        ) or hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size must be an integer >= 2.")
        if hdbscan_min_samples is not None and (
            isinstance(hdbscan_min_samples, (bool, np.bool_))
            or not isinstance(hdbscan_min_samples, Integral)
            or hdbscan_min_samples < 1
        ):
            raise ValueError("hdbscan_min_samples must be a positive integer or None.")
        if plotScree and reduction_method != 'pca':
            raise ValueError("plotScree is only defined for PCA.")
        if not np.isfinite(power) or power <= 0:
            raise ValueError("power must be a positive finite number.")
        if not np.isfinite(std_Threshold) or not 0 <= std_Threshold <= 1:
            raise ValueError("std_Threshold must be between 0 and 1.")
        if r_centerBeam is not None and (
            not np.isfinite(r_centerBeam) or r_centerBeam < 0
        ):
            raise ValueError("r_centerBeam must be nonnegative and finite.")
        if outer_ring is not None and (
            not np.isfinite(outer_ring)
            or outer_ring <= (0 if r_centerBeam is None else r_centerBeam)
        ):
            raise ValueError("outer_ring must be greater than r_centerBeam.")
        if polar is not None and (
            not isinstance(polar, (bool, np.bool_)) or polar != self.is_polar
        ):
            raise ValueError(
                "polar must agree with the object's polar metadata. "
                "The radial geometry is inferred from self.is_polar."
            )
        if intensity_transform not in ('log', 'log1p', 'sqrt', 'raw'):
            raise ValueError("intensity_transform must be 'log', 'log1p', 'sqrt', or 'raw'.")
        if normalize not in ('none', 'total'):
            raise ValueError("normalize must be 'none' or 'total'.")
        if filter_size is not None and (
            isinstance(filter_size, (bool, np.bool_))
            or not isinstance(filter_size, Integral)
            or filter_size < 1 or filter_size % 2 != 1
        ):
            raise ValueError("filter_size must be an odd positive integer or None.")
        if isinstance(filter_iterations, (bool, np.bool_)) or not isinstance(
            filter_iterations, Integral
        ) or filter_iterations < 0:
            raise ValueError("filter_iterations must be a nonnegative integer.")
        if split_connectivity not in (1, 2):
            raise ValueError("split_connectivity must be 1 (4-way) or 2 (8-way).")
        if include_mean_dps and not return_diagnostics:
            raise ValueError("include_mean_dps requires return_diagnostics=True.")
        if isinstance(max_plot_points, (bool, np.bool_)) or not isinstance(
            max_plot_points, Integral
        ) or max_plot_points < 1:
            raise ValueError("max_plot_points must be a positive integer.")

        A, B, C, D = self.shape
        n_patterns = A * B
        if n_components > n_patterns:
            raise ValueError("n_components exceeds the number of patterns.")
        if n_clusters is not None and n_clusters > n_patterns:
            raise ValueError("n_clusters exceeds the number of patterns.")
        if clustering_method == 'hdbscan' and hdbscan_min_cluster_size > n_patterns:
            raise ValueError("hdbscan_min_cluster_size exceeds the number of patterns.")
        if plot3dClusterMap and n_components < 3:
            raise ValueError("plot3dClusterMap requires at least 3 components.")
        if clustering_method == 'hierarchical' and n_patterns > 5000:
            raise ValueError(
                "Hierarchical clustering uses quadratic memory and is limited "
                "to 5000 patterns. Use 'k-means' or 'mini-batch-k-means'."
            )
        if isinstance(random_state, (bool, np.bool_)) or (
            random_state is not None and not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None.")
        rng = np.random.default_rng(random_state)

        if pca_fit_samples is not None and fit_samples is not None:
            raise ValueError("Specify only one of fit_samples and pca_fit_samples.")
        fit_samples = pca_fit_samples if fit_samples is None else fit_samples
        if fit_samples is None:
            n_fit_patterns = n_patterns
        elif isinstance(fit_samples, (bool, np.bool_)):
            raise ValueError("fit_samples must be an integer count or fraction.")
        elif isinstance(fit_samples, Integral):
            n_fit_patterns = int(fit_samples)
        elif isinstance(fit_samples, (float, np.floating)) and (
            np.isfinite(fit_samples) and 0 < fit_samples <= 1
        ):
            n_fit_patterns = int(np.ceil(n_patterns * fit_samples))
        else:
            raise ValueError("fit_samples must be an integer count or fraction in (0, 1].")
        if not n_components <= n_fit_patterns <= n_patterns:
            raise ValueError(
                "fit_samples must select between n_components and "
                "the total number of patterns."
            )

        if self.is_polar:
            _, detector_factor, detector_mode = self._resolve_polar_radius_units(
                detector_units
            )
            radius_range = (self.polar_metadata or {}).get(
                'radius_range_pixels', (0.0, float(C - 1)),
            )
            if len(radius_range) != 2 or not np.all(np.isfinite(radius_range)):
                raise ValueError("polar_metadata radius_range_pixels is invalid.")
            radial_values = np.linspace(*radius_range, C)[:, None]
            radial_values = np.broadcast_to(radial_values, (C, D))
        else:
            _, detector_factor, detector_mode = self._resolve_reciprocal_detector_units(
                detector_units
            )
            yy, xx = np.ogrid[:C, :D]
            radial_values = np.hypot(yy - (C - 1) / 2, xx - (D - 1) / 2)
        factor = detector_factor if detector_mode == 'calibrated' else 1.0
        detector_selection = np.ones((C, D), dtype=bool)
        if r_centerBeam is not None:
            detector_selection &= radial_values > r_centerBeam / factor
        if outer_ring is not None:
            detector_selection &= radial_values <= outer_ring / factor
        if detector_mask is not None:
            input_mask = np.asarray(detector_mask)
            if input_mask.shape != (C, D) or input_mask.dtype != np.bool_:
                raise ValueError(f"detector_mask must be Boolean with shape {(C, D)}.")
            detector_selection &= input_mask

        candidate_y, candidate_x = np.nonzero(detector_selection)
        n_candidate_features = len(candidate_y)
        if n_candidate_features == 0:
            raise ValueError("The detector selection contains no pixels.")
        if n_candidate_features < n_components:
            raise ValueError("Fewer detector pixels than components were selected.")

        batch_target_bytes = 32 * 1024 * 1024
        if batch_size is None:
            batch_size = max(
                n_components,
                min(512, max(1, batch_target_bytes // (4 * n_candidate_features))),
            )
        elif isinstance(batch_size, (bool, np.bool_)) or not isinstance(
            batch_size, Integral
        ) or batch_size < n_components:
            raise ValueError("batch_size must be an integer >= n_components.")

        def batch_edges(count, require_component_size=False):
            n_batches = max(1, int(np.ceil(count / batch_size)))
            if require_component_size:
                n_batches = min(n_batches, count // n_components)
            return np.linspace(0, count, n_batches + 1, dtype=int)

        def load_batch(scan_indices, kept_columns=None, feature_scales=None):
            # Only selected detector pixels are copied; normalization always
            # uses the same candidate region on every processing pass.
            batch = np.asarray(
                self.array[
                    (scan_indices // B)[:, None],
                    (scan_indices % B)[:, None],
                    candidate_y[None, :],
                    candidate_x[None, :],
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(batch)):
                raise ValueError("Clustering data contains nonfinite values.")
            np.maximum(batch, 0, out=batch)
            if normalize == 'total':
                mean_intensity = batch.sum(axis=1, keepdims=True, dtype=np.float64)
                mean_intensity /= n_candidate_features
                mean_intensity[mean_intensity == 0] = 1
                batch /= mean_intensity.astype(np.float32)
            if intensity_transform == 'log':
                np.maximum(batch, 1, out=batch)
                np.log(batch, out=batch)
            elif intensity_transform == 'log1p':
                np.log1p(batch, out=batch)
            elif intensity_transform == 'sqrt':
                np.sqrt(batch, out=batch)
            if power != 1:
                np.power(batch, power, out=batch)
            if not np.all(np.isfinite(batch)):
                raise ValueError("The chosen transform produced nonfinite values.")
            if kept_columns is None:
                return batch
            batch = batch[:, kept_columns]
            if feature_scales is not None:
                batch *= feature_scales
            return batch

        sums = np.zeros(n_candidate_features, dtype=np.float64)
        squared_sums = np.zeros(n_candidate_features, dtype=np.float64)
        all_edges = batch_edges(n_patterns)
        for start, stop in zip(all_edges[:-1], all_edges[1:]):
            batch = load_batch(np.arange(start, stop))
            sums += batch.sum(axis=0, dtype=np.float64)
            squared_sums += np.einsum('ij,ij->j', batch, batch, dtype=np.float64)

        means = sums / n_patterns
        variances = np.maximum(squared_sums / n_patterns - means**2, 0)
        deviations = np.sqrt(variances)
        resolved_weight_scale = None
        if feature_weighting == 'hard':
            keep_features = (deviations > 0) & (
                deviations >= std_Threshold * deviations.max()
            )
            selected_weights = np.ones(int(keep_features.sum()), dtype=np.float32)
        elif feature_weighting == 'soft':
            keep_features = deviations > 0
            if not np.any(keep_features):
                raise ValueError("No varying detector pixels remain for soft weighting.")
            resolved_weight_scale = (
                float(np.percentile(deviations[keep_features], 75))
                if soft_weight_scale is None else float(soft_weight_scale)
            )
            selected_deviations = deviations[keep_features]
            selected_weights = (
                selected_deviations / (selected_deviations + resolved_weight_scale)
            ).astype(np.float32)
        else:
            keep_features = np.ones(n_candidate_features, dtype=bool)
            selected_weights = np.ones(n_candidate_features, dtype=np.float32)

        feature_mask = np.zeros((C, D), dtype=bool)
        feature_mask[candidate_y[keep_features], candidate_x[keep_features]] = True
        feature_weights = np.zeros((C, D), dtype=np.float32)
        feature_weights[feature_mask] = selected_weights
        feature_scales = np.sqrt(selected_weights)
        n_features = int(keep_features.sum())
        if n_features < n_components:
            raise ValueError(
                f"Only {n_features} detector pixels remain; "
                f"n_components={n_components} is too large."
            )
        if plotStdMask:
            fig, ax = plt.subplots()
            ax.imshow(feature_weights, vmin=0, vmax=1)
            ax.set_title(f'Detector feature weights ({feature_weighting})')
            ax.set_axis_off()
            plt.show()
            plt.close(fig)

        fit_indices = (
            np.arange(n_patterns)
            if n_fit_patterns == n_patterns
            else np.sort(rng.choice(n_patterns, n_fit_patterns, replace=False))
        )
        if reduction_method == 'nmf':
            reducer = MiniBatchNMF(
                n_components=n_components, init='nndsvda',
                batch_size=batch_size, random_state=random_state,
            )
        else:
            reducer = IncrementalPCA(n_components=n_components)
        fit_edges = batch_edges(n_fit_patterns, require_component_size=True)
        n_fit_passes = nmf_epochs if reduction_method == 'nmf' else 1
        for _ in range(n_fit_passes):
            pass_indices = (
                rng.permutation(fit_indices)
                if reduction_method == 'nmf' else fit_indices
            )
            for start, stop in zip(fit_edges[:-1], fit_edges[1:]):
                batch = load_batch(
                    pass_indices[start:stop], keep_features, feature_scales,
                )
                reducer.partial_fit(batch)

        data_reduced = np.empty((n_patterns, n_components), dtype=np.float32)
        for start, stop in zip(all_edges[:-1], all_edges[1:]):
            batch = load_batch(
                np.arange(start, stop), keep_features, feature_scales,
            )
            data_reduced[start:stop] = reducer.transform(batch)

        if plotScree:
            fig, ax = plt.subplots()
            ax.plot(
                range(1, n_components + 1),
                reducer.explained_variance_ratio_, marker='o',
            )
            ax.set_title("Scree Plot")
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Variance Explained")
            plt.show()
            plt.close(fig)
    
        if clustering_method == "k-means":
            clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state)
            with threadpool_limits(limits=1, user_api='openmp'):
                clusters = clustering_model.fit_predict(data_reduced)
        elif clustering_method == 'mini-batch-k-means':
            clustering_model = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=random_state,
                batch_size=max(1024, 3 * n_clusters),
            )
            with threadpool_limits(limits=1, user_api='openmp'):
                clusters = clustering_model.fit_predict(data_reduced)
        elif clustering_method == 'gaussian-mixture':
            clustering_model = GaussianMixture(
                n_components=n_clusters, covariance_type=gmm_covariance_type,
                random_state=random_state,
            )
            with threadpool_limits(limits=1, user_api='openmp'):
                clusters = clustering_model.fit_predict(data_reduced)
        elif clustering_method == 'hdbscan':
            clustering_model = HDBSCAN(
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples,
            )
            clusters = clustering_model.fit_predict(data_reduced)
        else:
            from scipy.cluster.hierarchy import linkage, fcluster
            linkage_matrix = linkage(data_reduced, method='ward')
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            clustering_model = None

        if plot3dClusterMap:
            plotted = (
                np.arange(n_patterns) if n_patterns <= max_plot_points
                else np.sort(rng.choice(n_patterns, max_plot_points, replace=False))
            )
            plotted_labels = np.unique(clusters)
            colormap = plt.get_cmap(cluster_cmap or 'gnuplot', len(plotted_labels))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, label_value in enumerate(plotted_labels):
                chosen = plotted[clusters[plotted] == label_value]
                if chosen.size == 0:
                    continue
                ax.scatter(
                    data_reduced[chosen, 0], data_reduced[chosen, 1],
                    data_reduced[chosen, 2],
                    c=['0.35' if label_value == -1 else colormap(i)],
                    label='Noise' if label_value == -1 else f'Cluster {label_value + 1}',
                    s=2, alpha=0.5,
                )
            plt.show()
            plt.close(fig)

        cluster_map = clusters.reshape(A, B)
        if filter_size is not None:
            for _ in range(filter_iterations):
                cluster_map = self._majority_filter_labels(cluster_map, filter_size)

        mapping = None
        if split_disconnected:
            cluster_map, mapping = split_disconnected_clusters(
                cluster_map, connectivity=split_connectivity, background=-1,
            )

        cluster_labels, cluster_sizes = np.unique(cluster_map, return_counts=True)
        if plotClusterMap:
            colormap = plt.get_cmap(cluster_cmap or 'gnuplot', len(cluster_labels))
            palette = (colormap(np.arange(len(cluster_labels)))[:, :3] * 255).astype(np.uint8)
            palette[cluster_labels == -1] = (80, 80, 80)
            cluster_map_colored = palette[np.searchsorted(cluster_labels, cluster_map)]
            fig, ax = plt.subplots()
            ax.imshow(cluster_map_colored)
            n_found_clusters = int(np.count_nonzero(cluster_labels != -1))
            ax.set_title(f"Cluster Map ({A}x{B}) with {n_found_clusters} Clusters")
            ax.set_axis_off()
            plt.show()
            plt.close(fig)

        if not return_diagnostics:
            return cluster_map

        diagnostics = {
            'detector_mask': detector_selection,
            'feature_mask': feature_mask,
            'feature_weights': feature_weights,
            'feature_count': n_features,
            'feature_weighting': feature_weighting,
            'soft_weight_scale': resolved_weight_scale,
            'reduction_method': reduction_method,
            'n_components': n_components,
            'nmf_epochs': nmf_epochs if reduction_method == 'nmf' else None,
            'explained_variance_ratio': (
                reducer.explained_variance_ratio_.copy()
                if reduction_method == 'pca' else None
            ),
            'reduction_fit_patterns': n_fit_patterns,
            'pca_fit_patterns': n_fit_patterns,
            'cluster_labels': cluster_labels,
            'cluster_sizes': cluster_sizes,
            'clustering_method': clustering_method,
            'intensity_transform': intensity_transform,
            'normalize': normalize,
            'power': power,
            'detector_units': detector_units,
            'r_centerBeam': r_centerBeam,
            'outer_ring': outer_ring,
            'inertia': (
                float(clustering_model.inertia_)
                if hasattr(clustering_model, 'inertia_') else None
            ),
        }
        if reduction_method == 'nmf':
            diagnostics['component_vectors'] = reducer.components_.astype(
                np.float32, copy=True,
            )
        if clustering_method == 'gaussian-mixture':
            confidence = np.empty(n_patterns, dtype=np.float32)
            for start, stop in zip(all_edges[:-1], all_edges[1:]):
                confidence[start:stop] = clustering_model.predict_proba(
                    data_reduced[start:stop]
                ).max(axis=1)
            diagnostics['model_confidence'] = confidence.reshape(A, B)
            diagnostics['gmm_covariance_type'] = gmm_covariance_type
        if clustering_method == 'hdbscan':
            diagnostics['noise_count'] = int(np.count_nonzero(clusters == -1))
        if mapping is not None:
            diagnostics['split_mapping'] = mapping

        if include_mean_dps:
            required_bytes = len(cluster_labels) * C * D * np.dtype(np.float64).itemsize
            if required_bytes > 512 * 1024 * 1024:
                raise ValueError(
                    "Mean diffraction patterns would require over 512 MiB. "
                    "Use fewer clusters or omit include_mean_dps."
                )
            raw_sums = np.zeros((len(cluster_labels), C, D), dtype=np.float64)
            final_labels = cluster_map.ravel()
            full_pattern_bytes = max(1, self.array.dtype.itemsize * C * D)
            mean_batch_size = max(
                1, min(batch_size, batch_target_bytes // full_pattern_bytes),
            )
            for start in range(0, n_patterns, mean_batch_size):
                stop = min(start + mean_batch_size, n_patterns)
                scan_indices = np.arange(start, stop)
                batch = self.array[scan_indices // B, scan_indices % B]
                batch_labels = final_labels[start:stop]
                for row, label_value in enumerate(cluster_labels):
                    chosen = batch_labels == label_value
                    if np.any(chosen):
                        raw_sums[row] += batch[chosen].sum(axis=0, dtype=np.float64)
            diagnostics['mean_diffraction_patterns'] = (
                raw_sums / cluster_sizes[:, None, None]
            )

        return cluster_map, diagnostics


    #TODO: add option to automatically remove the per-cluster backgrounds by adding 
    #      a parameter `cluster_map` that will have the same shape as the real-space 
    def remove_bg(self, background, bg_frac=1, residual_bg_frac=0,
                  a_min=1, a_max=None, **resBg_kwargs):
        """
        Subtract a diffraction background and clip the resulting dataset.
        
        Parameters
        ----------
        background : ndarray or ReciprocalSpace
            Background diffraction pattern matching the last two data axes.
        bg_frac : float, optional
            The fraction of the background to be subtracted from the dataset.
            Must be between 0 and 1 (inclusive).
        residual_bg_frac : float, optional
            Fraction of the per-pattern residual background to subtract.
        a_min, a_max : float or None, optional
            Lower and upper clipping limits after subtraction. ``a_min=1``
            matches ``clip()``; set ``a_min=None`` to leave values unclipped.
        **resBg_kwargs
            Options passed to ``get_residualBg`` when residual subtraction is used.
        
        Returns
        -------
        HyperData
            A new instance of HyperData with the background subtracted.
        
        Raises
        ------
        ValueError
            If 'bg_frac' is not within the required range [0, 1] or 'background' is 
            not of the same shape as the diffraction patterns.
        """
        if not (0 <= bg_frac <= 1):
            raise ValueError("'bg_frac' must be between 0 and 1, inclusive.")
        
        if not (0 <= residual_bg_frac <= 1):
            raise ValueError("'residual_bg_frac' must be between 0 and 1, inclusive.")
        
        background_array = np.asarray(
            background.array if isinstance(background, ReciprocalSpace)
            else background
        )
        if background_array.shape != self.array.shape[-2:]:
            raise ValueError(
                "'background' must match the last two dimensions of the "
                f"dataset, {self.array.shape[-2:]}."
            )
        if a_min is not None and a_max is not None and a_min > a_max:
            raise ValueError("a_min must be less than or equal to a_max.")

        working_dtype = np.result_type(
            self.array.dtype, background_array.dtype, np.float32
        )
        result = np.asarray(self.array, dtype=working_dtype) - (
            np.asarray(background_array, dtype=working_dtype) * bg_frac
        )
        if residual_bg_frac > 0:
            residual_bg = self.get_residualBg(**resBg_kwargs)
            if not isinstance(residual_bg, np.ndarray) or residual_bg.shape != self.shape[:-2]:
                raise ValueError(
                    "remove_bg requires one residual background per diffraction "
                    "pattern; use bg_method='rings_mean' or 'grimms_ring'."
                )
            result -= residual_bg[..., None, None] * residual_bg_frac

        if a_min is not None or a_max is not None:
            np.clip(result, a_min, a_max, out=result)
        return self._spawn(result, preserve_unfold=True)

    def to_polar(self,
                 center: Tuple[float, float] = None,
                 r_max: float = None,
                 output_shape: Tuple[int, int] = None,
                 order: int = 1,
                 fill_value: float = 0.0,
                 clip: bool = False,
                 progress: bool = True
                 ) -> "HyperData":
        """
        Remap each diffraction pattern from Cartesian ``(ky, kx)`` to polar
        ``(radius, angle)`` coordinates.

        The default output samples the largest centered circle that fits inside
        the diffraction pattern. The radial size is ``ceil(r_max)`` and the
        angular size is ``ceil(2*pi*r_max)``, so angular sampling roughly
        matches the outer circumference. The returned object stores
        ``polar_metadata`` so downstream methods know that the last two axes
        are ``kr`` and ``ktheta``. ``ktheta`` is displayed from 0 to 360
        degrees, while ``kr`` is displayed in the original reciprocal units
        when calibration is available.

        Parameters
        ----------
        center : tuple of two floats, optional
            (y_center, x_center) in pixel coordinates. Defaults to the image midpoint.
        r_max : float, optional
            Maximum sampled radius in pixels. Defaults to the largest centered
            circle that fits inside the diffraction pattern. If a larger radius
            is requested, out-of-bounds samples are filled with ``fill_value``.
        output_shape : tuple (n_r, n_theta), optional
            Desired shape of the output polar image. 
            - n_r = number of radial samples
            - n_theta = number of angular samples
            Default: ``(ceil(r_max), ceil(2*pi*r_max))``.
        order : int, default=1
            The spline interpolation order for map_coordinates (0=nearest,
            1=bilinear, 3=cubic, etc.). For integer input, interpolation returns
            floating-point data; nearest-neighbor sampling retains integer dtype.
        fill_value : float, optional
            Value used for out-of-bounds samples if ``r_max`` extends beyond
            the input diffraction pattern.
        clip : bool, optional
            If True, apply :func:`clip_values` to each transformed pattern.
            Defaults to False to preserve zeros and sub-unit intensities.
        progress : bool, optional
            If True, display a progress bar.

        Returns
        -------
        HyperData
            New object with shape ``(B, n_r, n_theta)`` for 3D data or
            ``(Ry, Rx, n_r, n_theta)`` for 4D data.
        """
        arr = self.array
        shp = arr.shape

        # 1) Determine input dims and default center/r_max
        if len(shp) == 4:
            A, B, C, D = shp
            height, width = C, D
        elif len(shp) == 3:
            A = None
            B, C, D = shp
            height, width = C, D
        else:
            raise ValueError("HyperData must be 3D (B×C×D) or 4D (A×B×C×D).")

        # Default center at image midpoint
        cy = (height - 1) / 2.0
        cx = (width  - 1) / 2.0
        if center is not None:
            cy, cx = center
            cy = float(cy)
            cx = float(cx)

        if not (0 <= cy <= height - 1 and 0 <= cx <= width - 1):
            raise ValueError(
                "center must lie inside the diffraction pattern bounds."
            )

        if r_max is None:
            r_max_used = min(cy, cx, height - 1 - cy, width - 1 - cx)
        else:
            r_max_used = float(r_max)
        if not np.isfinite(r_max_used) or r_max_used <= 0:
            raise ValueError("r_max must be a positive finite value.")

        # 2) Determine output_shape = (n_r, n_theta)
        if output_shape is None:
            n_r = max(1, int(np.ceil(r_max_used)))
            n_theta = max(4, int(np.ceil(2 * np.pi * r_max_used)))
        else:
            n_r, n_theta = output_shape
            n_r = int(n_r)
            n_theta = int(n_theta)
        if n_r <= 0 or n_theta <= 0:
            raise ValueError("output_shape must contain positive integers.")

        # 3) Precompute the polar→Cartesian mapping grid
        #    Radial values from 0 to r_max_used in n_r steps
        r_vals = np.linspace(0, r_max_used, n_r)
        #    Theta from -π to +π in n_theta steps
        theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        #    Meshgrid in (r, θ), shape = (n_r, n_theta) when indexing='ij'
        #    But map_coordinates expects coords stacked as [row_coords; col_coords].
        theta_grid, r_grid = np.meshgrid(theta_vals, r_vals, indexing='xy')
        #    Convert to Cartesian (floating) coordinates
        x_grid = cx + r_grid * np.cos(theta_grid)  # shape: (n_r, n_theta)
        y_grid = cy + r_grid * np.sin(theta_grid)  # shape: (n_r, n_theta)

        #    Stack into a 2×(n_r·n_theta) array for map_coordinates
        coords = np.vstack((
            y_grid.ravel(),  # row indices
            x_grid.ravel()   # col indices
        ))

        # 4) Prepare an output array of the correct shape
        output_dtype = arr.dtype if order == 0 else np.result_type(arr.dtype, np.float32)
        if A is not None:
            out_arr = np.zeros((A, B, n_r, n_theta), dtype=output_dtype)
        else:
            out_arr = np.zeros((B, n_r, n_theta), dtype=output_dtype)

        # 5) The default r_max already crops to the useful centered circle.
        # 6) Loop over all slices. The coordinate grid is shared by every
        # pattern, so the per-pattern work is only interpolation.
        if A is not None:
            iterator = np.ndindex(A, B)
            if progress:
                iterator = tqdm(
                    iterator,
                    total=A * B,
                    desc="Diffraction patterns",
                )
            for i, j in iterator:
                diff = arr[i, j].astype(output_dtype, copy=False)
                # Interpolate the 2D slice onto our polar grid
                polar_flat = map_coordinates(
                    diff,
                    coords,
                    order=order,
                    mode='constant',
                    cval=fill_value
                )
                # Reshape back to (n_r, n_theta) and clip if requested.
                polar_img = polar_flat.reshape((n_r, n_theta))
                if clip:
                    polar_img = clip_values(polar_img)
                out_arr[i, j] = polar_img
        else:
            iterator = (
                tqdm(range(B), desc="Diffraction patterns")
                if progress
                else range(B)
            )
            for j in iterator:
                diff = arr[j].astype(output_dtype, copy=False)
                polar_flat = map_coordinates(
                    diff,
                    coords,
                    order=order,
                    mode='constant',
                    cval=fill_value
                )
                polar_img = polar_flat.reshape((n_r, n_theta))
                if clip:
                    polar_img = clip_values(polar_img)
                out_arr[j] = polar_img

        # 7) Return the new HyperData. The reciprocal calibration is cleared
        # because the last two axes are now radius/angle, not ky/kx pixels.
        polar_hd = self._spawn(
            out_arr,
            reciprocal_units=None,
            reciprocal_conv_factor=None,
            center_beam_metadata=None,
        )
        radius_step_pixels = r_max_used / n_r
        radius_sample_step_pixels = r_max_used / (n_r - 1) if n_r > 1 else 0.0
        radius_unit_scale = (
            self.reciprocal_conv_factor
            if self.reciprocal_conv_factor is not None
            else 1.0
        )
        radius_units = (
            self.reciprocal_units
            if self.reciprocal_units is not None
            else 'pixels'
        )
        polar_hd.polar_metadata = {
            'is_polar': True,
            'center': (cy, cx),
            'r_max': r_max_used,
            'output_shape': (n_r, n_theta),
            'cartesian_shape': (height, width),
            'cartesian_center': (cy, cx),
            'cartesian_reciprocal_units': self.reciprocal_units,
            'cartesian_reciprocal_conv_factor': self.reciprocal_conv_factor,
            'axis_order': ('radius', 'theta'),
            'radius_range': (0.0, r_max_used),
            'radius_range_pixels': (0.0, r_max_used),
            'radius_display_range': (
                0.0,
                r_max_used * radius_unit_scale,
            ),
            'radius_units': radius_units,
            'radius_conv_factor': radius_unit_scale,
            'radius_step_pixels': radius_step_pixels,
            'radius_step': radius_step_pixels * radius_unit_scale,
            'radius_sample_step_pixels': radius_sample_step_pixels,
            'radius_sample_step': radius_sample_step_pixels * radius_unit_scale,
            'theta_range': (0.0, 360.0),
            'theta_units': 'deg',
            'theta_step': 360.0 / n_theta,
            'order': order,
            'fill_value': fill_value,
        }
        return polar_hd

    def to_cartesian(self,
                     output_shape: Union[Tuple[int, int], str] = None,
                     center: Tuple[float, float] = None,
                     order: int = 1,
                     fill_value: float = 0.0,
                     clip: bool = False,
                     progress: bool = True
                     ) -> "HyperData":
        """
        Resample polar diffraction data back onto a Cartesian ``(ky, kx)`` grid.

        This is an inverse resampling operation, not an exact undo of
        :meth:`to_polar`. Any Cartesian pixels outside the sampled polar
        ``r_max`` are filled with ``fill_value``.

        Parameters
        ----------
        output_shape : tuple(int, int), 'original', or None, optional
            Cartesian diffraction shape ``(Ky, Kx)`` to reconstruct. If None,
            use ``(2*n_r, 2*n_r)``, where ``n_r`` is the polar radial axis
            size. This makes the polar ``kr_max`` map to a circle whose
            diameter is the output image width/height. Use ``'original'`` to
            reconstruct on the Cartesian shape stored by :meth:`to_polar`.
        center : tuple(float, float) or None, optional
            Cartesian center ``(cy, cx)`` in the output image. If None, use the
            output midpoint. With ``output_shape='original'``, use the stored
            original Cartesian center when available.
        order : int, optional
            Spline interpolation order passed to :func:`map_coordinates`.
            For integer input, interpolation returns floating-point data;
            nearest-neighbor sampling retains integer dtype.
        fill_value : float, optional
            Value assigned outside the polar support or outside polar bounds.
        clip : bool, optional
            If True, apply :func:`clip_values` to each reconstructed pattern.
            Defaults to False to preserve zeros and sub-unit intensities.
        progress : bool, optional
            If True, display a progress bar.

        Returns
        -------
        HyperData
            New Cartesian HyperData object. The returned object is not marked
            polar. Its reciprocal calibration is computed from the polar
            ``kr_max`` and the selected Cartesian output radius, so the
            displayed ``kx``/``ky`` scale matches the polar radial scale.
        """
        if not self.is_polar:
            raise ValueError(
                "to_cartesian can only be called on HyperData produced by "
                "to_polar, or on an object with polar_metadata."
            )

        arr = self.array
        shp = arr.shape
        if len(shp) == 4:
            A, B, n_r, n_theta = shp
        elif len(shp) == 3:
            A = None
            B, n_r, n_theta = shp
        else:
            raise ValueError(
                "Polar HyperData must be 3D (B, R, Theta) or 4D "
                "(Ry, Rx, R, Theta)."
            )

        metadata = self.polar_metadata or {}
        use_original_canvas = False
        if output_shape is None:
            output_shape = (2 * n_r, 2 * n_r)
        elif isinstance(output_shape, str):
            if output_shape.lower() != 'original':
                raise ValueError("output_shape must be a tuple, None, or 'original'.")
            output_shape = metadata.get('cartesian_shape')
            if output_shape is None:
                raise ValueError(
                    "output_shape is required because polar_metadata does not "
                    "contain 'cartesian_shape'."
                )
            use_original_canvas = True
        height, width = output_shape
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("output_shape must contain positive integers.")

        if center is None:
            if use_original_canvas:
                center = metadata.get(
                    'cartesian_center',
                    metadata.get(
                        'center',
                        ((height - 1) / 2.0, (width - 1) / 2.0),
                    ),
                )
            else:
                center = ((height - 1) / 2.0, (width - 1) / 2.0)
        cy, cx = center
        cy = float(cy)
        cx = float(cx)

        r_max = float(
            metadata.get(
                'r_max',
                metadata.get('radius_range_pixels', (0, n_r - 1))[1],
            )
        )
        if not np.isfinite(r_max) or r_max <= 0:
            raise ValueError("polar_metadata must define a positive finite r_max.")

        if use_original_canvas:
            output_radius_pixels = r_max
        else:
            output_radius_pixels = min(height, width) / 2.0
        if not np.isfinite(output_radius_pixels) or output_radius_pixels <= 0:
            raise ValueError("The output Cartesian radius must be positive.")

        y_grid, x_grid = np.indices((height, width), dtype=float)
        dy = y_grid - cy
        dx = x_grid - cx
        radius_grid = np.sqrt(dx**2 + dy**2)
        theta_grid = np.mod(np.arctan2(dy, dx), 2 * np.pi)

        if n_r > 1:
            radius_coord = radius_grid * (n_r - 1) / output_radius_pixels
        else:
            radius_coord = np.zeros_like(radius_grid)
        theta_coord = theta_grid * n_theta / (2 * np.pi)
        theta_coord = np.mod(theta_coord, n_theta)

        # Add one wrapped theta column so angular interpolation is continuous
        # at 0/360 without wrapping the radial axis.
        theta_coord_padded = theta_coord.copy()
        theta_coord_padded[theta_coord_padded >= n_theta] -= n_theta
        coords = np.vstack((
            radius_coord.ravel(),
            theta_coord_padded.ravel(),
        ))
        outside_support = radius_grid.ravel() > output_radius_pixels

        output_dtype = arr.dtype if order == 0 else np.result_type(arr.dtype, np.float32)
        if A is not None:
            out_arr = np.full((A, B, height, width), fill_value, dtype=output_dtype)
        else:
            out_arr = np.full((B, height, width), fill_value, dtype=output_dtype)

        if A is not None:
            iterator = np.ndindex(A, B)
            if progress:
                iterator = tqdm(
                    iterator,
                    total=A * B,
                    desc="Cartesian diffraction patterns",
                )
            for i, j in iterator:
                polar_img = arr[i, j].astype(output_dtype, copy=False)
                polar_for_interp = np.concatenate(
                    (polar_img, polar_img[:, :1]),
                    axis=1,
                )
                cart_flat = map_coordinates(
                    polar_for_interp,
                    coords,
                    order=order,
                    mode='constant',
                    cval=fill_value,
                )
                cart_img = cart_flat.reshape((height, width))
                if clip:
                    cart_img = clip_values(cart_img)
                cart_img.ravel()[outside_support] = fill_value
                out_arr[i, j] = cart_img
        else:
            iterator = (
                tqdm(range(B), desc="Cartesian diffraction patterns")
                if progress
                else range(B)
            )
            for j in iterator:
                polar_img = arr[j].astype(output_dtype, copy=False)
                polar_for_interp = np.concatenate(
                    (polar_img, polar_img[:, :1]),
                    axis=1,
                )
                cart_flat = map_coordinates(
                    polar_for_interp,
                    coords,
                    order=order,
                    mode='constant',
                    cval=fill_value,
                )
                cart_img = cart_flat.reshape((height, width))
                if clip:
                    cart_img = clip_values(cart_img)
                cart_img.ravel()[outside_support] = fill_value
                out_arr[j] = cart_img

        radius_display_range = metadata.get(
            'radius_display_range',
            (0.0, r_max),
        )
        radius_display_max = float(radius_display_range[1])
        reciprocal_units = metadata.get(
            'radius_units',
            metadata.get('cartesian_reciprocal_units'),
        )
        reciprocal_conv_factor = radius_display_max / output_radius_pixels
        if reciprocal_units is None:
            reciprocal_units = 'pixels'

        return self._spawn(
            out_arr,
            reciprocal_units=reciprocal_units,
            reciprocal_conv_factor=reciprocal_conv_factor,
            polar_metadata=None,
            center_beam_metadata=None,
        )

    def get_average_clusters(self,
                             cluster_map: np.ndarray,
                             domain: str = 'real',
                             plot_averages: bool = True,
                             vmin: float = 4,
                             vmax: float = 14,
                             cmap: str = 'turbo',
                             logScale: bool = True) -> "HyperData":
        """
        Average over clusters defined in either the real (scan) domain or
        the reciprocal (diffraction) domain.

        Parameters
        ----------
        cluster_map : 2D array of ints
            If domain=='real', shape must equal (A, B).
            If domain=='reciprocal', shape must equal (C, D).
        domain : {'real', 'reciprocal'}
            Which axes of `self.array` the cluster_map indexes:
              - 'real'      → cluster_map shape = (A, B)
              - 'reciprocal'→ cluster_map shape = (C, D)
        plot_averages : bool
            If True, show each cluster’s average as an image.
        vmin, vmax, cmap : passed to plt.imshow
        logScale : bool
            If True, plot np.log(average + 1) instead of raw average.

        Returns
        -------
        HyperData
            Holds an array of shape
              - (n_clusters, C, D) for domain='real'
              - (n_clusters, A, B) for domain='reciprocal'
        """
        arr = self.array
        shp = arr.shape

        # must be 4D
        if arr.ndim != 4:
            raise ValueError("`get_average_clusters` only supports 4D HyperData")

        A, B, C, D = shp
        E, F = cluster_map.shape

        # pick domain
        if domain == 'real':
            if (E, F) != (A, B):
                raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ data real‐domain {(A,B)}")
            # cluster labels and counts
            labels = np.unique(cluster_map)
            n_clusters = labels.size
            # output: average diffraction per real‐domain cluster
            avg = np.zeros((n_clusters, C, D), dtype=arr.dtype)

            for idx, lbl in enumerate(labels):
                mask = (cluster_map == lbl)
                # arr[mask, :, :] shapes to (#pixels_in_cluster, C, D)
                avg[idx] = arr[mask, :, :].mean(axis=0)

        elif domain == 'reciprocal':
            if (E, F) != (C, D):
                raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ data reciprocal‐domain {(C,D)}")
            labels = np.unique(cluster_map)
            n_clusters = labels.size
            # output: average real‐space image per reciprocal‐domain cluster
            avg = np.zeros((n_clusters, A, B), dtype=arr.dtype)

            for idx, lbl in enumerate(labels):
                mask = (cluster_map == lbl)
                # arr[:, :, mask] → shape (A, B, #pixels)
                avg[idx] = arr[:, :, mask].mean(axis=2)

        else:
            raise ValueError("`domain` must be 'real' or 'reciprocal'")

        # plotting
        if plot_averages:
            for i in range(n_clusters):
                data = avg[i] + 1
                if logScale:
                    disp = np.log(data)
                else:
                    disp = data

                plt.figure(dpi=150)
                plt.imshow(disp, vmin=vmin, vmax=vmax, cmap=cmap,)
                plt.axis('off')
                plt.title(f"Cluster {i} ({domain})")
                plt.show()

        if domain == 'real':
            return self._spawn(
                avg,
                real_units=None,
                real_conv_factor=None,
            )

        return self._spawn(
            avg,
            reciprocal_units=None,
            reciprocal_conv_factor=None,
        )

    def get_stdev_clusters(self,
                           cluster_map: np.ndarray,
                           threshold: float     = None,
                           r_min: float         = 0.0,
                           r_max: float         = None,
                           logScale: bool       = False,
                           power: float         = None
                          ) -> "HyperData":
        """
        Compute per‐cluster standard deviations in the diffraction domain
        (real‐space clustering only), with optional preprocessing
        (power‐law or log transform), and optional thresholding
        between radii [r_min, r_max].

        Parameters
        ----------
        cluster_map : 2D int array, shape (A, B)
            Real‐space cluster labels.
        threshold : float in [0,1], optional
            If None, returns raw std‐dev arrays.
            If set, returns boolean masks where
              std >= threshold * max(std within [r_min,r_max]).
        r_min : float, default=0.0
            Inner radius (px) to exclude (previously `r_center`).
        r_max : float, optional
            Outer radius (px) to exclude beyond this in thresholding.
            Defaults to `min(cy, cx)` (the largest fully‐inside radius).
        logScale : bool, default=False
            If True, apply np.log(data + 1) before computing std‐dev.
        power : float, optional
            If provided, raise each pattern to this power before any log.

        Returns
        -------
        HyperData
            If `threshold is None`:  `.array` shape = (n_clusters, C, D) of float std‐devs.
            If `threshold` set:      `.array` shape = (n_clusters, C, D) of bool masks.
        """
        arr = self.array
        if arr.ndim != 4:
            raise ValueError("get_stdev_clusters only supports 4D HyperData")
        A, B, C, D = arr.shape

        if cluster_map.shape != (A, B):
            raise ValueError(f"cluster_map shape {cluster_map.shape} ≠ (A,B)=({A},{B})")

        # 1) compute raw std‐dev per cluster
        labels = np.unique(cluster_map)
        n_clusters = labels.size
        stdev_arr = np.zeros((n_clusters, C, D), dtype=float)

        for i, lbl in enumerate(labels):
            sel = (cluster_map == lbl)
            data = arr[sel]  # shape = (#pixels_in_cluster, C, D)
            if power is not None:
                data = data ** power
            if logScale:
                data = np.log(data + 1)
            stdev_arr[i] = data.std(axis=0) if data.size else 0.0

        # 2) if no threshold → return the raw std‐dev map
        if threshold is None:
            return self._spawn(
                stdev_arr,
                real_units=None,
                real_conv_factor=None,
            )

        # 3) validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("`threshold` must be between 0 and 1")

        # 4) build radial masks
        y, x = np.indices((C, D))
        cy, cx = (C - 1) / 2.0, (D - 1) / 2.0
        dist = np.hypot(y - cy, x - cx)

        # default r_max → largest fully‐inside radius
        r_max_used = r_max if r_max is not None else min(cy, cx)

        # we only consider pixels with r_min ≤ r ≤ r_max_used
        region = (dist >= r_min) & (dist <= r_max_used)

        # 5) build boolean mask per cluster
        bool_arr = np.zeros_like(stdev_arr, dtype=bool)
        for i in range(n_clusters):
            sd = stdev_arr[i]
            valid = region
            max_sd = sd[valid].max() if np.any(valid) else 0.0
            thresh_val = threshold * max_sd
            bool_arr[i] = (sd >= thresh_val) & region

        return self._spawn(
            bool_arr,
            real_units=None,
            real_conv_factor=None,
        )
    
#%%

class ReciprocalSpace:
    """
    Container for a single 2D reciprocal-space image or diffraction pattern.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional reciprocal-space data.
    units : str or None, optional
        Physical units associated with the reciprocal-space pixel spacing
        (for example ``'mrad'`` or ``'A^-1'``).
    conv_factor : float or None, optional
        Conversion factor from pixels to physical units, expressed as
        ``units / pixel``. When omitted, plots default to pixel units.
    polar_metadata : dict or None, optional
        Description of a ``(kr, ktheta)`` image produced by a polar transform.
    center_beam_metadata : dict or None, optional
        Direct-beam center and radius information, when available.
    """

    def __init__(self, data, units: str = None, conv_factor: float = None,
                 polar_metadata: dict = None,
                 center_beam_metadata: dict = None):
        """Store the image and copy optional polar and direct-beam metadata."""
        self.array = data
        self.shape = data.shape
        self._denoise_engine = _DenoiseEngine(data)
        self.units = None
        self.conv_factor = None
        self.polar_metadata = deepcopy(polar_metadata) if polar_metadata is not None else None
        self.center_beam_metadata = (
            deepcopy(center_beam_metadata)
            if center_beam_metadata is not None
            else None
        )

        if units is not None or conv_factor is not None:
            self.set_scale(units=units, conv_factor=conv_factor)

    @property
    def is_polar(self):
        """Return True when this 2D image is in polar ``(kr, ktheta)`` space."""
        return self.polar_metadata is not None

    def set_scale(self, units: str, conv_factor: float):
        """
        Attach a reciprocal-space calibration to the diffraction pattern.

        Parameters
        ----------
        units : str
            Physical reciprocal-space units, for example ``'mrad'`` or
            ``'A^-1'``.
        conv_factor : float
            Conversion factor in ``units / pixel``.

        Returns
        -------
        ReciprocalSpace
            The current object, updated in place.
        """
        if units is None or conv_factor is None:
            raise ValueError("'units' and 'conv_factor' must both be provided.")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("'units' must be a non-empty string.")
        if not np.isscalar(conv_factor) or conv_factor <= 0:
            raise ValueError("'conv_factor' must be a positive scalar.")

        self.units = units.strip()
        self.conv_factor = float(conv_factor)
        return self

    def clear_scale(self):
        """Remove any stored reciprocal-space calibration."""
        self.units = None
        self.conv_factor = None
        return self

    def _resolve_scale(self, units=None, conv_factor=None):
        """Resolve plot calibration from explicit inputs or stored metadata."""
        resolved_units = self.units if units is None else units
        resolved_factor = self.conv_factor if conv_factor is None else conv_factor

        if resolved_units is None and resolved_factor is None:
            return None, None
        if resolved_units is None or resolved_factor is None:
            raise ValueError(
                "'units' and 'conv_factor' must be defined together, either "
                "on the object or in the method call."
            )
        if not np.isscalar(resolved_factor) or resolved_factor <= 0:
            raise ValueError("'conv_factor' must be a positive scalar.")

        return str(resolved_units).strip(), float(resolved_factor)

    def _format_unit_text(self, units):
        """Return a display-friendly unit label."""
        if units is None:
            return "px"

        normalized = units.lower().replace(" ", "")
        if normalized in {'inv_ang', 'invang', 'a-1', 'a^-1', 'å^-1', 'å-1', 'ang^-1', 'ang-1'}:
            return r"Å$^{-1}$"
        if normalized in {'mrad', 'mrads'}:
            return "mrad"
        if normalized in {'deg', 'degree', 'degrees'}:
            return r"$^\circ$"
        return units

    def _axis_extent(self, conv_factor=None):
        """
        Return imshow-compatible axis limits centered at the diffraction origin.
        """
        ky, kx = self.shape
        scale = 1.0 if conv_factor is None else conv_factor
        half_y = ky / 2.0
        half_x = kx / 2.0
        # With imshow's upper origin, array row 0 belongs at positive ky.
        return (-half_x * scale, half_x * scale, -half_y * scale, half_y * scale)

    def _polar_axis_info(self, axis_units='auto'):
        """Return imshow extent and labels for polar ``(kr, ktheta)`` data."""
        metadata = self.polar_metadata or {}
        n_r, n_theta = self.shape
        mode = _normalize_unit_mode(axis_units, label='axis_units')
        radius_range_pixels = metadata.get(
            'radius_range_pixels',
            (0.0, float(n_r - 1)),
        )
        source_units = metadata.get('cartesian_reciprocal_units')
        source_factor = metadata.get('cartesian_reciprocal_conv_factor')

        if mode == 'pixels':
            radius_min, radius_max = radius_range_pixels
            radius_units = 'pixels'
            radius_sample_step = metadata.get(
                'radius_sample_step_pixels',
                metadata.get('radius_step_pixels', 1.0),
            )
        elif source_units is None or source_factor is None:
            if mode == 'calibrated':
                raise ValueError(
                    "axis_units='calibrated' requires polar metadata with "
                    "the original reciprocal-space calibration."
                )
            radius_min, radius_max = radius_range_pixels
            radius_units = 'pixels'
            radius_sample_step = metadata.get(
                'radius_sample_step_pixels',
                metadata.get('radius_step_pixels', 1.0),
            )
        else:
            if not np.isscalar(source_factor) or source_factor <= 0:
                raise ValueError(
                    "polar_metadata['cartesian_reciprocal_conv_factor'] must "
                    "be a positive scalar."
                )
            radius_min, radius_max = metadata.get(
                'radius_display_range',
                (
                    radius_range_pixels[0] * source_factor,
                    radius_range_pixels[1] * source_factor,
                ),
            )
            radius_units = source_units
            radius_sample_step = metadata.get(
                'radius_sample_step',
                metadata.get('radius_step', source_factor),
            )

        theta_min, theta_max = metadata.get('theta_range', (0.0, 360.0))
        theta_units = metadata.get('theta_units', 'deg')

        radius_unit_text = self._format_unit_text(radius_units)
        theta_unit_text = self._format_unit_text(theta_units)
        extent = (theta_min, theta_max, radius_max, radius_min)
        return extent, theta_unit_text, radius_unit_text, radius_min, radius_sample_step

    def _spawn(self, data, units=_SCALE_UNSET, conv_factor=_SCALE_UNSET,
               polar_metadata=_SCALE_UNSET,
               center_beam_metadata=_SCALE_UNSET):
        """Create a new ReciprocalSpace object while preserving calibration."""
        if units is _SCALE_UNSET:
            units = self.units
        if conv_factor is _SCALE_UNSET:
            conv_factor = self.conv_factor
        if polar_metadata is _SCALE_UNSET:
            polar_metadata = self.polar_metadata
        if center_beam_metadata is _SCALE_UNSET:
            center_beam_metadata = self.center_beam_metadata
        return ReciprocalSpace(
            data,
            units=units,
            conv_factor=conv_factor,
            polar_metadata=deepcopy(polar_metadata) if polar_metadata is not None else None,
            center_beam_metadata=(
                deepcopy(center_beam_metadata)
                if center_beam_metadata is not None
                else None
            ),
        )

    def copy(self):
        """
        Return an independent copy of this ReciprocalSpace object.

        The image array, calibration, and polar metadata are copied so the
        returned object can be edited without changing this object.
        """
        return self._spawn(np.array(self.array, copy=True))

    @staticmethod
    def _plot_tick_count(value, name, minimum=1):
        """Validate a requested number of displayed tick marks."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, np.integer)
        ) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        return int(value)

    @staticmethod
    def _draw_scale_bar(ax, length, unit_text, color, position, label,
                        mode, *, extent=None, radius_max=None):
        """Draw a reciprocal-length bar without changing image coordinates."""
        margin = 0.06
        if mode == 'polar':
            span = 2 * radius_max
            bar_width = length / span
            # Keep the bar inside the lower chord of the circular image.
            y_position = 0.15
            chord_half_width = np.sqrt(0.25 - (y_position - 0.5) ** 2)
            x_min = 0.5 - chord_half_width + 0.02
            x_max = 0.5 + chord_half_width - 0.02
            if bar_width > x_max - x_min:
                raise ValueError("scale_bar is too long for the polar plot.")
            cap_height = 0.012
            transform = ax.transAxes
        else:
            span = extent[1] - extent[0]
            bar_width = length
            if bar_width > (1 - 2 * margin) * span:
                raise ValueError("scale_bar is too long for the displayed image.")
            y_position = extent[2] + 0.07 * (extent[3] - extent[2])
            cap_height = 0.012 * (extent[3] - extent[2])
            transform = ax.transData

        if position == 'left':
            x_start = x_min if mode == 'polar' else extent[0] + margin * span
        elif position == 'right':
            x_start = (x_max - bar_width) if mode == 'polar' else (
                extent[1] - margin * span - bar_width
            )
        else:
            x_start = (1 - bar_width) / 2 if mode == 'polar' else (
                (extent[0] + extent[1] - bar_width) / 2
            )
        x_end = x_start + bar_width
        outline = [
            path_effects.Stroke(linewidth=5, foreground='black'),
            path_effects.Normal(),
        ]
        ax.plot(
            [x_start, x_end], [y_position, y_position],
            color=color, linewidth=3, transform=transform,
            solid_capstyle='butt', clip_on=False, zorder=5,
            path_effects=outline,
        )
        for x_position in (x_start, x_end):
            ax.plot(
                [x_position, x_position],
                [y_position - cap_height, y_position + cap_height],
                color=color, linewidth=2, transform=transform,
                clip_on=False, zorder=5, path_effects=outline,
            )
        if label:
            ax.text(
                (x_start + x_end) / 2,
                y_position + 1.8 * cap_height,
                f"{length:g} {unit_text}",
                color=color, ha='center', va='bottom',
                transform=transform, clip_on=False, zorder=5,
                path_effects=outline,
            )
    
    def show(self,
             power: float = 1,
             title: str = 'Diffraction Pattern',
             logScale: bool = True,
             axes: bool = True,
             vmin=None,
             vmax=None,
             figsize=(10, 10),
             aspect=None,
             cmap: str = 'turbo',
             coords: np.ndarray | None = None,
             axis_units='auto',
             y=None,
             x=None,
             overlay_mask=None,
             overlay_color='green',
             overlay_alpha=0.3,
             mode='cartesian',
             grid=False,
             grid_ticks=None,
             radial_ticks=None,
             angular_ticks=None,
             grid_color='white',
             grid_alpha=0.35,
             scale_bar=None,
             scale_bar_position=None,
             scale_bar_color='white',
             scale_bar_label=True,
             **scatter_kwargs):
        """
        Visualize the diffraction pattern stored in this ReciprocalSpace object.
    
        The diffraction pattern is displayed with optional log-scaling and
        intensity exponentiation. Optionally, a set of (y, x) peak coordinates
        can be overlaid as a scatter plot on top of the image. ``mode='polar'``
        draws a circular plot. Cartesian input is sampled within its largest
        centered circle for this display only; the stored array is unchanged.
        Angles increase clockwise from the right, matching array row order.
    
        Parameters
        ----------
        power : float, optional
            Intensity exponent. If `logScale` is True, the displayed image is
            `power * log(self.array)`. If `logScale` is False, the displayed
            image is `self.array ** power`. Default is 1.
        title : str, optional
            Title for the diffraction pattern (used on the axes when `axes=True`).
        logScale : bool, optional
            If True, use a logarithmic transform (`log`) of the data before
            exponentiation. If False, use a pure power-law transform.
        axes : bool, optional
            If True, show axes, ticks, and a colorbar. If False, hide these;
            a requested grid or scale bar still appears.
        vmin : float or None, optional
            Minimum intensity for color scaling. If None, it is inferred from
            the transformed data.
        vmax : float or None, optional
            Maximum intensity for color scaling. If None, it is inferred from
            the transformed data.
        figsize : tuple, optional
            Figure size passed to `plt.figure(figsize=...)`.
        aspect : float, str, or None, optional
            Aspect ratio for the displayed image. If not None, passed to
            `ax.set_aspect(aspect)` in rectangular mode. Circular polar mode
            keeps an equal aspect ratio.
        cmap : str, optional
            Matplotlib colormap name for the image. Default is 'turbo'.
        coords : array-like of shape (N, 2), optional
            Optional array of peak coordinates to overlay as scatter points.
            Each row should be `[y, x]`. When provided, points are plotted at
            `x = coords[:, 1]` and `y = coords[:, 0]` on top of the image.
        y, x : scalar or array-like, optional
            Alternative pixel coordinates for scatter points. Supply both,
            with the same number of values. Do not combine with ``coords``.
        overlay_mask : ndarray of bool or None, optional
            A detector region to tint over the diffraction pattern. Must have
            the same shape as the image.
        overlay_color : matplotlib color, optional
            Color used for ``overlay_mask``.
        overlay_alpha : float, optional
            Opacity of ``overlay_mask`` between 0 and 1.
        axis_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system used for the displayed axes. ``'auto'`` uses stored
            reciprocal-space calibration when available and otherwise falls
            back to pixels. For polar data, calibrated display uses ``kr`` in
            the source reciprocal units and ``ktheta`` in degrees.
        mode : {'cartesian', 'polar'}, optional
            ``'cartesian'`` preserves the current rectangular display,
            including an angle-versus-radius view for stored polar data.
            ``'polar'`` displays a circular radius/angle projection.
        grid : bool, optional
            Draw coordinate grid lines. The grid also appears with ``axes=False``.
        grid_ticks : int or (int, int), optional
            Number of Cartesian/rectangular grid ticks, ``(y, x)``, including
            endpoints. A scalar applies to both axes. Defaults to 6 when
            ``grid=True``. Not used in circular polar mode.
        radial_ticks, angular_ticks : int or None, optional
            Number of radial rings and angular spokes in circular polar mode.
            Defaults to 5 and 12 when ``grid=True``. Not used in Cartesian mode.
        grid_color : matplotlib color, optional
            Grid-line color; defaults to white.
        grid_alpha : float, optional
            Grid-line opacity between 0 and 1.
        scale_bar : positive float or None, optional
            Scale-bar length in the displayed horizontal units for rectangular
            mode, or radial units for circular polar mode. Thus a rectangular
            view of stored polar data uses degrees. If None, no bar is drawn.
        scale_bar_position : {'left', 'center', 'right'} or None, optional
            Horizontal placement. Defaults to left for rectangular mode and
            center for circular polar mode.
        scale_bar_color : matplotlib color, optional
            Scale-bar and label color; defaults to white.
        scale_bar_label : bool, optional
            If False, omit the numeric length and unit text.
        **scatter_kwargs :
            Additional keyword arguments forwarded to `plt.scatter(...)` for
            the overlay points (e.g., `c='r'`, `s=20`, `marker='x'`, etc.).
    
        Notes
        -----
        - The method assumes `self.array` is a 2D diffraction pattern.
        - For `logScale=True`, non-positive values in `self.array` will produce
          `-inf` or `nan` in the log; it is recommended to use background-
          subtracted and strictly positive data when using log scaling.
        """
        if self.array.ndim != 2:
            raise ValueError("ReciprocalSpace.show requires a 2D array.")
        if 'units' in scatter_kwargs or 'conv_factor' in scatter_kwargs:
            raise TypeError(
                "Use axis_units='auto', 'pixels', or 'calibrated' instead of "
                "passing units/conv_factor to show()."
            )
        if mode not in ('cartesian', 'polar'):
            raise ValueError("mode must be 'cartesian' or 'polar'.")
        if mode == 'polar' and grid_ticks is not None:
            raise ValueError("grid_ticks applies only to rectangular mode.")
        if mode == 'cartesian' and (radial_ticks is not None or angular_ticks is not None):
            raise ValueError("radial_ticks and angular_ticks require mode='polar'.")
        if grid_ticks is not None:
            if isinstance(grid_ticks, (Integral, np.integer)) and not isinstance(
                grid_ticks, (bool, np.bool_)
            ):
                grid_ticks = (grid_ticks, grid_ticks)
            if not isinstance(grid_ticks, (tuple, list, np.ndarray)) or len(grid_ticks) != 2:
                raise ValueError("grid_ticks must be an integer or a (y, x) pair.")
            grid_ticks = tuple(
                self._plot_tick_count(count, 'grid_ticks', minimum=2)
                for count in grid_ticks
            )
        if radial_ticks is not None:
            radial_ticks = self._plot_tick_count(radial_ticks, 'radial_ticks')
        if angular_ticks is not None:
            angular_ticks = self._plot_tick_count(angular_ticks, 'angular_ticks')
        try:
            grid_alpha = float(grid_alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("grid_alpha must be between 0 and 1.") from exc
        if not np.isfinite(grid_alpha) or not 0 <= grid_alpha <= 1:
            raise ValueError("grid_alpha must be between 0 and 1.")
        to_rgba(grid_color)
        to_rgba(scale_bar_color)
        if scale_bar_position is None:
            scale_bar_position = 'center' if mode == 'polar' else 'left'
        if scale_bar_position not in ('left', 'center', 'right'):
            raise ValueError("scale_bar_position must be 'left', 'center', or 'right'.")
        if scale_bar is not None:
            if isinstance(scale_bar, (bool, np.bool_)):
                raise ValueError("scale_bar must be a positive length.")
            try:
                scale_bar = float(scale_bar)
            except (TypeError, ValueError) as exc:
                raise ValueError("scale_bar must be a positive length.") from exc
            if not np.isfinite(scale_bar) or scale_bar <= 0:
                raise ValueError("scale_bar must be a positive length.")
        if coords is not None and (y is not None or x is not None):
            raise ValueError("Provide either coords or y and x, not both.")
        if (y is None) != (x is None):
            raise ValueError("Supply both y and x for scatter points.")
        if y is not None:
            y_values = np.atleast_1d(np.asarray(y, dtype=float))
            x_values = np.atleast_1d(np.asarray(x, dtype=float))
            if y_values.ndim != 1 or x_values.ndim != 1:
                raise ValueError("y and x must be scalars or 1D arrays.")
            if y_values.shape != x_values.shape:
                raise ValueError("y and x must contain the same number of points.")
            coords = np.column_stack((y_values, x_values))
        if coords is not None:
            coords = np.asarray(coords, dtype=float)
            if coords.shape == (2,):
                coords = coords.reshape(1, 2)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("coords must have shape (N, 2).")
            if not np.all(np.isfinite(coords)):
                raise ValueError("Scatter coordinates must be finite.")
        if overlay_mask is not None:
            overlay_mask = np.asarray(overlay_mask, dtype=bool)
            if overlay_mask.shape != self.shape:
                raise ValueError(
                    f"overlay_mask must have shape {self.shape}; got "
                    f"{overlay_mask.shape}."
                )
            if not np.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1:
                raise ValueError("overlay_alpha must be between 0 and 1.")

        if self.is_polar:
            (
                extent,
                theta_unit_text,
                radius_unit_text,
                radius_axis_start,
                radius_sample_step,
            ) = self._polar_axis_info(axis_units=axis_units)
            conv_factor = None
        else:
            units, conv_factor, _ = _resolve_unit_mode(
                axis_units, self.units, self.conv_factor, label='axis_units',
            )
            axis_unit_text = self._format_unit_text(units)
            extent = self._axis_extent(conv_factor=conv_factor)

        image_data = self.array
        plot_overlay = overlay_mask
        if mode == 'polar':
            if self.is_polar:
                theta_edges = np.deg2rad(
                    np.linspace(extent[0], extent[1], self.shape[1] + 1)
                )
                radius_edges = np.linspace(extent[3], extent[2], self.shape[0] + 1)
                radius_max = float(extent[2])
            else:
                ky, kx = self.shape
                center_y, center_x = (ky - 1) / 2.0, (kx - 1) / 2.0
                radius_max_pixels = min(center_y, center_x)
                if radius_max_pixels <= 0:
                    raise ValueError("A circular view requires at least a 2x2 image.")
                n_radius = max(2, int(np.ceil(radius_max_pixels)) + 1)
                n_angle = max(24, int(np.ceil(2 * np.pi * radius_max_pixels)))
                radius_samples = np.linspace(0, radius_max_pixels, n_radius)
                angle_samples = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
                sample_y = center_y + radius_samples[:, None] * np.sin(angle_samples)
                sample_x = center_x + radius_samples[:, None] * np.cos(angle_samples)
                sampling_coords = np.vstack((sample_y.ravel(), sample_x.ravel()))
                image_data = map_coordinates(
                    self.array, sampling_coords, order=1, mode='nearest',
                ).reshape(n_radius, n_angle)
                if overlay_mask is not None:
                    plot_overlay = map_coordinates(
                        overlay_mask.astype(np.uint8), sampling_coords,
                        order=0, mode='nearest',
                    ).reshape(n_radius, n_angle).astype(bool)
                plot_scale = 1.0 if conv_factor is None else conv_factor
                radius_max = radius_max_pixels * plot_scale
                radius_edges = np.linspace(0, radius_max, n_radius + 1)
                theta_edges = np.linspace(0, 2 * np.pi, n_angle + 1)
                radius_unit_text = axis_unit_text
            if not np.isfinite(radius_max) or radius_max <= 0:
                raise ValueError("Polar display requires a positive radial range.")

        with np.errstate(divide='ignore', invalid='ignore'):
            processed_data = power * np.log(image_data) if logScale else image_data ** power
        finite_values = np.asarray(processed_data)[np.isfinite(processed_data)]
        if finite_values.size == 0:
            raise ValueError("Diffraction pattern has no finite values to display.")
        if vmin is None:
            vmin = np.min(finite_values)
        if vmax is None:
            vmax = np.max(finite_values)
        processed_data = np.ma.masked_invalid(processed_data)

        if mode == 'polar':
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
            ax.grid(False)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            im1 = ax.pcolormesh(
                theta_edges, radius_edges, processed_data,
                cmap=cmap, vmin=vmin, vmax=vmax, shading='flat', zorder=1,
            )
            ax.set_rlim(0, radius_max)
            if plot_overlay is not None:
                mask_cmap = ListedColormap([
                    (0, 0, 0, 0), to_rgba(overlay_color, overlay_alpha),
                ])
                ax.pcolormesh(
                    theta_edges, radius_edges, plot_overlay.astype(np.uint8),
                    cmap=mask_cmap, vmin=0, vmax=1, shading='flat', zorder=2,
                )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            im1 = ax.imshow(
                processed_data, vmin=vmin, vmax=vmax, cmap=cmap,
                extent=extent, origin='upper', zorder=1,
            )
            if plot_overlay is not None:
                rgba = np.empty((*self.shape, 4), dtype=np.float32)
                rgba[:] = to_rgba(overlay_color)
                rgba[..., 3] = plot_overlay * overlay_alpha
                ax.imshow(
                    rgba, extent=extent, origin='upper',
                    interpolation='nearest', zorder=2,
                )
            if aspect is not None:
                ax.set_aspect(aspect)

        if coords is not None and coords.size > 0:
            if self.is_polar:
                metadata = self.polar_metadata or {}
                theta_step = metadata.get('theta_step', 360.0 / self.shape[1])
                theta_min = metadata.get('theta_range', (0.0, 360.0))[0]
                angle_coords = theta_min + coords[:, 1] * theta_step
                radius_coords = radius_axis_start + coords[:, 0] * radius_sample_step
                x_coords = np.deg2rad(angle_coords) if mode == 'polar' else angle_coords
                y_coords = radius_coords
            else:
                plot_scale = 1.0 if conv_factor is None else conv_factor
                center_y = (self.shape[0] - 1) / 2.0
                center_x = (self.shape[1] - 1) / 2.0
                delta_y = coords[:, 0] - center_y
                delta_x = coords[:, 1] - center_x
                if mode == 'polar':
                    x_coords = np.mod(np.arctan2(delta_y, delta_x), 2 * np.pi)
                    y_coords = np.hypot(delta_y, delta_x) * plot_scale
                else:
                    x_coords = delta_x * plot_scale
                    y_coords = -delta_y * plot_scale
            scatter_kwargs.setdefault('zorder', 3)
            ax.scatter(x_coords, y_coords, **scatter_kwargs)

        if mode == 'polar':
            n_radial = radial_ticks if radial_ticks is not None else (5 if grid else None)
            n_angular = angular_ticks if angular_ticks is not None else (12 if grid else None)
            if n_radial is not None:
                radius_locations = np.linspace(0, radius_max, n_radial + 1)[1:]
                ax.set_yticks(radius_locations)
                ax.set_yticklabels([f"{value:g}" for value in radius_locations])
            if n_angular is not None:
                angle_locations = np.linspace(0, 360, n_angular, endpoint=False)
                ax.set_xticks(np.deg2rad(angle_locations))
                ax.set_xticklabels([f"{value:g}°" for value in angle_locations])
            if grid and axes:
                ax.grid(color=grid_color, alpha=grid_alpha, linewidth=0.8)
            elif grid:
                angles_for_lines = np.linspace(0, 2 * np.pi, 361)
                for radius in radius_locations:
                    ax.plot(
                        angles_for_lines,
                        np.full_like(angles_for_lines, radius),
                        color=grid_color, alpha=grid_alpha, linewidth=0.8, zorder=2.5,
                    )
                for angle in np.deg2rad(angle_locations):
                    ax.plot(
                        [angle, angle], [0, radius_max],
                        color=grid_color, alpha=grid_alpha, linewidth=0.8, zorder=2.5,
                    )
            if axes:
                ax.set_rlabel_position(145)
                for tick_label in ax.get_yticklabels():
                    tick_label.set_color('white')
                    tick_label.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='black'),
                        path_effects.Normal(),
                    ])
                ax.set_title(
                    f"{title}  ($k_r$: {radius_unit_text})",
                    fontsize=16, pad=28,
                )
            else:
                ax.set_axis_off()
            bar_unit_text = radius_unit_text
        else:
            tick_counts = grid_ticks if grid_ticks is not None else ((6, 6) if grid else None)
            if tick_counts is not None:
                n_y, n_x = tick_counts
                x_locations = np.linspace(extent[0], extent[1], n_x)
                y_locations = np.linspace(extent[2], extent[3], n_y)
                ax.set_xticks(x_locations)
                ax.set_yticks(y_locations)
            if grid and axes:
                ax.grid(color=grid_color, alpha=grid_alpha, linewidth=0.8)
            elif grid:
                for location in x_locations:
                    ax.axvline(location, color=grid_color, alpha=grid_alpha,
                               linewidth=0.8, zorder=2.5)
                for location in y_locations:
                    ax.axhline(location, color=grid_color, alpha=grid_alpha,
                               linewidth=0.8, zorder=2.5)
            if axes:
                if self.is_polar:
                    ax.set_xlabel(rf"$k_\theta$ ({theta_unit_text})", fontsize=14)
                    ax.set_ylabel(rf"$k_r$ ({radius_unit_text})", fontsize=14)
                else:
                    ax.set_xlabel(rf"$k_x$ ({axis_unit_text})", fontsize=14)
                    ax.set_ylabel(rf"$k_y$ ({axis_unit_text})", fontsize=14)
                ax.set_title(title, fontsize=18)
            else:
                ax.set_axis_off()
            bar_unit_text = theta_unit_text if self.is_polar else axis_unit_text

        if axes:
            if mode == 'polar':
                cb = fig.colorbar(im1, ax=ax, pad=0.1, shrink=0.8)
            else:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = fig.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            if logScale:
                cbar_title = (
                    "log(Intensity)" if power == 1
                    else f"log(Intensity)\n(Power = {power})"
                )
            else:
                cbar_title = (
                    "Intensity" if power == 1
                    else f"Intensity\n(Power = {power})"
                )
            cb.set_label(cbar_title, fontsize=14)

        if scale_bar is not None:
            self._draw_scale_bar(
                ax, scale_bar, bar_unit_text, scale_bar_color,
                scale_bar_position, scale_bar_label, mode,
                extent=extent, radius_max=radius_max if mode == 'polar' else None,
            )
        if mode == 'polar':
            fig.tight_layout(pad=1.5)
        plt.show()
    
    def block_direct_beam(self, radius=None, center=None, beam_units='auto',
                          fill_value=1, return_mask=False):
        """
        Block the central/direct beam in this diffraction pattern.

        The selected circular beam region is replaced with ``fill_value``.
        Explicit ``radius`` and ``center`` values take precedence over stored
        ``center_beam_metadata`` from alignment. If ``radius`` is omitted, the
        method uses metadata and raises a helpful error if no stored direct-
        beam radius is available.

        Parameters
        ----------
        radius : float or None, optional
            Direct-beam radius. Interpreted according to ``beam_units`` when
            provided explicitly.
        center : array-like of two floats or None, optional
            Direct-beam center as ``(ky, kx)``. Pixel centers are array
            coordinates. Calibrated centers are reciprocal coordinates
            relative to the diffraction origin, with positive ``ky`` upward
            and positive ``kx`` to the right.
        beam_units : {'auto', 'pixels', 'calibrated'}, optional
            Unit system for explicit ``radius`` and ``center``. ``'auto'``
            uses this object's calibration when available and otherwise falls
            back to pixels.
        fill_value : scalar, optional
            Value assigned to blocked direct-beam pixels. Defaults to 1.
        return_mask : bool, optional
            If True, return ``(blocked_pattern, beam_mask)`` where
            ``beam_mask`` is True for blocked pixels.

        Returns
        -------
        ReciprocalSpace or tuple
            New diffraction-pattern object with the direct beam blocked,
            optionally followed by the 2D beam mask.

        Notes
        -----
        ``block_direct_beam`` is the preferred name. ``remove_center_beam`` is
        kept as an alias for older notebook wording.
        """
        if self.array.ndim != 2:
            raise ValueError(
                "ReciprocalSpace.block_direct_beam requires a single 2D "
                "diffraction pattern."
            )
        if self.is_polar:
            raise ValueError(
                "block_direct_beam currently expects Cartesian reciprocal "
                "axes. Apply it before converting to polar coordinates, or "
                "use an explicit polar mask."
            )
        if not np.isscalar(fill_value):
            raise ValueError("fill_value must be a scalar.")

        ky, kx = self.shape
        center_px, radius_px, resolved_metadata = _resolve_center_beam_parameters(
            (ky, kx),
            metadata=self.center_beam_metadata,
            radius=radius,
            center=center,
            units=self.units,
            conv_factor=self.conv_factor,
            beam_units=beam_units,
        )
        beam_mask = make_mask(center_px, radius_px, mask_dim=(ky, kx))
        if not np.any(beam_mask):
            raise ValueError(
                "The resolved direct-beam mask selects no pixels. Check "
                "radius, center, and beam_units."
            )

        output_dtype = np.result_type(self.array.dtype, np.asarray(fill_value).dtype)
        blocked = self.array.astype(output_dtype, copy=True)
        blocked[beam_mask] = fill_value

        result = self._spawn(blocked)
        resolved_metadata.update({
            'source': 'block_direct_beam',
            'fill_value': fill_value,
            'previous_metadata': deepcopy(self.center_beam_metadata),
        })
        result.center_beam_metadata = resolved_metadata

        if return_mask:
            return result, beam_mask
        return result

    remove_center_beam = block_direct_beam
    
    def crop(self, kylim=None, kxlim=None, kshape=None):
        """
        Crop a diffraction pattern (ReciprocalSpace object) with optional
        subpixel precision and enforced resizing.
    
        Parameters
        ----------
        kylim : int, float, or tuple, optional
            Vertical reciprocal-space limits. Can be float for subpixel cropping.
        kxlim : int, float, or tuple, optional
            Horizontal reciprocal-space limits. Can be float for subpixel cropping.
        kshape : tuple of (int, int), optional
            Output shape (A, B) for resizing the cropped pattern.
            If not provided, inferred from crop size.
            Interpolating integer data returns floating-point output. A crop
            that needs no interpolation retains the original dtype.
    
        Returns
        -------
        ReciprocalSpace
            New object containing the cropped (and optionally resized) diffraction pattern.
        """
    
        def parse_limits(limits, max_length):
            if limits is None:
                return (0, max_length)
            elif isinstance(limits, (int, float)):
                if limits < 0 or limits > max_length:
                    raise ValueError("Index out of bounds")
                return (limits, limits + 1)
            elif isinstance(limits, tuple):
                start, end = limits
                if start < 0 or end > max_length:
                    raise ValueError("Invalid range or out of bounds")
                return (start, end)
            else:
                raise ValueError("Limits must be int, float, tuple, or None")
    
        # --- Parse reciprocal-space limits ---
        ky, kx = self.shape
        kylim_range = parse_limits(kylim, ky)
        kxlim_range = parse_limits(kxlim, kx)
    
        # Integer boundaries for extraction
        y0, y1 = int(np.floor(kylim_range[0])), int(np.ceil(kylim_range[1]))
        x0, x1 = int(np.floor(kxlim_range[0])), int(np.ceil(kxlim_range[1]))
    
        # Detect if subpixel cropping is requested
        subpixel = not all(isinstance(v, int) for v in kylim_range + kxlim_range)
    
        # Default output shape
        if kshape is None:
            A = int(round(kylim_range[1] - kylim_range[0]))
            B = int(round(kxlim_range[1] - kxlim_range[0]))
            kshape = (A, B)
    
        if kshape[0] != kshape[1]:
            print(f"Warning: Non-square output shape {kshape}. Proceeding anyway.")
    
        # --- Extract raw region ---
        cropped = self.array[y0:y1, x0:x1]
    
        new_conv_factor = self.conv_factor

        # --- Handle subpixel or resize ---
        if subpixel or kshape != cropped.shape:
            if self.conv_factor is not None:
                y_scale = (kylim_range[1] - kylim_range[0]) / kshape[0]
                x_scale = (kxlim_range[1] - kxlim_range[0]) / kshape[1]

                if np.isclose(y_scale, x_scale):
                    new_conv_factor = self.conv_factor * x_scale
                else:
                    print("Warning: anisotropic resizing cleared the stored reciprocal-space calibration.")
                    new_conv_factor = None

            resize_dtype = np.result_type(cropped.dtype, np.float32)
            cropped = transform.resize(
                cropped.astype(resize_dtype, copy=False), kshape,
                order=1, mode='reflect', anti_aliasing=True,
                preserve_range=True,
            )

        new_units = self.units if new_conv_factor is not None else None
        geometry_changed = (y0, y1, x0, x1) != (0, ky, 0, kx) or (
            subpixel or kshape != (ky, kx)
        )
        beam_metadata = self.center_beam_metadata
        if geometry_changed:
            beam_metadata = None if subpixel or self.polar_metadata is not None else (
                _resampled_center_beam_metadata(
                    self.center_beam_metadata, (ky, kx), cropped.shape,
                    crop_origin=(y0, x0), crop_shape=(y1 - y0, x1 - x0),
                    units=new_units, conv_factor=new_conv_factor,
                )
            )
        return self._spawn(
            cropped, units=new_units, conv_factor=new_conv_factor,
            center_beam_metadata=beam_metadata,
        )
    
    def get_spotCenter(self, ky, kx, r, method='CoM', plotSpot=False,):
        """
        Refine a diffraction-spot center inside a local circular window.

        Missing pixels near detector edges are treated as zero, as with a
        zero-padded diffraction pattern, without padding the whole image.
        """
        return _spot_center_from_array(
            self.array, ky, kx, r, method=method, plotSpot=plotSpot
        )
    
    def get_centers(self, r, ref_coords, show=False, method='CoM'):
        """
        Refine reference-peak centers in this diffraction pattern.

        ``r`` may be a scalar or one radius per reference peak. ``method``
        selects center of mass, axis-aligned Gaussian, or rotated elliptical
        Gaussian fitting.
        """

        assert len(self.shape) == 2, "Input data must be of 2-dimensional"

        num_peaks = len(ref_coords)
        radii = _peak_radii(r, num_peaks)
        centers = np.zeros((num_peaks, 2))
        for j in range(num_peaks):
            centers[j] = self.get_spotCenter(
                ref_coords[j, 0], ref_coords[j, 1],
                radii[j] + 1e-10, method, show,
            )
        return centers

    #TODO: enable functionality for 4-fold symmetry as well
    #TODO: automatically find the number of peaks on each order by looking at the 
    #      distance from center beam (frequency)
    def masked_DPs(self, mask_radius, centers=None, ref_coords=None, order=None, title=None, 
                   return_mask=False, plot=True, method='CoM'):
        """ 
        Generate masked diffraction plots for each order.
        Order = 1,2,3,4; Last option plots all.
        """
        
        if centers is None and ref_coords is None:
            raise ValueError("Either 'centers' is None or 'ref_coords' is None but not both.")
        
        A,B = self.shape
        compound_mask = np.zeros((A, B))
        
        if centers is None:
            centers = self.get_centers(r=mask_radius, ref_coords=ref_coords, method=method)
    
        for mask_center in centers:
                    
            mask_spot = make_mask(mask_center, mask_radius, mask_dim=(A,B))
            compound_mask = compound_mask + mask_spot     
        
        # Apply compound mask
        masked_data = self.array*compound_mask
        
        if plot: 
            
            plt.figure(figsize=(10, 10))
            base_cmap = plt.cm.turbo
        
            # Create a new colormap from the existing colormap
            # np.concatenate combines the arrays. The first array is just [1, 1, 1, 1] which corresponds to white in RGBA.
            # We take the colormap 'turbo', convert it to an array, and exclude the first color to make room for white.
            custom_cmap = ListedColormap(np.concatenate(([np.array([1, 1, 1, 1])], 
                                                         base_cmap(np.linspace(0, 1, 2**12))[1:]), axis=0))
        
            # Plotting 
            im1 = plt.imshow(masked_data, cmap=custom_cmap)
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            # Title settings based on 'spots_order'
            if title:
                assert type(title) == str, "The title of the plot must be a string."
                ax.set_title(title, fontsize=18)
            else:
                ax.set_title('Masked Diffraction Pattern', fontsize=18)
            
            # Colorbar settings
            cb = plt.colorbar(im1, cax=cax)
            cb.ax.tick_params(labelsize=15)
            plt.title("Intensity", fontsize=14, pad=20)  # Adjusted to avoid overlapping with the main title
        
            plt.show()
        
        if return_mask:
            return compound_mask
        
    def get_intensities(self,
                        r,
                        centers=None,  
                        ref_coords=None, 
                        method='CoM',
                        compute_resBg=False,
                        residual_frac=0.9,
                        **resBg_kwargs):
        """
        Integrate Bragg intensities in circular windows around peak centers.

        ``r`` may be one radius or one radius per peak. Windows are clipped
        to the detector boundaries; no pixels wrap around an image edge. When
        subtracting residual background, the correction uses the number of
        pixels actually included in each integration window.
        """
    
        if centers is None:
            centers = self.get_centers(r, ref_coords=ref_coords, method=method)
    
        if compute_resBg:
            ints, pixel_counts = _peak_intensities_from_array(
                self.array, r, centers, return_pixel_counts=True
            )
            res_bg = self.get_residualBg(centers=centers, **resBg_kwargs)
            correction = np.zeros_like(ints)
            np.multiply(
                res_bg, pixel_counts, out=correction, where=pixel_counts > 0
            )
            ints -= correction * residual_frac
        else:
            ints = _peak_intensities_from_array(self.array, r, centers)
                
        return ints
    
    
    def get_residualBg(self, centers, r_spots, bg_method='rings', t_ring=None, show=False, **kwargs):
        """
        Calculate the residual background value around Bragg peaks.
    
        The function supports three methods to calculate the background:
        
        - The 'rings' method masks the immediate annular region around each Bragg peak 
          at the positions in 'centers' and returns the mean background value 
          surrounding each Bragg peak.
        - The 'rings_mean' method creates a ring around each Bragg peak and returns 
          the average background value per pixel for all the rings combined.
        - The 'grimms_ring' method creates a large annular mask that ideally passes 
          through the Bragg peaks located at 'centers'. It hollows out the circular 
          regions enclosing the Bragg peaks, so only the space between the peaks is 
          masked. This method assumes that the Bragg peaks at positions 'centers' fall 
          within a common annular region.
    
        Parameters
        ----------
        centers : array-like
            An array of shape (N, 2) containing the (ky, kx) positions of the Bragg peaks.
        r_spots : float or tuple of floats
            Radius of the spots to mask. If a tuple (inner_radius, outer_radius) 
            is provided, an annular mask is created.
        bg_method : str, optional
            Method to use for calculating the background. Options are 'rings', 
            'rings_mean', and 'grimms_ring'. Default is 'rings'.
        t_ring : float, optional
            Thickness of the ring for the 'grimms_ring' method. If not specified, 
            it is set to 1.5 times the radius of the spots.
        show : bool, optional
            If True, display the masked data. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the display function if `show` is True.
    
        Returns
        -------
        res_bgs : np.ndarray
            The mean background value surrounding each Bragg peak (for 'rings' 
            method) or the average background value per pixel (for 'rings_mean' 
            and 'grimms_ring' methods).
        """
        
        bg_methods = ['rings', 'rings_mean', 'grimms_ring']
        assert bg_method in bg_methods, f"Input 'method' must be one of the follwing: {bg_method}"
        
        A, B = self.shape
        dp_center = ((A-1)/2, (B-1)/2)
        
        if bg_method=='rings':
            if not isinstance(r_spots, tuple):
                raise ValueError(""""For the 'rings' method, the input parameter 
                                     'r_spots' must be a tuple specifying the inner 
                                     and outer radius of bg. region arounf each each spot""")
            
            res_bgs = np.zeros(len(centers))
            dp_mask = np.zeros_like(self.array, dtype=bool) if show else None
            # We collect multiple values for the intensities corresponding to each Bragg peak
            for c_idx, center in enumerate(centers):
                cy, cx = center
                extent = abs(r_spots[1])
                y0 = max(0, int(np.floor(cy - extent)))
                y1 = min(A, int(np.ceil(cy + extent)) + 1)
                x0 = max(0, int(np.floor(cx - extent)))
                x1 = min(B, int(np.ceil(cx + extent)) + 1)
                if y0 >= y1 or x0 >= x1:
                    res_bgs[c_idx] = np.nan
                    continue
                local_mask = make_mask(
                    (cy - y0, cx - x0), r_spots,
                    mask_dim=(y1 - y0, x1 - x0),
                )
                values = self.array[y0:y1, x0:x1][local_mask]
                res_bgs[c_idx] = values.mean() if values.size else np.nan
                if show:
                    dp_mask[y0:y1, x0:x1] |= local_mask
            if show:
                self._spawn(self.array * dp_mask).show(**kwargs)
            return res_bgs
     
        if bg_method=='rings_mean':            
            if not isinstance(r_spots, tuple):
                raise ValueError(""""For the 'rings_mean' method, the input parameter 
                                     'r_spots' must be a tuple specifying the inner 
                                     and outer radius of bg. region arounf each each spot""")
            # In this case, a ring is drawn araound every spot
            bool_mask = make_mask(centers, r_spots, mask_dim=(A,B))
        
        if bg_method=='grimms_ring':
            if isinstance(r_spots, tuple):
                raise ValueError(""""For the 'grimms_ring' method, the input parameter 
                                     'r_spots' must be an integer or float specifying 
                                     the radius to blank each spot""")
            
            # If not defined, we define a ring thickness equal to 1.5 times the radii of the enclosed Bragg peaks
            # Note that larger ring thickness may result in inaccurate results due to enclosing signal from other
            # Bragg peaks
            if isinstance(r_spots, (list, np.ndarray)):
                r_mean = np.mean(np.array(r_spots))
            else:
                r_mean = r_spots
            if t_ring is None:
                t_ring = 1.5*r_mean
            elif not isinstance(t_ring, (int, float)):
                raise ValueError("'t_ring' must be an integer or float.")
            elif t_ring > 3*r_mean or t_ring < 1.5:
                raise ValueError("""The input parameter t_ring must be within the 
                                    range [1.5, 3*r_spots]""")
            
            # We automatically define the mean radius of the ring based on Bragg peak positions
            distances = np.sqrt((centers[:, 0] - dp_center[0]) ** 2 + (centers[:, 1] - dp_center[1]) ** 2)
            r_ring = np.mean(distances)
            
            # We combine the masks enclosing the Bragg peaks and the annular region covering them
            spots_mask = make_mask(centers, r_mean, mask_dim=(A,B), invert=True)
            annular_mask = make_mask(dp_center, (r_ring - t_ring/2, r_ring + t_ring/2))
            
            bool_mask = np.logical_and(spots_mask, annular_mask)
        
        # Apply mask
        masked_dp = self.array * bool_mask
        
        if show:
            self._spawn(masked_dp).show(**kwargs)
        
        # Return the average background value per pixel
        return np.sum(masked_dp)/np.sum(bool_mask)
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _peak_template(radius, trench_width, kernel_amp, trench_amp):
        """Build a zero-sum disk/trench template for background-neutral matching."""
        size = 2 * int(np.ceil(radius + trench_width)) + 1
        yy, xx = np.indices((size, size))
        distance = np.hypot(yy - size // 2, xx - size // 2)
        disk = distance <= radius
        trench = (distance > radius) & (distance <= radius + trench_width)
        if not np.any(trench):
            raise ValueError(
                "trench_width is too small to cover any pixels at this radius."
            )

        kernel = np.zeros((size, size), dtype=float)
        kernel[disk] = kernel_amp
        kernel[trench] = trench_amp
        support = disk | trench
        kernel[support] -= np.mean(kernel[support])
        kernel.flags.writeable = False
        return kernel

    @staticmethod
    @lru_cache(maxsize=64)
    def _peak_search_geometry(shape, center, r_range, halo):
        """Cache an annular search mask and a kernel-padded correlation ROI."""
        height, width = shape
        cy, cx = center
        if r_range is None:
            y0, y1, x0, x1 = 0, height, 0, width
            valid = np.ones(shape, dtype=bool)
        else:
            r_min, r_max = r_range
            # The halo ensures correlation at every valid center sees exactly
            # the same input pixels as full-image convolution.
            y0 = max(0, int(np.ceil(cy - r_max)) - halo)
            y1 = min(height, int(np.floor(cy + r_max)) + halo + 1)
            x0 = max(0, int(np.ceil(cx - r_max)) - halo)
            x1 = min(width, int(np.floor(cx + r_max)) + halo + 1)
            if y0 >= y1 or x0 >= x1:
                raise ValueError("r_range selects no diffraction-pattern pixels.")
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance = np.hypot(yy - cy, xx - cx)
            valid = (distance >= r_min) & (distance <= r_max)
            if not np.any(valid):
                raise ValueError("r_range selects no diffraction-pattern pixels.")
        valid.flags.writeable = False
        return (y0, y1, x0, x1), valid

    def _peak_search_center(self, center):
        """Resolve a pixel center from an explicit value or current beam metadata."""
        if center is not None:
            return _validate_center_pair(center, 'center')

        metadata = self.center_beam_metadata
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("center_beam_metadata must be a dictionary or None.")
            if 'shape' in metadata:
                try:
                    metadata_shape = tuple(metadata['shape'])
                except TypeError as exc:
                    raise ValueError(
                        "center_beam_metadata['shape'] must be a 2D shape."
                    ) from exc
                if metadata_shape != tuple(self.shape):
                    raise ValueError(
                        "center_beam_metadata has a stale pattern shape; "
                        "provide center explicitly or update the metadata."
                    )
            for key in ('center_px', 'mean_fit_center_px'):
                if key in metadata:
                    return _validate_center_pair(metadata[key], f'metadata[{key!r}]')

        return ((self.shape[0] - 1) / 2.0, (self.shape[1] - 1) / 2.0)

    def get_peaks(self,
                  radius: float,
                  min_distance: int,
                  trench_width: float = 1.0,
                  kernel_amp: float = 1.0,
                  trench_amp: float = -0.5,
                  threshold_abs: float = 1.0,
                  threshold_rel: float = None,
                  r_range: tuple = None,
                  n_fold: int | None = None,
                  sym_mode: str = "none",
                  sym_tolerance_px: float = 2.0,
                  center_tolerance_px: float = 2.0,
                  orbit_min_fraction: float = 0.5,
                  reorder: bool = False,
                  center=None,
                  return_details: bool = False,
                  *, _array=None) -> np.ndarray | PeakDetectionResult:
        """
        Detect peaks in a 2D diffraction pattern via template matching with a
        disk and a negative "trench" kernel, optionally restricted to an
        annular region [r_min, r_max] (to exclude, for example, the central
        beam and far-away, low-signal peaks).
    
        Optionally, enforce n-fold rotational symmetry about the search center
        over the full Bragg peak array (recommended for centrosymmetric
        diffraction patterns with approximate n-fold symmetry).
    
        Notes
        -----
        - It is recommended to use background-subtracted data and to have all
          diffraction patterns center-beam aligned.
        - When using n_fold symmetry, it is generally recommended to have a
          reasonably clean peak detection (thresholds, min_distance) so that
          orbits can be reliably inferred.
        - The template is mean-centered on its disk and trench. Non-finite
          pixels are replaced by the finite-image median for the FFT, and
          candidate centers whose template touches them are excluded.
          A pattern with no finite pixels raises ``ValueError``.
    
        Parameters
        ----------
        radius : float
            Radius of the positive disk (in pixels).
        trench_width : float, optional
            Width of the negative surround ring (in pixels).
        kernel_amp : float, optional
            Disk weight before the template is mean-centered.
        trench_amp : float, optional
            Trench weight before mean-centering. Must be less than
            ``kernel_amp``.
        threshold_abs : float, optional
            Absolute correlation threshold for peak detection.
        threshold_rel : float, optional
            Relative threshold in [0, 1], as a fraction of the maximum valid
            correlation. If both thresholds are supplied, the stricter one
            applies. With ``r_range``, the maximum is measured in the annulus.
        min_distance : int, optional
            Minimum number of pixels separating peaks (for suppression).
        r_range : tuple (r_min, r_max), optional
            If provided, only search for peaks whose radial distance from the
            search center lies within [r_min, r_max] (in pixels).
        center : (ky, kx) or None, optional
            Search center in pixels for the radial range and symmetry. Uses
            ``center_beam_metadata['center_px']`` when available, otherwise
            the geometric midpoint. Explicit input takes precedence.
        n_fold : int or None, optional
            Order of rotational symmetry to enforce (e.g., 4 for 4-fold, 6 for
            6-fold). If None or < 2, no symmetry enforcement is applied.
        sym_mode : {"none", "repair", "prune", "both"}, optional
            How to enforce ``n_fold`` symmetry. ``none`` leaves detected peaks
            unchanged; ``repair`` adds missing peaks in sufficiently complete
            orbits; ``prune`` keeps only sufficiently complete orbits; and
            ``both`` prunes incomplete orbits and repairs retained ones.
        sym_tolerance_px : float, optional
            Maximum Euclidean distance (in pixels) between a detected peak and
            its ideal symmetric position for them to be considered the same.
            Repaired peaks closer than this to a measured or previously
            repaired peak are not added.
        center_tolerance_px : float, optional
            Radial tolerance (in pixels) used to classify peaks as belonging
            to the central beam region, which is excluded from symmetry
            grouping and passed through unchanged (after de-duplication).
        orbit_min_fraction : float, optional
            Minimum fraction of the n_fold orbit that must be present in the
            detected peaks for that orbit to be considered "real". Orbits
            below this fraction are treated as noise and ignored in "prune"
            and "both" modes (and will not be repaired in "repair" mode).
        return_details : bool, optional
            Return a :class:`PeakDetectionResult` containing correlation
            scores and measured/synthetic labels instead of coordinates alone.

        Returns
        -------
        coords : (M, 2) np.ndarray
            Array of (y, x) coordinates of detected peaks (possibly augmented
            and/or pruned by symmetry enforcement). If ``return_details`` is
            True, return a :class:`PeakDetectionResult` instead.
    
        by Adan J. Mireles
        Applied Physics Graduate Program, Rice University
    
        July 2025
        """

        def _finite_scalar(value, name, *, positive=False, nonnegative=False):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite scalar.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite scalar.") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar.")
            if positive and value <= 0:
                raise ValueError(f"{name} must be positive.")
            if nonnegative and value < 0:
                raise ValueError(f"{name} must be non-negative.")
            return value

        if self.is_polar:
            raise ValueError(
                "get_peaks requires a Cartesian diffraction pattern; "
                "convert polar data to Cartesian coordinates first."
            )
        dp = np.asarray(self.array if _array is None else _array)
        if dp.ndim != 2 or not (
            np.issubdtype(dp.dtype, np.number)
            or np.issubdtype(dp.dtype, np.bool_)
        ) or np.iscomplexobj(dp):
            raise TypeError("get_peaks requires a real-valued 2D numeric pattern.")
        if dp.shape != self.shape:
            raise ValueError("Peak detector pattern shape differs from its search geometry.")
        if not isinstance(return_details, (bool, np.bool_)):
            raise ValueError("return_details must be a Boolean value.")

        radius = _finite_scalar(radius, 'radius', positive=True)
        trench_width = _finite_scalar(
            trench_width, 'trench_width', positive=True,
        )
        kernel_amp = _finite_scalar(kernel_amp, 'kernel_amp')
        trench_amp = _finite_scalar(trench_amp, 'trench_amp')
        if kernel_amp <= trench_amp:
            raise ValueError("kernel_amp must exceed trench_amp.")
        if isinstance(min_distance, (bool, np.bool_)) or not isinstance(
            min_distance, (Integral, np.integer)
        ) or min_distance < 1:
            raise ValueError("min_distance must be a positive integer.")
        min_distance = int(min_distance)

        if threshold_abs is not None:
            threshold_abs = _finite_scalar(threshold_abs, 'threshold_abs')
        if threshold_rel is not None:
            threshold_rel = _finite_scalar(
                threshold_rel, 'threshold_rel', nonnegative=True,
            )
            if threshold_rel > 1:
                raise ValueError("threshold_rel must be between 0 and 1.")

        if r_range is not None:
            try:
                limits = np.asarray(r_range, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "r_range must be a finite (r_min, r_max) pair."
                ) from exc
            if (
                limits.shape != (2,) or not np.all(np.isfinite(limits))
                or limits[0] < 0 or limits[0] >= limits[1]
            ):
                raise ValueError(
                    "r_range must satisfy 0 <= r_min < r_max."
                )
            r_min, r_max = map(float, limits)

        if n_fold is not None and (
            isinstance(n_fold, (bool, np.bool_))
            or not isinstance(n_fold, (Integral, np.integer))
            or n_fold < 1
        ):
            raise ValueError("n_fold must be a positive integer or None.")
        if not isinstance(sym_mode, str) or sym_mode not in (
            'none', 'repair', 'prune', 'both'
        ):
            raise ValueError(
                "sym_mode must be 'none', 'repair', 'prune', or 'both'."
            )
        if sym_mode != 'none' and (n_fold is None or n_fold < 2):
            raise ValueError("sym_mode requires n_fold >= 2.")
        if not isinstance(reorder, (bool, np.bool_)):
            raise ValueError("reorder must be a Boolean value.")
        if reorder and (n_fold is None or n_fold < 2):
            raise ValueError("reorder=True requires n_fold >= 2.")
        sym_tolerance_px = _finite_scalar(
            sym_tolerance_px, 'sym_tolerance_px', positive=True,
        )
        center_tolerance_px = _finite_scalar(
            center_tolerance_px, 'center_tolerance_px', nonnegative=True,
        )
        orbit_min_fraction = _finite_scalar(
            orbit_min_fraction, 'orbit_min_fraction', positive=True,
        )
        if orbit_min_fraction > 1:
            raise ValueError("orbit_min_fraction must be in (0, 1].")

        Cy, Cx = self._peak_search_center(center)

        def _enforce_nfold_symmetry(coords, is_valid):
            """Select disjoint measured orbits, then fill only valid missing slots."""
            radii = np.hypot(coords[:, 0] - Cy, coords[:, 1] - Cx)
            center_mask = radii <= center_tolerance_px
            ring_global = np.flatnonzero(~center_mask)
            ring = coords[ring_global].astype(float)
            ring_radii = radii[ring_global]
            orbit_ids = np.full(len(coords), -1, dtype=int)
            if not len(ring):
                return coords, np.zeros(len(coords), dtype=bool), orbit_ids

            angles = np.arange(n_fold) * (2.0 * np.pi / n_fold)
            cosines, sines = np.cos(angles), np.sin(angles)
            tolerance2 = sym_tolerance_px ** 2
            candidates = []
            for seed, (y, x) in enumerate(ring):
                dy, dx = y - Cy, x - Cx
                expected = np.column_stack((
                    Cy + cosines * dy - sines * dx,
                    Cx + sines * dy + cosines * dx,
                ))
                # Match each orbit slot to a distinct measured peak. Dummy
                # columns let the assignment represent genuinely missing slots.
                nearby = np.flatnonzero(
                    np.abs(ring_radii - ring_radii[seed]) <= sym_tolerance_px
                )
                nearby = nearby[nearby != seed]
                observed = {seed}
                missing = []
                residual = 0.0
                if n_fold > 1:
                    distances2 = np.sum(
                        (expected[1:, None, :] - ring[nearby][None, :, :]) ** 2,
                        axis=2,
                    )
                    dummy_cost = tolerance2 + 1.0
                    costs = np.full((n_fold - 1, len(nearby) + n_fold - 1), dummy_cost)
                    costs[:, :len(nearby)] = np.where(
                        distances2 <= tolerance2, distances2, dummy_cost + 1.0,
                    )
                    rows, columns = linear_sum_assignment(costs)
                    for row, column in zip(rows, columns):
                        if column < len(nearby) and distances2[row, column] <= tolerance2:
                            observed.add(int(nearby[column]))
                            residual += float(distances2[row, column])
                        else:
                            missing.append(expected[row + 1])
                if len(observed) / n_fold >= orbit_min_fraction:
                    candidates.append((
                        -len(observed), residual, seed, observed, missing,
                    ))

            used = set()
            synthetic = []
            synthetic_ids = []
            orbit_number = 0
            for _, _, _, observed, missing in sorted(candidates, key=lambda c: c[:3]):
                if not used.isdisjoint(observed):
                    continue
                used.update(observed)
                orbit_ids[ring_global[list(observed)]] = orbit_number
                if sym_mode in ('repair', 'both'):
                    for position in missing:
                        pixel = np.rint(position).astype(int)
                        if not is_valid(pixel[0], pixel[1]):
                            continue
                        if np.any(np.sum((coords - pixel) ** 2, axis=1) <= tolerance2):
                            continue
                        if synthetic and np.any(
                            np.sum((np.asarray(synthetic) - pixel) ** 2, axis=1)
                            <= tolerance2
                        ):
                            continue
                        synthetic.append(pixel)
                        synthetic_ids.append(orbit_number)
                orbit_number += 1

            if sym_mode == 'repair':
                keep = np.ones(len(coords), dtype=bool)
            else:
                keep = center_mask | (orbit_ids >= 0)
            kept = coords[keep]
            kept_ids = orbit_ids[keep]
            if synthetic:
                kept = np.vstack((kept, np.asarray(synthetic, dtype=int)))
                kept_ids = np.concatenate((kept_ids, synthetic_ids))
            synthetic_mask = np.r_[
                np.zeros(np.count_nonzero(keep), dtype=bool),
                np.ones(len(synthetic), dtype=bool),
            ]
            return kept, synthetic_mask, kept_ids
        
        def _reorder_peaks(coords: np.ndarray,
                           Cy: float,
                           Cx: float,
                           n_fold: int,
                           center_tolerance_px: float,
                           sym_tolerance_px: float) -> np.ndarray:
            """
            Reorder peaks in shells (orbits) of size n_fold.
        
            Strategy
            --------
            - Separate central peaks (within center_tolerance_px) from the rest.
            - On off-center peaks:
                * Work in polar coordinates (radius, angle).
                * For each unused seed peak (starting from smallest radius), build an
                  n_fold rotational orbit by rotating its position by k * (2π / n_fold)
                  around the center and matching the nearest unused peaks within a
                  distance sym_tolerance_px.
                * If a full orbit of size n_fold is found, commit that orbit as one
                  "shell" and mark its peaks as used.
            - After all possible orbits are built:
                * Sort shells by increasing mean radius.
                * Within each shell, sort by increasing angle in [0, 2π).
                * Append any remaining unassigned peaks at the end, sorted by radius.
        
            Coordinates are returned in cartesian form (y, x) with their original
            floating-point values preserved (no rounding to int).
            """
            if coords.size == 0:
                return coords
        
            coords_f = np.asarray(coords, float)
            ys = coords_f[:, 0]
            xs = coords_f[:, 1]
        
            dy = ys - Cy
            dx = xs - Cx
            radii = np.hypot(dx, dy)
        
            # Separate central peaks
            center_mask = radii <= center_tolerance_px
            center_coords = coords_f[center_mask]
            ring_coords = coords_f[~center_mask]
        
            # Optionally sort central peaks by radius (they should all be ~0)
            if center_coords.size > 0:
                cy = center_coords[:, 0]
                cx = center_coords[:, 1]
                cr = np.hypot(cx - Cx, cy - Cy)
                center_order = np.argsort(cr)
                center_coords = center_coords[center_order]
        
            # If no off-center peaks, just return central peaks (floats)
            if ring_coords.size == 0:
                return center_coords
        
            # Polar coordinates for off-center peaks
            ys_r = ring_coords[:, 0]
            xs_r = ring_coords[:, 1]
            dy_r = ys_r - Cy
            dx_r = xs_r - Cx
            radii_r = np.hypot(dx_r, dy_r)
            angles_r = np.arctan2(dy_r, dx_r)
            angles_r = np.mod(angles_r, 2.0 * np.pi)  # map to [0, 2π)
        
            Nr = len(ring_coords)
            used = np.zeros(Nr, dtype=bool)
            tried = np.zeros(Nr, dtype=bool)
        
            # Seed order: increasing radius
            seed_order = np.argsort(radii_r)
        
            theta = 2.0 * np.pi / float(n_fold)
            tol2 = sym_tolerance_px ** 2
        
            def rotate_point(y, x, k):
                """Rotate (y, x) by k * theta about (Cy, Cx)."""
                dy_ = y - Cy
                dx_ = x - Cx
                ang = k * theta
                ca = np.cos(ang)
                sa = np.sin(ang)
                dy_r_ = ca * dy_ - sa * dx_
                dx_r_ = sa * dy_ + ca * dx_
                return Cy + dy_r_, Cx + dx_r_
        
            groups_idx = []
        
            # Build rotational orbits
            for i in seed_order:
                if used[i] or tried[i]:
                    continue
        
                tried[i] = True
                y0 = ring_coords[i, 0]
                x0 = ring_coords[i, 1]
        
                local_indices = []
                local_used = set()
        
                for k in range(n_fold):
                    yk, xk = rotate_point(y0, x0, k)
        
                    best_j = None
                    best_d2 = tol2
        
                    # Find nearest unused peak to the expected rotated position
                    for j in range(Nr):
                        if used[j] or (j in local_used):
                            continue
                        dy_ = ring_coords[j, 0] - yk
                        dx_ = ring_coords[j, 1] - xk
                        d2 = dy_ * dy_ + dx_ * dx_
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_j = j
        
                    if best_j is None:
                        # This seed cannot form a complete n_fold orbit under tolerance
                        local_indices = []
                        break
        
                    local_indices.append(best_j)
                    local_used.add(best_j)
        
                # Commit only full orbits
                if len(local_indices) == n_fold:
                    for j in local_indices:
                        used[j] = True
                    groups_idx.append(local_indices)
        
            # Sort groups (orbits) by mean radius
            group_radii = [np.mean(radii_r[idxs]) for idxs in groups_idx]
            group_order = np.argsort(group_radii)
        
            ordered_ring_list = []
        
            for g in group_order:
                idxs = np.array(groups_idx[g], dtype=int)
                # Sort within group by angle
                ang_g = angles_r[idxs]
                order_g = np.argsort(ang_g)
                ordered_ring_list.append(ring_coords[idxs[order_g]])
        
            if ordered_ring_list:
                ordered_ring = np.vstack(ordered_ring_list)
            else:
                ordered_ring = ring_coords[:0]
        
            # Append any remaining unused peaks (no complete orbit) at the end
            unused_idx = np.where(~used)[0]
            if unused_idx.size > 0:
                # Sort leftover by radius before appending
                order_unused = np.argsort(radii_r[unused_idx])
                remainder = ring_coords[unused_idx[order_unused]]
                if ordered_ring.size > 0:
                    ordered_ring = np.vstack([ordered_ring, remainder])
                else:
                    ordered_ring = remainder
        
            # Combine central and ring peaks (all as float coordinates)
            if center_coords.size > 0:
                combined = np.vstack([center_coords, ordered_ring])
            else:
                combined = ordered_ring
        
            return combined

        
        # -------------------------------------------------------------------------
        # Template-matching peak detection
        # -------------------------------------------------------------------------
    
        kernel = self._peak_template(
            radius, trench_width, kernel_amp, trench_amp,
        )
        (y0, y1, x0, x1), base_valid = self._peak_search_geometry(
            tuple(dp.shape), (Cy, Cx),
            None if r_range is None else (r_min, r_max),
            kernel.shape[0] // 2,
        )
        dp_work = np.array(dp[y0:y1, x0:x1], dtype=float, copy=True)
        local_finite = np.isfinite(dp_work)
        if not np.any(local_finite):
            if not np.any(np.isfinite(dp)):
                raise ValueError("Diffraction pattern contains no finite values.")
            empty = np.empty((0, 2), dtype=int)
            if return_details:
                return PeakDetectionResult(
                    empty, np.empty(0), np.empty(0, dtype=bool),
                    np.empty(0, dtype=int),
                )
            return empty
        if (
            r_range is not None and np.all(local_finite)
            and y0 > 0 and y1 < dp.shape[0]
            and x0 > 0 and x1 < dp.shape[1]
        ):
            # The zero-sum kernel cancels a constant baseline at every valid
            # center when the padded ROI stays inside the detector.
            baseline = float(np.median(dp_work))
        else:
            finite = np.isfinite(dp)
            baseline = float(np.median(dp[finite]))
        dp_work[~local_finite] = baseline
        dp_work -= baseline
        corr = fftconvolve(dp_work, kernel[::-1, ::-1], mode='same')

        H, W = dp.shape
        valid_centers = base_valid.copy()

        if not np.all(local_finite):
            invalid_footprint = ndimage.binary_dilation(
                ~local_finite, structure=(kernel != 0), border_value=0,
            )
            valid_centers &= ~invalid_footprint
        if not np.any(valid_centers):
            empty = np.empty((0, 2), dtype=int)
            if return_details:
                return PeakDetectionResult(
                    empty, np.empty(0), np.empty(0, dtype=bool),
                    np.empty(0, dtype=int),
                )
            return empty

        if threshold_rel is not None:
            peak_response = float(np.max(corr[valid_centers]))
            if peak_response <= 0:
                empty = np.empty((0, 2), dtype=int)
                if return_details:
                    return PeakDetectionResult(
                        empty, np.empty(0), np.empty(0, dtype=bool),
                        np.empty(0, dtype=int),
                    )
                return empty
            relative_threshold = np.nextafter(
                threshold_rel * peak_response, -np.inf,
            )
            threshold_abs = (
                relative_threshold if threshold_abs is None
                else max(threshold_abs, relative_threshold)
            )

        if not np.all(valid_centers):
            corr = corr.copy()
            corr[~valid_centers] = -np.inf

        # Local maxima in correlation map
        coords = peak_local_max(
            corr,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
    
        if coords.size:
            coords = coords[valid_centers[coords[:, 0], coords[:, 1]]]
        score_lookup = {
            (int(y + y0), int(x + x0)): float(corr[y, x])
            for y, x in coords
        }
        coords = coords + np.array((y0, x0), dtype=int)
        synthetic_mask = np.zeros(len(coords), dtype=bool)
        orbit_ids = np.full(len(coords), -1, dtype=int)

        def is_valid(y, x):
            return (
                0 <= y < H and 0 <= x < W
                and y0 <= y < y1 and x0 <= x < x1
                and bool(valid_centers[y - y0, x - x0])
            )
    
        # Optional n-fold symmetry enforcement on the full peak array
        if (
            n_fold is not None
            and n_fold >= 2
            and coords.size > 0
            and sym_mode != "none"
        ):
            coords, synthetic_mask, orbit_ids = _enforce_nfold_symmetry(
                coords, is_valid,
            )
        
        # Optional reordering into shells of size n_fold
        if (reorder
            and n_fold is not None
            and n_fold >= 2
            and coords.size > 0
        ):
            details_by_coord = {
                tuple(map(int, point)): (is_synthetic, orbit)
                for point, is_synthetic, orbit in zip(
                    coords, synthetic_mask, orbit_ids,
                )
            }
            coords = _reorder_peaks(
                coords=coords,
                Cy=Cy,
                Cx=Cx,
                n_fold=n_fold,
                center_tolerance_px=center_tolerance_px,
                sym_tolerance_px=sym_tolerance_px
            )
            synthetic_mask = np.array([
                details_by_coord[tuple(map(int, point))][0]
                for point in coords
            ], dtype=bool)
            orbit_ids = np.array([
                details_by_coord[tuple(map(int, point))][1]
                for point in coords
            ], dtype=int)

        if coords.size:
            pixel_coords = np.rint(coords).astype(int)
            within_image = (
                (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < H)
                & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < W)
            )
            coords = coords[within_image]
            synthetic_mask = synthetic_mask[within_image]
            orbit_ids = orbit_ids[within_image]
            pixel_coords = pixel_coords[within_image]
            valid = np.array([
                is_valid(y, x) for y, x in pixel_coords
            ], dtype=bool)
            coords = coords[valid]
            synthetic_mask = synthetic_mask[valid]
            orbit_ids = orbit_ids[valid]

        coords = np.rint(coords).astype(int).reshape(-1, 2)
        if not return_details:
            return coords
        scores = np.array([
            np.nan if synthetic else score_lookup[tuple(point)]
            for point, synthetic in zip(coords, synthetic_mask)
        ], dtype=float)
        return PeakDetectionResult(coords, scores, synthetic_mask, orbit_ids)

    def clip(self, a_min=1, a_max=None):
        """
        Clip diffraction-pattern intensities to a specified range.

        Values below ``a_min`` are set to ``a_min`` and values above
        ``a_max`` are set to ``a_max``. If either bound is ``None``, clipping
        on that side is skipped. The original object is not modified.

        Parameters
        ----------
        a_min : float or None, optional
            Lower clipping bound. Defaults to 1. Use ``None`` to disable lower
            clipping.
        a_max : float or None, optional
            Upper clipping bound. Defaults to None, meaning no upper clipping
            is applied.

        Returns
        -------
        ReciprocalSpace
            New diffraction-pattern object with clipped intensities and
            preserved calibration/metadata.
        """
        
        return self._spawn(clip_values(self.array, a_min, a_max))
    
    def inpaint_background(self, centers=None, radius=6, mask=None,
                           method='biharmonic', split_into_regions=False):
        """
        Estimate a local background by inpainting selected diffraction spots.

        Circular regions centered at ``centers`` with radius ``radius`` are
        treated as spots to remove. An explicit Boolean ``mask`` can also be
        provided, and is combined with any circular regions. The missing
        regions are filled by ``inpaint_background`` to produce a smooth
        background estimate. The original diffraction pattern is not modified.

        Parameters
        ----------
        centers : array-like or None, optional
            One ``(ky, kx)`` center or a sequence of centers identifying spots
            to inpaint. Required when ``mask`` is not provided.
        radius : float or array-like of floats, optional
            Radius, in pixels, around each center to inpaint. Defaults to 6.
        mask : ndarray of bool or None, optional
            Explicit inpainting mask with the same shape as this diffraction
            pattern. True-valued pixels are inpainted.
        method : {'biharmonic'}, optional
            Inpainting method. Currently only ``'biharmonic'`` is implemented.
        split_into_regions : bool, optional
            Passed to ``skimage.restoration.inpaint_biharmonic``.

        Returns
        -------
        ReciprocalSpace
            New diffraction-pattern object containing the estimated
            background and preserved calibration/metadata.
        """

        return self._spawn(
            inpaint_background(
                self.array,
                centers=centers,
                radius=radius,
                mask=mask,
                method=method,
                split_into_regions=split_into_regions,
            )
        )

    def remove_bg(self, background, bg_frac=1, a_min=1):
    
        """
        Subtracts a fraction of the background from the dataset and clips the 
        result to handle underflows.
        
        Parameters
        ----------
        background : ndarray
            The background data array which must be of the same shape as self.array.
        bg_frac : float, optional
            The fraction of the background to be subtracted from the dataset. 
            Must be between 0 and 1 (inclusive).
        
        Returns
        -------
        ReciprocalSpace
            A new instance of ReciprocalSpace with the background subtracted 
            diffraction pattern.
        
        Raises
        ------
        ValueError
            If bg_frac is not within the required range [0, 1] or 
        """
        if not (0 <= bg_frac <= 1):
            raise ValueError("'bg_frac' must be between 0 and 1, inclusive.")
        
        if background.shape != self.shape:
            raise ValueError("""'background' must match the shape of the diffraction pattern.""")
        
        if type(background) != np.ndarray:
            background = background.array
        
        return self._spawn(self.array - background*bg_frac).clip(a_min=a_min)


    def get_radialProfile(self,
                          r_min: float = 0,
                          r_max: float = None,
                          plot: bool = False,
                          mask: np.ndarray = None,
                          centers: Union[Sequence[Tuple[float, float]], np.ndarray] = None,
                          r: float = None,
                          title: str = None,
                          return_radialProfile: bool = True,
                          return_integral: bool = False,
                          units: str = None,
                          conv_factor: float = None,
                          **plot_kwargs):
        """
        Compute the mean radial intensity profile (and optionally its integral)
        of this 2D diffraction pattern.

        Parameters
        ----------
        r_min : float
            Minimum radius (in pixels) to include. Defaults to 0.
        r_max : float
            Maximum radius to include. Defaults to distance from center to corner.
        plot : bool
            If True, plot log(Intensity) vs. r (or converted units).
        mask : 2D bool array
            If provided, only True pixels are used.
        centers : array-like of shape (N,2) or (2,N)
            If provided, disks of radius `r` around these (ky,kx) coords are excluded.
        r : float
            Radius of exclusion disks around `centers`. Required if `centers` is not None.
        title : str
            Custom plot title.
        return_radialProfile : bool
            If True, return the 1D radial profile.
        return_integral : bool
            If True, return the integral under the radial profile.
        units : str, optional
            Physical units for the radial axis. Accepts any of:
              - inverse Å⁻¹: "inv_ang", "inv_Ang", "invAng", "invang", "a-1", "A-1"
              - milliradians: "mrad"
              - degrees: "deg", "degree"
            If provided, `conv_factor` must also be given.
        conv_factor : float, optional
            Multiply raw pixel–radii by this to get the chosen `units`.
        **plot_kwargs : dict
            Forwarded to plt.plot(), e.g. color, linestyle, linewidth, marker.

        Returns
        -------
        radial_profile : np.ndarray, shape (M,)
            Mean intensity for radii from r_min up to r_max.
        integral : float
            (Optional) Integral under the radial_profile curve
            (in the same units as the x-axis).
        """

        ky, kx = self.shape
        cy, cx = (ky - 1) / 2.0, (kx - 1) / 2.0

        # default outer radius
        if r_max is None:
            r_max = np.hypot(cy, cx)

        # build distance map
        Y, X = np.indices(self.shape)
        distances = np.hypot(Y - cy, X - cx)

        # initial valid‐pixel mask
        valid = np.ones(self.shape, dtype=bool)
        if mask is not None:
            if mask.shape != self.shape:
                raise ValueError("`mask` must have same shape as data")
            valid &= mask

        # exclude disks around centers if requested
        if centers is not None:
            if r is None:
                raise ValueError("`r` must be provided when `centers` is not None")
            centers_arr = np.asarray(centers)
            # handle shape (2,N)
            if centers_arr.ndim == 2 and centers_arr.shape[0] == 2:
                centers_arr = centers_arr.T
            for (cy_i, cx_i) in centers_arr:
                valid &= (np.hypot(Y - cy_i, X - cx_i) > r)

        # bin into integer radii
        max_bin = int(np.floor(r_max)) + 1
        bin_idx = distances.astype(int)

        # sum and count per bin
        sums   = np.bincount(bin_idx[valid].ravel(),
                             weights=self.array[valid].ravel(),
                             minlength=max_bin)
        counts = np.bincount(bin_idx[valid].ravel(),
                             minlength=max_bin)

        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            profile = sums / counts

        # trim to [r_min, r_max]
        # make radii match the profile length exactly
        radii = np.arange(profile.size)
        sel   = (radii >= r_min) & (radii <= r_max)
        raw_profile = profile[sel]
        raw_radii   = radii[sel]

        # units + conversion
        units, conv_factor = self._resolve_scale(units=units, conv_factor=conv_factor)
        if units is not None:
            phys_r = raw_radii * conv_factor
            unit_text = self._format_unit_text(units)
            normalized = units.lower().replace(" ", "")
            if normalized in {'inv_ang', 'invang', 'a-1', 'a^-1', 'å^-1', 'å-1', 'ang^-1', 'ang-1'}:
                xlabel = rf"Frequency ({unit_text})"
            elif normalized in {'mrad', 'mrads'}:
                xlabel = f"Scattering angle ({unit_text})"
            elif normalized in {'deg', 'degree', 'degrees'}:
                xlabel = f"Scattering angle ({unit_text})"
            else:
                xlabel = f"r ({unit_text})"
        else:
            phys_r = raw_radii
            xlabel = "r (px)"

        if plot:
            plt.figure()
            plt.plot(phys_r, np.log(raw_profile), **plot_kwargs)
            plt.xlabel(xlabel)
            plt.ylabel('log(Intensity)')
            plt.title(title if title else 'Radial Profile')
            plt.xlim(phys_r.min(), phys_r.max())
            plt.show()

        outs = []
        if return_radialProfile:
            outs.append(raw_profile)
        if return_integral:
            integral = np.trapz(raw_profile, phys_r)
            outs.append(integral)

        if not outs:
            return None
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)
    
    def select_peaks(self,
              n_points=None,
              **imshow_kwargs):
        """
        Manually select peaks on this 2D diffraction pattern.
    
        Parameters
        ----------
        n_points : int or None
            Maximum number of points to select with mouse clicks.
            If None, selection continues until the user presses Enter.
        **imshow_kwargs : dict
            Passed directly to ax.imshow(); e.g. vmin=…, vmax=…, cmap='inferno', origin='lower', etc.
    
        Returns
        -------
        coords : np.ndarray, shape (N, 2)
            Array of (y, x) locations (in pixel coords) of the selected points.
        """
    
        fig, ax = plt.subplots()
        # show log of intensity
        img = ax.imshow(np.log(self.array), **imshow_kwargs)
        fig.colorbar(img, ax=ax)
        ax.set_title(
            "Click to select peaks.\n"
            + ("Select up to %d points." % n_points if n_points else "Press Enter when done.")
        )
        # n=n_points if int, else n=0 for unlimited until Enter
        ginput_n = n_points if isinstance(n_points, int) else 0
        pts = plt.ginput(n=ginput_n, timeout=0)
        plt.close(fig)
    
        if len(pts) == 0:
            return np.empty((0, 2), float)
    
        # ginput returns list of (x, y) in data coords; convert to (y, x)
        arr = np.array(pts)
        coords = np.column_stack((arr[:, 1], arr[:, 0]))
        return coords

#%% Real-space Class

class RealSpace:
    """
    Container for a single 2D real-space image.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional real-space data.
    units : str or None, optional
        Physical units associated with the real-space pixel spacing
        (for example ``'nm'`` or ``'Å'``).
    conv_factor : float or (float, float) or None, optional
        Physical units per pixel. A scalar means equal ``(y, x)`` spacing;
        a pair supports rectangular scan pixels.
    origin : (float, float), optional
        Physical ``(y, x)`` coordinate of pixel center ``(0, 0)``. Defaults
        to ``(0, 0)`` and is preserved through crop and resize.
    quantity, value_units : str or None, optional
        Meaning and units of pixel values, used to label the colorbar.
    """

    def __init__(self, data, units: str = None, conv_factor=None,
                 origin=(0.0, 0.0), quantity='Intensity', value_units=None):
        """Validate a 2D image and attach its optional spatial calibration."""
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError("RealSpace requires a 2D image.")
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ) or np.iscomplexobj(array):
            raise TypeError("RealSpace requires a real-valued numeric or Boolean image.")
        if not isinstance(quantity, str) or not quantity.strip():
            raise ValueError("quantity must be a non-empty string.")
        if value_units is not None and (
            not isinstance(value_units, str) or not value_units.strip()
        ):
            raise ValueError("value_units must be a non-empty string or None.")

        self.array = array
        self.shape = array.shape
        self.units = None
        self.conv_factor = None
        self.origin = _normalize_real_origin(origin)
        self.quantity = quantity.strip()
        self.value_units = value_units.strip() if value_units is not None else None

        if units is not None or conv_factor is not None:
            self.set_scale(units=units, conv_factor=conv_factor)

    @property
    def pixel_size(self):
        """Return calibrated ``(y, x)`` pixel spacing, or None."""
        return None if self.conv_factor is None else _real_spacing_pair(self.conv_factor)

    def set_scale(self, units: str, conv_factor):
        """Attach scalar or ``(y, x)`` real-space calibration."""
        if units is None or conv_factor is None:
            raise ValueError("'units' and 'conv_factor' must both be provided.")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("'units' must be a non-empty string.")

        factor = _normalize_real_spacing(conv_factor)
        self.units = units.strip()
        self.conv_factor = factor
        return self

    def set_origin(self, origin):
        """Set the physical ``(y, x)`` coordinate of pixel ``(0, 0)``."""
        self.origin = _normalize_real_origin(origin)
        return self

    def clear_scale(self):
        """Remove any stored real-space calibration."""
        self.units = None
        self.conv_factor = None
        self.origin = (0.0, 0.0)
        return self

    def _format_unit_text(self, units):
        """Return a display-friendly unit label."""
        if units is None:
            return "px"

        normalized = units.lower().replace(" ", "")
        if normalized in {'ang', 'angstrom', 'angstroms', 'å', 'ångström', 'a'}:
            return "Å"
        if normalized in {'nm', 'nanometer', 'nanometers'}:
            return "nm"
        if normalized in {'um', 'µm', 'micron', 'microns'}:
            return "µm"
        return units

    def _axis_extent(self, conv_factor=None):
        """Return image-edge limits for a top-left row-zero convention."""
        ny, nx = self.shape
        sy, sx = _real_spacing_pair(conv_factor)
        oy, ox = self.origin if conv_factor is not None else (0.0, 0.0)
        return (
            ox - 0.5 * sx, ox + (nx - 0.5) * sx,
            oy + (ny - 0.5) * sy, oy - 0.5 * sy,
        )

    def _spawn(self, data, units=_SCALE_UNSET, conv_factor=_SCALE_UNSET,
               origin=_SCALE_UNSET, quantity=_SCALE_UNSET,
               value_units=_SCALE_UNSET):
        """Create a RealSpace image while preserving metadata."""
        if units is _SCALE_UNSET:
            units = self.units
        if conv_factor is _SCALE_UNSET:
            conv_factor = self.conv_factor
        if origin is _SCALE_UNSET:
            origin = self.origin
        if quantity is _SCALE_UNSET:
            quantity = self.quantity
        if value_units is _SCALE_UNSET:
            value_units = self.value_units
        return RealSpace(
            data, units=units, conv_factor=conv_factor, origin=origin,
            quantity=quantity, value_units=value_units,
        )

    def copy(self):
        """
        Return an independent copy of this RealSpace object.

        The image array and calibration metadata are copied so the returned
        object can be edited without changing this object.
        """
        return self._spawn(np.array(self.array, copy=True))

    def crop(self, ylim=None, xlim=None, selection_units='pixels'):
        """Return a cropped image while retaining its physical origin.

        ``ylim`` and ``xlim`` are indices or half-open ``(start, stop)``
        ranges. Use ``selection_units='calibrated'`` for physical coordinates.
        Pixel selections are the default, even on calibrated images.
        """
        _, factor, mode = _resolve_unit_mode(
            selection_units, self.units, self.conv_factor,
            label='selection_units',
        )
        sy, sx = _real_spacing_pair(factor)
        y0, y1, _ = _parse_real_selection(
            ylim, self.shape[0], 'ylim', mode, sy, self.origin[0],
        )
        x0, x1, _ = _parse_real_selection(
            xlim, self.shape[1], 'xlim', mode, sx, self.origin[1],
        )
        source_sy, source_sx = _real_spacing_pair(self.conv_factor)
        new_origin = (
            self.origin[0] + y0 * source_sy,
            self.origin[1] + x0 * source_sx,
        )
        return self._spawn(self.array[y0:y1, x0:x1], origin=new_origin)

    def resize(self, shape, method='area'):
        """Resize the image and update pixel spacing and pixel-center origin.

        ``method='area'`` computes area-weighted downsampling. ``'linear'``
        and ``'nearest'`` also support upsampling. Pixel values are preserved
        in their original range; integer data may become floating point.
        """
        values = np.asarray(shape, dtype=object)
        if values.shape != (2,) or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Integral, np.integer)) or value < 1
            for value in values
        ):
            raise ValueError("shape must be a positive integer (y, x) pair.")
        output_shape = tuple(int(value) for value in values)
        if method not in ('area', 'linear', 'nearest'):
            raise ValueError("method must be 'area', 'linear', or 'nearest'.")
        if method == 'area':
            if any(new > old for new, old in zip(output_shape, self.shape)):
                raise ValueError("method='area' supports downsampling only.")
            resized = HyperData._resize_area(
                self.array, output_shape, axes=(0, 1),
            )
        else:
            order = 1 if method == 'linear' else 0
            resized = transform.resize(
                self.array.astype(float) if order == 1 and self.array.dtype == bool
                else self.array,
                output_shape, order=order, mode='reflect',
                anti_aliasing=(order > 0 and any(
                    new < old for new, old in zip(output_shape, self.shape)
                )),
                preserve_range=True,
            )
            if order == 0:
                resized = resized.astype(self.array.dtype, copy=False)

        factors = tuple(old / new for old, new in zip(self.shape, output_shape))
        sy, sx = _real_spacing_pair(self.conv_factor)
        new_origin = (
            self.origin[0] + 0.5 * (factors[0] - 1) * sy,
            self.origin[1] + 0.5 * (factors[1] - 1) * sx,
        )
        new_factor = (
            None if self.conv_factor is None
            else _scaled_real_spacing(self.conv_factor, factors)
        )
        return self._spawn(resized, conv_factor=new_factor, origin=new_origin)

    @staticmethod
    def _draw_scale_bar(ax, length, unit_text, color, position, label, extent):
        """Draw a real-space bar in displayed coordinate units."""
        left, right, bottom, top = extent
        width = abs(right - left)
        height = abs(bottom - top)
        margin = 0.06 * width
        if length > width - 2 * margin:
            raise ValueError("scale_bar is too long for the displayed image.")
        x_direction = np.sign(right - left)
        if position == 'left':
            x0 = left + x_direction * margin
        elif position == 'right':
            x0 = right - x_direction * (margin + length)
        else:
            x0 = left + x_direction * (width - length) / 2
        x1 = x0 + x_direction * length
        direction = np.sign(bottom - top)
        y0 = bottom - 0.08 * height * direction
        cap = 0.012 * height
        outline = [
            path_effects.Stroke(linewidth=5, foreground='black'),
            path_effects.Normal(),
        ]
        ax.plot(
            [x0, x1], [y0, y0], color=color, linewidth=3,
            zorder=5, path_effects=outline,
        )
        for xpos in (x0, x1):
            ax.plot(
                [xpos, xpos], [y0 - cap, y0 + cap],
                color=color, linewidth=2, zorder=5, path_effects=outline,
            )
        if label:
            ax.text(
                (x0 + x1) / 2, y0 - 2 * cap * direction,
                f"{length:g} {unit_text}", color=color,
                ha='center', va='bottom', zorder=5, path_effects=outline,
            )

    def show(self,
             title: str = 'Real-Space Image',
             axes: bool = True,
             grid: bool = False,
             num_div=10,
             gridColor: str = 'black',
             vmin=None,
             vmax=None,
             figsize=(8, 8),
             aspect=None,
             cmap: str = 'gray',
             coords: np.ndarray | None = None,
             axis_units='auto',
             *, ax=None, show=True, colorbar=None, colorbar_label=None,
             grid_color=None, grid_ticks=None,
             scale_bar=None, scale_bar_position='left',
             scale_bar_color='white', scale_bar_label=True,
             coords_units='pixels', y=None, x=None,
             percentiles=None, symmetric=False,
             **scatter_kwargs):
        """Display the image on new or existing Matplotlib axes.

        Parameters
        ----------
        axis_units : {'auto', 'pixels', 'calibrated'}, optional
            Axis and scale-bar units. ``'auto'`` uses calibration if available.
        ax : matplotlib.axes.Axes or None, optional
            Existing axes for a subplot. If None, create a new figure.
        show : bool, optional
            Call ``plt.show()`` when True. Set False when composing figures.
        colorbar : bool or None, optional
            Draw a colorbar. None follows ``axes``; an explicit value is
            independent of whether image axes are visible.
        colorbar_label : str or None, optional
            Override the label from ``quantity`` and ``value_units``.
        coords, y, x : array-like, optional
            Scatter positions, with ``coords`` in ``(y, x)`` order. Supply
            either ``coords`` or both ``y`` and ``x``. The default input is
            pixel indices even when displayed axes are calibrated.
        coords_units : {'pixels', 'calibrated'}, optional
            Input coordinate system for scatter positions.
        grid_color, grid_ticks : optional
            Grid color and tick count (scalar or ``(y, x)``). ``num_div`` and
            ``gridColor`` remain accepted for existing plotting calls.
        scale_bar : positive float or None, optional
            Length in the displayed axis units. Its location, color, and
            numeric label are controlled by the ``scale_bar_*`` arguments.
        percentiles : (float, float) or None, optional
            Use these finite-data percentiles when ``vmin`` or ``vmax`` is
            omitted. Explicit limits take priority.
        symmetric : bool, optional
            Use color limits symmetric around zero, useful for strain maps.

        Returns
        -------
        tuple[Figure, Axes]
            The figure and image axes. The image is not modified.
        """
        if 'units' in scatter_kwargs or 'conv_factor' in scatter_kwargs:
            raise TypeError(
                "RealSpace.show uses axis_units='auto', 'pixels', or "
                "'calibrated'. Set the object scale with set_scale() instead "
                "of passing units/conv_factor to show()."
            )
        if coords is not None and (y is not None or x is not None):
            raise ValueError("Provide either coords or y and x, not both.")
        if (y is None) != (x is None):
            raise ValueError("Supply both y and x for scatter points.")
        if y is not None:
            y_values = np.atleast_1d(np.asarray(y, dtype=float))
            x_values = np.atleast_1d(np.asarray(x, dtype=float))
            if y_values.ndim != 1 or x_values.ndim != 1 or y_values.shape != x_values.shape:
                raise ValueError("y and x must be equally sized scalar or 1D arrays.")
            coords = np.column_stack((y_values, x_values))
        if coords is not None:
            coords = np.asarray(coords, dtype=float)
            if coords.shape == (2,):
                coords = coords.reshape(1, 2)
            if coords.size == 0:
                coords = np.empty((0, 2), dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("coords must have shape (N, 2) in (y, x) order.")
            if not np.all(np.isfinite(coords)):
                raise ValueError("Scatter coordinates must be finite.")
        elif scatter_kwargs:
            raise TypeError("Scatter styling requires coords or both y and x.")

        input_mode = _normalize_unit_mode(coords_units, label='coords_units')
        if input_mode == 'auto':
            input_mode = 'pixels'
        if input_mode == 'calibrated' and self.conv_factor is None:
            raise ValueError("coords_units='calibrated' requires image calibration.")
        units, conv_factor, _ = _resolve_unit_mode(
            axis_units, self.units, self.conv_factor, label='axis_units',
        )
        axis_unit_text = self._format_unit_text(units)
        extent = self._axis_extent(conv_factor=conv_factor)
        display_sy, display_sx = _real_spacing_pair(conv_factor)
        display_origin = self.origin if conv_factor is not None else (0.0, 0.0)

        if grid_color is None:
            grid_color = gridColor
        to_rgba(grid_color)
        to_rgba(scale_bar_color)
        if scale_bar_position not in ('left', 'center', 'right'):
            raise ValueError("scale_bar_position must be 'left', 'center', or 'right'.")
        if scale_bar is not None:
            if isinstance(scale_bar, (bool, np.bool_)):
                raise ValueError("scale_bar must be positive and finite.")
            try:
                scale_bar = float(scale_bar)
            except (TypeError, ValueError) as exc:
                raise ValueError("scale_bar must be positive and finite.") from exc
            if not np.isfinite(scale_bar) or scale_bar <= 0:
                raise ValueError("scale_bar must be positive and finite.")
            if scale_bar > 0.88 * abs(extent[1] - extent[0]):
                raise ValueError("scale_bar is too long for the displayed image.")

        def tick_counts(value, label, minimum):
            if isinstance(value, (Integral, np.integer)) and not isinstance(value, (bool, np.bool_)):
                value = (value, value)
            if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
                raise ValueError(f"{label} must be an integer or (y, x) pair.")
            if any(
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (Integral, np.integer)) or count < minimum
                for count in value
            ):
                raise ValueError(f"{label} must contain integers >= {minimum}.")
            return tuple(int(count) for count in value)

        if grid_ticks is not None:
            ny_ticks, nx_ticks = tick_counts(grid_ticks, 'grid_ticks', 1)
        elif num_div is None:
            ny_ticks = nx_ticks = None
        else:
            ny_div, nx_div = tick_counts(num_div, 'num_div', 0)
            ny_ticks = ny_div + 1 if ny_div else None
            nx_ticks = nx_div + 1 if nx_div else None

        finite_values = np.asarray(self.array)[np.isfinite(self.array)]
        if finite_values.size == 0:
            raise ValueError("Real-space image has no finite values to display.")
        if percentiles is not None:
            try:
                limits = np.asarray(percentiles, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "percentiles must satisfy 0 <= low < high <= 100."
                ) from exc
            if (
                limits.shape != (2,) or not np.all(np.isfinite(limits))
                or limits[0] < 0 or limits[1] > 100 or limits[0] >= limits[1]
            ):
                raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
            low, high = np.percentile(finite_values, limits)
            vmin = low if vmin is None else vmin
            vmax = high if vmax is None else vmax
        if symmetric:
            low = np.min(finite_values) if vmin is None else vmin
            high = np.max(finite_values) if vmax is None else vmax
            limit = max(abs(low), abs(high))
            vmin, vmax = -limit, limit

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            if not hasattr(ax, 'imshow') or not hasattr(ax, 'figure'):
                raise TypeError("ax must be a Matplotlib Axes instance.")
            fig = ax.figure
        im1 = ax.imshow(
            self.array, vmin=vmin, vmax=vmax, cmap=cmap,
            extent=extent, origin='upper',
        )

        if aspect is not None:
            ax.set_aspect(aspect)

        if coords is not None and coords.size:
            if input_mode == 'calibrated':
                source_sy, source_sx = self.pixel_size
                pixel_y = (coords[:, 0] - self.origin[0]) / source_sy
                pixel_x = (coords[:, 1] - self.origin[1]) / source_sx
            else:
                pixel_y, pixel_x = coords[:, 0], coords[:, 1]
            x_positions = display_origin[1] + pixel_x * display_sx
            y_positions = display_origin[0] + pixel_y * display_sy
            ax.scatter(x_positions, y_positions, **scatter_kwargs)

        x_positions = (
            ax.get_xticks() if nx_ticks is None else np.linspace(
                display_origin[1],
                display_origin[1] + (self.shape[1] - 1) * display_sx,
                nx_ticks,
            )
        )
        y_positions = (
            ax.get_yticks() if ny_ticks is None else np.linspace(
                display_origin[0],
                display_origin[0] + (self.shape[0] - 1) * display_sy,
                ny_ticks,
            )
        )

        if axes:
            ax.set_xlabel(f"x ({axis_unit_text})", fontsize=14)
            ax.set_ylabel(f"y ({axis_unit_text})", fontsize=14)
            if nx_ticks is not None:
                ax.set_xticks(x_positions)
            if ny_ticks is not None:
                ax.set_yticks(y_positions)
            if grid:
                ax.grid(color=grid_color)
        else:
            if grid:
                for xpos in x_positions:
                    ax.axvline(xpos, color=grid_color, linewidth=0.7, zorder=2)
                for ypos in y_positions:
                    ax.axhline(ypos, color=grid_color, linewidth=0.7, zorder=2)
            ax.set_axis_off()

        if title is not None:
            ax.set_title(title, fontsize=18)
        if colorbar is None:
            colorbar = axes
        if colorbar:
            cb = fig.colorbar(im1, ax=ax)
            label = colorbar_label or self.quantity
            if colorbar_label is None and self.value_units is not None:
                label += f" ({self.value_units})"
            cb.set_label(label, fontsize=14)
        if scale_bar is not None:
            self._draw_scale_bar(
                ax, scale_bar, axis_unit_text, scale_bar_color,
                scale_bar_position, scale_bar_label, extent,
            )

        if show:
            plt.show()
        return fig, ax

#%% Denoising Functions and Classes

class _DenoisingMethods:
    """Private collection of numerical denoising algorithms."""
    
    @staticmethod
    def _unfold_with_hyperdata(array, unfold_domain=None, unfold_method='row_major'):
        """
        Unfold an array through the HyperData API and return array + metadata.

        Decomposition methods operate on arrays, but unfolding belongs to
        HyperData. This helper keeps that ownership clear while letting the
        numerical routines work with plain ndarrays.
        """
        if unfold_domain is None:
            return array, None

        unfolded = HyperData(array).unfold(
            domain=unfold_domain,
            method=unfold_method,
        )
        return unfolded.array, unfolded.unfold_metadata

    @staticmethod
    def _refold_with_hyperdata(array, unfold_metadata):
        """Restore an unfolded array through HyperData.unfold(undo=True)."""
        if unfold_metadata is None:
            return array

        return HyperData(array).unfold(
            undo=True,
            metadata=unfold_metadata,
        ).array

    @staticmethod
    def _decomposition_preset(method_name, performance_preset):
        """
        Return practical decomposition defaults for denoising workflows.

        TensorLy's native defaults are careful general-purpose optimization
        settings. For denoising, a lower-iteration approximate factorization is
        often more useful than a slow, tightly converged decomposition.
        """
        preset_name = str(performance_preset).lower()
        presets = {
            'parafac': {
                'fast': {
                    'n_iter_max': 15,
                    'init': 'random',
                    'tol': 1e-4,
                },
                'balanced': {
                    'n_iter_max': 35,
                    'init': 'random',
                    'tol': 1e-5,
                },
                'accurate': {
                    'n_iter_max': 100,
                    'init': 'svd',
                    'tol': 1e-8,
                },
                'tensorly': {
                    'n_iter_max': 100,
                    'init': 'svd',
                    'tol': 1e-8,
                },
            },
            'parafac2': {
                'fast': {
                    'n_iter_max': 10,
                    'n_iter_parafac': 1,
                    'init': 'random',
                    'tol': 1e-4,
                    'linesearch': False,
                },
                'balanced': {
                    'n_iter_max': 25,
                    'n_iter_parafac': 2,
                    'init': 'random',
                    'tol': 1e-5,
                    'linesearch': True,
                },
                'accurate': {
                    'n_iter_max': 2000,
                    'n_iter_parafac': 5,
                    'init': 'random',
                    'tol': 1e-8,
                    'linesearch': True,
                },
                'tensorly': {
                    'n_iter_max': 2000,
                    'n_iter_parafac': 5,
                    'init': 'random',
                    'tol': 1e-8,
                    'linesearch': True,
                },
            },
        }

        if method_name not in presets:
            raise ValueError(f"No performance presets defined for {method_name!r}.")
        if preset_name not in presets[method_name]:
            valid = ', '.join(presets[method_name])
            raise ValueError(
                f"performance_preset must be one of: {valid}, or None."
            )
        return presets[method_name][preset_name]

    @staticmethod
    def _split_cp_result(result, return_errors=False):
        """Return ``(cp_tensor, errors)`` from TensorLy CP-style outputs."""
        if return_errors:
            cp_tensor, errors = result
        else:
            cp_tensor = result
            errors = None
        return cp_tensor, errors

    @staticmethod
    def _looks_like_error_sequence(values):
        """Return True when ``values`` looks like TensorLy reconstruction errors."""
        try:
            values = list(values)
        except TypeError:
            return False

        return all(np.isscalar(value) or np.asarray(value).ndim == 0 for value in values)

    @classmethod
    def _split_tucker_result(cls, result, return_errors=False):
        """Return ``(tucker_tensor, errors)`` from TensorLy Tucker-style outputs."""
        if (
            return_errors
            and isinstance(result, tuple)
            and len(result) == 2
            and cls._looks_like_error_sequence(result[1])
        ):
            return result[0], result[1]

        return result, None

    # =============================================================================
    # Spatial Filters
    # =============================================================================

    @staticmethod
    def _validate_filter_array(target_data, method, allowed_ndim):
        """Require finite, real-valued image data for spatial filters."""
        image = np.asarray(target_data)
        if image.ndim not in allowed_ndim:
            dimensions = ', '.join(str(ndim) for ndim in allowed_ndim)
            raise ValueError(f"{method} requires a {dimensions}D array.")
        if (
            image.size == 0
            or not np.issubdtype(image.dtype, np.number)
            or np.iscomplexobj(image)
        ):
            raise ValueError(f"{method} requires a nonempty, real-valued numeric array.")
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{method} requires finite input values.")
        return image

    def gaussian(self, target_data, kernel_size=3, sigma=1):
        """Smooth a 2D image with a Gaussian kernel, preserving count scale.

        Parameters
        ----------
        target_data : ndarray
            Real-valued 2D image. Integer input is promoted to float so
            interpolated intensities are not rounded back to integers.
        kernel_size : odd int or (odd y_size, odd x_size), optional
            Spatial extent of the filter; ``0`` is not accepted.
        sigma : float, optional
            Nonnegative Gaussian standard deviation in pixels. Zero asks
            OpenCV to infer it from the kernel size.

        Returns
        -------
        ndarray
            Filtered image in the same intensity units as the input.
        """
        image = self._validate_filter_array(target_data, 'gaussian', (2,))
        if (
            isinstance(kernel_size, (int, np.integer))
            and not isinstance(kernel_size, (bool, np.bool_))
        ):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2 or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0 or size % 2 == 0
            for size in kernel_size
        ):
            raise ValueError(
                "kernel_size must be an odd positive integer or a pair of them."
            )
        try:
            sigma = float(sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError("sigma must be a nonnegative finite number.") from exc
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a nonnegative finite number.")

        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        working = np.ascontiguousarray(image, dtype=dtype)
        ky, kx = kernel_size
        return cv2.GaussianBlur(working, (int(kx), int(ky)), sigmaX=sigma, sigmaY=sigma)
    
    
    # Tested successfully
    def median(self, target_data, window_size=5, mode='reflect', cval=0.0,
               origin=0, axes=None):
        """Apply an axis-aware median filter to an image or volume.
    
        The median filter replaces each pixel value with the median value of its neighborhood,
        effectively removing salt-and-pepper noise while preserving edges.
    
        Parameters
        ----------
        target_data : ndarray
            Image or volume to be denoised.
        window_size : int or tuple of int, optional
            Window width for every filtered axis, or one width per axis.
        mode : str, optional
            Boundary-extension mode passed to ``scipy.ndimage.median_filter``.
        cval : scalar, optional
            Constant fill value used when ``mode='constant'``.
        origin : int or sequence, optional
            Placement of the filter relative to each filtered pixel.
        axes : tuple of int or None, optional
            Axes over which to apply the median. ``None`` filters every axis,
            including the stack axis of unfolded 3D data. For 4D data,
            ``HyperData.denoise`` supplies the selected real- or
            reciprocal-space axes unless ``domain=None`` is requested.
    
        Returns
        -------
        ndarray
            Filtered data with the input shape and dtype.
    
        Notes
        -----
        The median filter is particularly effective for removing salt-and-pepper noise.
        It may be less effective for Gaussian noise.
        """
        return median_filter(
            target_data,
            size=window_size,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )


    # Tested successfully on (4/30/2024) for real- and reciprocal-space denoising
    def bilateral(self, target_data, d=9, sigma_color=75, sigma_space=75):
        """Smooth a 2D image while retaining intensity-dependent edges.

        Parameters
        ----------
        target_data : ndarray
            Real-valued 2D image.
        d : int, optional
            Pixel-neighborhood diameter. Zero lets OpenCV infer it from
            ``sigma_space``.
        sigma_color : float, optional
            Positive intensity-distance scale in the input's units.
        sigma_space : float, optional
            Positive spatial-distance scale in pixels.

        Returns
        -------
        ndarray
            Float image in the original intensity units. OpenCV filters in
            float32; subtracting the minimum first preserves small differences
            on top of a large baseline.
        """
        image = self._validate_filter_array(target_data, 'bilateral', (2,))
        if (
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, np.integer))
            or d < 0
        ):
            raise ValueError("d must be a nonnegative integer.")
        try:
            sigma_color = float(sigma_color)
            sigma_space = float(sigma_space)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sigma_color and sigma_space must be positive finite numbers."
            ) from exc
        if (
            not np.isfinite(sigma_color) or sigma_color <= 0
            or not np.isfinite(sigma_space) or sigma_space <= 0
        ):
            raise ValueError("sigma_color and sigma_space must be positive finite numbers.")

        working = np.asarray(image, dtype=np.float64)
        baseline = float(working.min())
        shifted = np.ascontiguousarray(working - baseline, dtype=np.float32)
        if not np.all(np.isfinite(shifted)):
            raise ValueError("bilateral input range exceeds float32 capacity.")
        filtered = cv2.bilateralFilter(shifted, int(d), sigma_color, sigma_space)
        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        return filtered.astype(dtype) + np.asarray(baseline, dtype=dtype)
    
    
    # Tested successfully on (4/30/2024) for real-space denoising
    def non_local_means(self, target_data, h=1.15, patch_size=5, patch_distance=6, fast_mode=True):
        """Apply non-local means to a 2D image or 3D volume.

        ``h`` multiplies the estimated noise standard deviation; it is not
        an absolute intensity threshold. Intensities stay in input units,
        including for integer-valued diffraction data.

        Parameters
        ----------
        target_data : ndarray
            Finite, real-valued 2D image or 3D volume.
        h : float, optional
            Nonnegative multiplier for the estimated noise level. Zero
            returns an independent floating-point copy.
        patch_size : int, optional
            Positive patch width in pixels.
        patch_distance : int, optional
            Nonnegative search distance in pixels.
        fast_mode : bool, optional
            Use scikit-image's faster algorithm (at higher memory cost).
        """
        image = self._validate_filter_array(target_data, 'non_local_means', (2, 3))
        try:
            h = float(h)
        except (TypeError, ValueError) as exc:
            raise ValueError("h must be a nonnegative finite number.") from exc
        if not np.isfinite(h) or h < 0:
            raise ValueError("h must be a nonnegative finite number.")
        for name, value, minimum in (
            ('patch_size', patch_size, 1),
            ('patch_distance', patch_distance, 0),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        working = np.asarray(image, dtype=dtype)
        if h == 0:
            return working.copy()
        sigma_est = float(np.mean(estimate_sigma(working, channel_axis=None)))
        if not np.isfinite(sigma_est) or sigma_est < 0:
            raise ValueError("non_local_means could not estimate a finite noise level.")
        if sigma_est == 0:
            return working.copy()
        return denoise_nl_means(
            working, h=h * sigma_est, patch_size=int(patch_size),
            patch_distance=int(patch_distance), fast_mode=fast_mode,
            preserve_range=True, channel_axis=None,
        )


    @staticmethod
    def _estimate_noise_sigma(image):
        """Estimate scalar Gaussian noise level for a 2D image or 3D volume."""
        try:
            sigma_est = estimate_sigma(image, channel_axis=None)
        except TypeError:
            sigma_est = estimate_sigma(image, multichannel=False)
        return float(np.mean(sigma_est))

    @staticmethod
    def _resolve_bm_stage(module, stage, enum_name):
        """Resolve user-friendly BM3D/BM4D stage names to package enums."""
        if stage is None:
            stage = 'all'
        if not isinstance(stage, str):
            return stage

        normalized = stage.strip().lower().replace('-', '_')
        stage_enum = getattr(module, enum_name, None)
        if stage_enum is None:
            return stage

        aliases = {
            'all': 'ALL_STAGES',
            'both': 'ALL_STAGES',
            'all_stages': 'ALL_STAGES',
            'full': 'ALL_STAGES',
            'hard': 'HARD_THRESHOLDING',
            'ht': 'HARD_THRESHOLDING',
            'hard_thresholding': 'HARD_THRESHOLDING',
            'wiener': 'WIENER_FILTERING',
            'wf': 'WIENER_FILTERING',
            'wiener_filtering': 'WIENER_FILTERING',
        }
        if normalized not in aliases:
            valid = ', '.join(sorted(aliases))
            raise ValueError(f"stage must be one of: {valid}, or a {enum_name} value.")
        return getattr(stage_enum, aliases[normalized])

    def bm3d(self, target_data, sigma_psd=None, stage='all', profile='np',
             clip_output=False, **kwargs):
        """
        Denoise 2D images with BM3D block matching and 3D filtering.

        BM3D normally takes a 2D image as input. The "3D" in BM3D refers to
        the internal grouping/filtering of similar 2D patches. For a 3D stack
        with shape ``(N, Y, X)``, this wrapper applies BM3D independently to
        each 2D image along the first axis.

        Parameters
        ----------
        target_data : ndarray
            Two-dimensional image or three-dimensional image stack to denoise.
        sigma_psd : float, ndarray, or sequence, optional
            Noise standard deviation or noise power spectral density passed to
            ``bm3d.bm3d``. If omitted, a scalar Gaussian noise level is
            estimated from each image. For 3D stacks, a 1D sequence with
            length ``N`` is interpreted as one scalar noise level per image.
        stage : {'all', 'hard', 'wiener'} or BM3DStages value, optional
            BM3D stage to run.
        profile : str or BM3DProfile, optional
            BM3D profile passed through to the external package.
        clip_output : bool, optional
            If True, clip each denoised image to the min/max range of the
            corresponding input image.
        **kwargs
            Additional keyword arguments passed to ``bm3d.bm3d``.

        Returns
        -------
        ndarray
            Denoised image or image stack with the same shape as the input.
        """
        try:
            import bm3d as bm3d_module
        except ImportError as exc:
            raise ImportError(
                "The BM3D denoising method requires the optional 'bm3d' "
                "package. Install it with `pip install bm3d` in the active "
                "environment, then rerun denoise(method='bm3d', ...)."
            ) from exc

        data = np.asarray(target_data)
        if data.ndim not in (2, 3):
            raise ValueError(
                "bm3d expects a 2D image or a 3D stack with shape "
                f"(N, Y, X); got shape {data.shape}."
            )
        if np.iscomplexobj(data):
            raise TypeError("bm3d requires real-valued data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("bm3d requires finite input values.")
        if not isinstance(clip_output, (bool, np.bool_)):
            raise ValueError("clip_output must be a boolean.")

        stage_arg = self._resolve_bm_stage(
            bm3d_module,
            stage,
            'BM3DStages',
        )

        def _denoise_image(image, image_sigma):
            image = np.asarray(image)
            if image_sigma is None:
                image_sigma = self._estimate_noise_sigma(image)
            result = bm3d_module.bm3d(
                image.astype(np.result_type(image.dtype, np.float32), copy=False),
                sigma_psd=image_sigma,
                profile=profile,
                stage_arg=stage_arg,
                **kwargs,
            )
            if isinstance(result, tuple):
                result = result[0]
            result = np.asarray(result)
            if result.shape != image.shape:
                raise ValueError(
                    "bm3d changed image shape from "
                    f"{image.shape} to {result.shape}; 4Denoise expects "
                    "shape-preserving denoising methods."
                )
            if clip_output:
                result = np.clip(result, np.min(image), np.max(image))
            return result

        if data.ndim == 2:
            return _denoise_image(data, sigma_psd)

        per_image_sigma = None
        shared_sigma = sigma_psd
        if sigma_psd is not None:
            sigma_array = np.asarray(sigma_psd)
            if sigma_array.ndim == 1 and sigma_array.size == data.shape[0]:
                per_image_sigma = sigma_array
                shared_sigma = None

        first_result = _denoise_image(
            data[0],
            per_image_sigma[0] if per_image_sigma is not None else shared_sigma,
        )
        denoised = np.empty(
            (data.shape[0],) + first_result.shape,
            dtype=first_result.dtype,
        )
        denoised[0] = first_result
        for idx in range(1, data.shape[0]):
            image_sigma = (
                per_image_sigma[idx]
                if per_image_sigma is not None
                else shared_sigma
            )
            denoised[idx] = _denoise_image(data[idx], image_sigma)

        return denoised

    def bm4d(self, target_data, sigma_psd=None, stage='all', profile='np',
             clip_output=False, **kwargs):
        """
        Denoise a 3D volume with BM4D block matching and 4D filtering.

        BM4D is the volumetric counterpart to BM3D and expects a single 3D
        volume. To apply BM4D to 4D-STEM data, unfold the 4D tensor to a 3D
        representation with ``HyperData.denoise(..., unfold_domain=...)``.

        Parameters
        ----------
        target_data : ndarray
            Three-dimensional volume to denoise.
        sigma_psd : float or ndarray, optional
            Noise standard deviation or noise power spectral density passed to
            ``bm4d.bm4d``. If omitted, a scalar Gaussian noise level is
            estimated from the volume.
        stage : {'all', 'hard', 'wiener'} or BM4DStages value, optional
            BM4D stage to run.
        profile : str or BM4DProfile, optional
            BM4D profile passed through to the external package.
        clip_output : bool, optional
            If True, clip the denoised volume to the min/max range of the
            input volume.
        **kwargs
            Additional keyword arguments passed to ``bm4d.bm4d``.

        Returns
        -------
        ndarray
            Denoised volume with the same shape as the input.
        """
        try:
            import bm4d as bm4d_module
        except ImportError as exc:
            raise ImportError(
                "The BM4D denoising method requires the optional 'bm4d' "
                "package. Install it with `pip install bm4d` in the active "
                "environment, then rerun denoise(method='bm4d', ...)."
            ) from exc

        data = np.asarray(target_data)
        if data.ndim != 3:
            raise ValueError(
                "bm4d expects a single 3D volume. For 4D-STEM data, use "
                "unfold_domain to convert the tensor to 3D before denoising. "
                f"Got shape {data.shape}."
            )
        if np.iscomplexobj(data):
            raise TypeError("bm4d requires real-valued data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("bm4d requires finite input values.")
        if not isinstance(clip_output, (bool, np.bool_)):
            raise ValueError("clip_output must be a boolean.")

        stage_arg = self._resolve_bm_stage(
            bm4d_module,
            stage,
            'BM4DStages',
        )
        volume_sigma = (
            self._estimate_noise_sigma(data)
            if sigma_psd is None
            else sigma_psd
        )

        result = bm4d_module.bm4d(
            data.astype(np.result_type(data.dtype, np.float32), copy=False),
            volume_sigma,
            profile=profile,
            stage_arg=stage_arg,
            **kwargs,
        )
        if isinstance(result, tuple):
            result = result[0]
        result = np.asarray(result)
        if result.shape != data.shape:
            raise ValueError(
                "bm4d changed volume shape from "
                f"{data.shape} to {result.shape}; 4Denoise expects "
                "shape-preserving denoising methods."
            )
        if clip_output:
            result = np.clip(result, np.min(data), np.max(data))

        return result


    def anisotropic_diffusion(self, target_data, niter=10, kappa=30, gamma=0.2, option=2):
        """Apply edge-preserving Perona-Malik diffusion to a real N-D array.

        Each neighboring pixel pair exchanges equal and opposite flux, so
        constant images and total intensity are preserved with no-flux edges.
        The explicit update is split into stable substeps when ``gamma`` is
        larger than ``1 / (2 * target_data.ndim)``.

        Parameters
        ----------
        target_data : ndarray
            Real-valued image or volume with at least two dimensions.
        niter : int, optional
            Number of diffusion iterations; zero returns an independent copy.
        kappa : float, optional
            Positive contrast scale for edge inhibition.
        gamma : float, optional
            Nonnegative total step size per iteration.
        option : {1, 2}, optional
            Exponential or reciprocal conduction function, respectively.

        Returns
        -------
        ndarray
            Diffused data. Integer inputs are promoted to floating point;
            values are not clipped to an arbitrary minimum.
        """
        data = np.asarray(target_data)
        if data.ndim < 2:
            raise ValueError("anisotropic_diffusion expects at least 2D data.")
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise TypeError("anisotropic_diffusion requires real numeric data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("anisotropic_diffusion requires finite input values.")
        if isinstance(niter, (bool, np.bool_)) or not isinstance(niter, Integral) or niter < 0:
            raise ValueError("niter must be a nonnegative integer.")
        if isinstance(kappa, (bool, np.bool_)) or not np.isscalar(kappa):
            raise ValueError("kappa must be positive and finite.")
        if isinstance(gamma, (bool, np.bool_)) or not np.isscalar(gamma):
            raise ValueError("gamma must be nonnegative and finite.")
        try:
            kappa = float(kappa)
            gamma = float(gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError("kappa and gamma must be finite numbers.") from exc
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be positive and finite.")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite.")
        if isinstance(option, (bool, np.bool_)) or option not in (1, 2):
            raise ValueError("option must be 1 or 2.")
        if niter == 0 or gamma == 0:
            return np.array(data, copy=True)

        result = data.astype(np.result_type(data.dtype, np.float32), copy=True)
        substeps = max(1, int(np.ceil(gamma * 2 * data.ndim)))
        step = gamma / substeps

        for _ in range(niter):
            for _ in range(substeps):
                update = np.zeros_like(result)
                for axis in range(data.ndim):
                    difference = np.diff(result, axis=axis)
                    normalized = difference / kappa
                    if option == 1:
                        conduction = np.exp(-(normalized ** 2))
                    else:
                        conduction = 1 / (1 + normalized ** 2)
                    flux = conduction * difference
                    lower = (slice(None),) * axis + (slice(None, -1),)
                    upper = (slice(None),) * axis + (slice(1, None),)
                    trailing = (slice(None),) * (data.ndim - axis - 1)
                    update[lower + trailing] += flux
                    update[upper + trailing] -= flux
                result += step * update

        return result

    
    # Tested successfully on (4/30/2024) for real-space denoising
    def total_variation(self, target_data, weight=30, eps=0.0001, max_num_iter=100):
        """Apply Chambolle total-variation denoising in native intensity units.

        A 3D unfolded stack is treated as a volume: its first axis is filtered
        together with the two image axes. Integer data are converted to float
        without scikit-image's automatic 0-to-1 intensity rescaling.

        Parameters
        ----------
        target_data : ndarray
            Finite real-valued 2D, 3D, or 4D data.
        weight : float, optional
            Nonnegative regularization strength in the input's intensity units.
            Zero returns an independent floating-point copy.
        eps : float, optional
            Positive relative convergence tolerance.
        max_num_iter : int, optional
            Positive maximum number of iterations.

        Returns
        -------
        ndarray
            Denoised floating-point array in the input's intensity units.
        """
        data = np.asarray(target_data)
        if data.ndim not in (2, 3, 4):
            raise ValueError("total_variation expects a 2D, 3D, or 4D array.")
        if data.size == 0 or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise TypeError("total_variation requires real numeric data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("total_variation requires finite input values.")
        if isinstance(weight, (bool, np.bool_)) or isinstance(eps, (bool, np.bool_)):
            raise ValueError("weight and eps must be finite numbers.")
        try:
            weight = float(weight)
            eps = float(eps)
        except (TypeError, ValueError) as exc:
            raise ValueError("weight and eps must be finite numbers.") from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be nonnegative and finite.")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")
        if (
            isinstance(max_num_iter, (bool, np.bool_))
            or not isinstance(max_num_iter, (Integral, np.integer))
            or max_num_iter < 1
        ):
            raise ValueError("max_num_iter must be a positive integer.")

        working = data.astype(np.result_type(data.dtype, np.float32), copy=False)
        if weight == 0:
            return working.copy()
        return denoise_tv_chambolle(
            working, weight=weight, eps=eps,
            max_num_iter=max_num_iter, channel_axis=None,
        )
    
    def adaptive_median_filter(self, target_data, s=3, sMax=7):
        """
        Apply an adaptive median filter to reduce noise while preserving edges.

        Parameters
        ----------
        target_data : numpy.ndarray
            The single-channel (grayscale) image to denoise.
        s : int, optional
            Odd positive initial window size for the median filter.
        sMax : int, optional
            Odd positive maximum window size, at least ``s``.

        Returns
        -------
        filtered_target_data : numpy.ndarray
            The denoised image.

        Notes
        -----
        This filter grows a reflected-boundary window until the median is
        distinguishable from its extrema. If this never happens, it returns
        the largest window's median rather than retaining an impulse.
        """
        
        image = self._validate_filter_array(target_data, 'adaptive_median_filter', (2,))
        for name, size in (('s', s), ('sMax', sMax)):
            if (
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or size < 1 or size % 2 == 0
            ):
                raise ValueError(f"{name} must be an odd positive integer.")
        if sMax < s:
            raise ValueError("sMax must be at least s.")

        padded_target_data = np.pad(image, sMax // 2, mode='reflect')
        H, W = image.shape
        filtered_target_data = np.empty_like(image)
        
        for i in range(H):
            for j in range(W):
                value = self._process_pixel(padded_target_data, i + sMax//2, j + sMax//2, s, sMax)
                filtered_target_data[i, j] = value

        return filtered_target_data

    # Private method (helper function for adaptive median filter)
    def _process_pixel(self, padded_target_data, y, x, s, sMax):
        """
        Private method (helper function for adaptive median filter): process 
        each pixel by adapting the window size and applying levels A and B checks.

        Parameters
        ----------
        padded_target_data : numpy.ndarray
            The padded input image.
        y : int
            y-coordinate in the padded image.
        x : int
            x-coordinate in the padded image.
        s : int
            Current window size.
        sMax : int
            Maximum window size.

        Returns
        -------
        value : float
            The new pixel value after filtering.

        Notes
        -----
        This method increases the window size until the conditions of Levels A 
        or B are met or the maximum window size is reached.
        """
        
        while True:
            window = padded_target_data[y-s//2:y+s//2+1, x-s//2:x+s//2+1]
            Z_min, Z_med, Z_max = np.min(window), np.median(window), np.max(window)

            if Z_min < Z_med < Z_max:
                return self._level_b(window, Z_min, Z_med, Z_max)
            
            s += 2
            if s > sMax:
                return Z_med

    # Private method (helper function for adaptive median filter)
    def _level_b(self, window, Z_min, Z_med, Z_max):
        """
        Level B processing to determine the output pixel value based on window statistics.

        Parameters
        ----------
        window : numpy.ndarray
            The current window of pixel values.
        Z_min : float
            Minimum value in the window.
        Z_med : float
            Median value in the window.
        Z_max : float
            Maximum value in the window.

        Returns
        -------
        value : float
            Either the original or the median pixel value, based on conditions.

        Notes
        -----
        This method checks if the central pixel in the window is not an extreme 
        value. If true, it returns the original central pixel value; otherwise, 
        it returns the median value.
        """
        
        Z_xy = window[window.shape[0]//2, window.shape[1]//2]
        
        if Z_min < Z_xy < Z_max:
            return Z_xy
        else:
            return Z_med
    
    # #
    # def wiener(self, other):
        
        
        
    #     return clip_values(filtered_data)    
    
    # =============================================================================
    # Unsupervised Learning
    # =============================================================================
    
    # #
    # def PCA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    # #
    # def kernelPCA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    # #
    # def fastICA(self, other):
        
        
        
    #     return clip_values(filtered_data)
    
    def nmf(self, X, n_components=None, init='random', update_H=True, solver='cd', 
            beta_loss='frobenius', tol=0.0001, max_iter=200, alpha_W=0.0, alpha_H='same', 
            l1_ratio=0.0, random_state=None, verbose=0, shuffle=False, return_decomposition=False,
            plot_eigenvalues=False, unfold_domain=None, unfold_method='row_major'):
        """
        Non-negative Matrix Factorization (NMF) using scikit-learn's `non_negative_factorization`.
        
        Computes a decomposition of matrix X into two non-negative matrices W and H such that:
        X ≈ W @ H
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix to decompose.
        
        n_components : int, optional, default=None
            Number of components to use for the factorization. If not set, all features are kept.
        
        init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, optional, default='random'
            Initialization method to use:
            - 'random': non-negative random matrices
            - 'nndsvd': Nonnegative Double Singular Value Decomposition
            - 'nndsvda': NNDSVD with zeros filled with the average of X
            - 'nndsvdar': NNDSVD with zeros filled with small random values
            - 'custom': Custom matrices W and H must be provided if update_H is True
            
        update_H : bool, default=True
            Whether to update the matrix H in the factorization. If False, H is fixed.
        
        solver : {'cd', 'mu'}, default='cd'
            The numerical solver to use:
            - 'cd': Coordinate Descent
            - 'mu': Multiplicative Update
        
        beta_loss : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}, default='frobenius'
            Beta divergence to be minimized, measuring the distance between X and the dot product WH.
        
        tol : float, default=0.0001
            Tolerance of the stopping condition.
        
        max_iter : int, default=200
            Maximum number of iterations to run the algorithm.
        
        alpha_W : float, default=0.0
            Regularization parameter for W. Set to 0 for no regularization.
        
        alpha_H : float or "same", default="same"
            Regularization parameter for H. Set to 0 for no regularization. If "same", it takes the same value as alpha_W.
        
        l1_ratio : float, default=0.0
            The regularization mixing parameter, with 0 <= l1_ratio <= 1. Determines the balance between L1 and L2 penalties.
        
        random_state : int, RandomState instance or None, default=None
            Used for NMF initialization and in the Coordinate Descent solver. Pass an int for reproducible results.
        
        verbose : int, default=0
            The verbosity level.
        
        shuffle : bool, default=False
            If True, randomize the order of coordinates in the Coordinate Descent solver.
    
        return_decomposition : bool, default=False
            If True, return the decomposition W and H.
    
        plot_scree : bool, default=False
            If True, plot a scree plot showing the variance explained by each component.
        
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            The resulting matrix of the factorization.
        
        H : ndarray of shape (n_components, n_features)
            The resulting matrix of the factorization.
        
        n_iter : int
            The number of iterations run.
    
        reconstruction : ndarray of shape (n_samples, n_features)
            The reconstructed matrix X from W and H, returned if return_reconstruction is True.
        """
        from sklearn.decomposition import non_negative_factorization
    
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            X, unfold_metadata = self._unfold_with_hyperdata(
                X, unfold_domain, unfold_method
            )
    
        # Perform NMF
        W, H, n_iter = non_negative_factorization(X, n_components=n_components, init=init,
                                                  update_H=update_H, solver=solver,
                                                  beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                                                  alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                                                  random_state=random_state, verbose=verbose, shuffle=shuffle)
        
        # Calculate the sum of squared elements in each component (row of H)
        component_significance = np.sum(H ** 2, axis=1)
        
        # Sort components by significance
        sorted_indices = np.argsort(-component_significance)
        H = H[sorted_indices]
        W = W[:, sorted_indices]
        component_significance = component_significance[sorted_indices]
        
        # Plotting option for log10 of eigenvalues vs. number of components
        if plot_eigenvalues:
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(component_significance) + 1), np.log10(component_significance), 'o-')
            plt.title('Log10 of Eigenvalues vs. Number of Components')
            plt.xlabel('Component Number')
            plt.ylabel('Log10(Eigenvalue)')
            plt.grid()
            plt.show()
        
        # Reconstruct decomposition
        reconstruction = W @ H
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:               
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            return [reconstruction, [W, H, n_iter]]

        else:
            return reconstruction

    ##########################
    # TensorLy Decomposition #
    ##########################
    
    
    # Good results with 'reciprocal' unfolding and rank=50
    def parafac(self, tensor, rank, n_iter_max=100, init='svd', svd='truncated_svd', 
                normalize_factors=False, orthogonalise=False, tol=1e-08, random_state=None, 
                verbose=0, return_errors=False, sparsity=None, l2_reg=0, mask=None, 
                cvg_criterion='abs_rec_error', fixed_modes=None, svd_mask_repeats=5, 
                linesearch=False, callback=None, implementation='tensorly', 
                return_decomposition=False, unfold_domain=None,
                unfold_method='row_major', performance_preset=None,
                working_dtype=None):
        """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
        
        Computes a rank-rank decomposition of tensor such that:
        tensor = [|weights; factors[0], ..., factors[-1] |]
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random', CPTensor}, optional, default is 'svd'
            Type of factor matrix initialization. If a CPTensor is passed, this 
            is directly used for initialization.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape 
            (rank, ), which will contain the norms of the factors.
        
        orthogonalise : bool, optional, default is False
            If True, enforce orthogonality on the factors.
        
        tol : float, optional, default is 1e-08
            Relative reconstruction error tolerance. The algorithm is considered 
            to have found the global minimum when the reconstruction error is
            less than tol.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        return_errors : bool, optional, default is False
            Activate return of iteration errors.
        
        sparsity : float or int, optional, default is None
            If sparsity is not None, we approximate tensor as a sum of low_rank_component 
            and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). 
            sparsity denotes desired fraction or number of non-zero elements in 
            the sparse_component of the tensor.
        
        l2_reg : float, optional, default is 0
            L2 regularization parameter.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where 
            the values are missing and 1 everywhere else.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', 
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol. 
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified. 
            The last mode cannot be fixed due to error computation.
        
        svd_mask_repeats : int, optional, default is 5
            If using a tensor with masked values, this initializes using SVD 
            multiple times to remove the effect of these missing values on the initialization.
        
        linesearch : bool, optional, default is False
            Whether to perform line search as proposed by Bro.
        
        callback : callable, optional
            Function to call at the end of each iteration.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).
        performance_preset : {'fast', 'balanced', 'accurate', 'tensorly'} or None
            Optional 4Denoise speed/accuracy preset. If None, use TensorLy's
            documented defaults exactly. If provided, this overrides
            ``n_iter_max``, ``init``, and ``tol``.
        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.
        
        Returns
        -------
        CPTensor(weight, factors) : tuple
            weights : 1D array of shape (rank, )
                All ones if normalize_factors is False (default).
                Weights of the (normalized) factors otherwise.
            factors : list of ndarray
                List of factors of the CP decomposition. Element i is of shape 
                (tensor.shape[i], rank).
            sparse_component : nD array of shape tensor.shape
                Returns only if sparsity is not None.
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithms 
            (if return_errors is True).
        
        Notes
        -----
        CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
        """
        if performance_preset is not None:
            preset = self._decomposition_preset('parafac', performance_preset)
            n_iter_max = preset['n_iter_max']
            init = preset['init']
            tol = preset['tol']

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # CANDECOMP/PARAFAC decomposition via ALS
            result = par(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, normalize_factors=normalize_factors,
                                                    orthogonalise=orthogonalise, tol=tol, random_state=random_state, verbose=verbose, return_errors=return_errors,
                                                    sparsity=sparsity, l2_reg=l2_reg, mask=mask, cvg_criterion=cvg_criterion, fixed_modes=fixed_modes,
                                                    svd_mask_repeats=svd_mask_repeats, linesearch=linesearch, callback=callback)
            if return_errors:
                cp_result, errors = result
            else:
                cp_result = result
                errors = None
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        if sparsity is not None:
            cp_tensor, sparse_component = cp_result
        else:
            cp_tensor = cp_result
            sparse_component = None
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            if sparsity is not None:
                results.append(sparse_component)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    # Dataset must be 3D
    def parafac2(self, tensor_slices, rank, n_iter_max=2000, init='random',
                 svd='truncated_svd', normalize_factors=False, tol=1e-08,
                 nn_modes=None, random_state=None, verbose=False,
                 return_errors=False, n_iter_parafac=5, linesearch=True,
                 implementation='tensorly', return_decomposition=False,
                 unfold_domain=None, unfold_method='row_major',
                 performance_preset=None, working_dtype=None):
        """
        Apply PARAFAC2 decomposition to a 3D input tensor.

        The TensorLy-facing parameters mirror
        ``tensorly.decomposition.parafac2``. ``performance_preset`` and
        ``working_dtype`` are 4Denoise conveniences and are optional.
        """
        if implementation != 'tensorly':
            raise ValueError(
                f"Unknown parafac2 implementation {implementation!r}; "
                "supported implementation: 'tensorly'."
            )
        if performance_preset is not None:
            preset = self._decomposition_preset('parafac2', performance_preset)
            n_iter_max = preset['n_iter_max']
            n_iter_parafac = preset['n_iter_parafac']
            init = preset['init']
            tol = preset['tol']
            linesearch = preset['linesearch']
        
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor_slices, unfold_metadata = self._unfold_with_hyperdata(
                tensor_slices, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor_slices = np.asarray(tensor_slices, dtype=working_dtype)

        if np.ndim(tensor_slices) != 3:
            raise ValueError(
                "parafac2 requires a 3D array after optional unfolding. "
                "Use unfold_domain='real' or unfold_domain='reciprocal' for "
                "4D data; unfold_domain='both' produces a 2D matrix and is not "
                "valid for parafac2."
            )

        decomposition_result = par2(
            tensor_slices,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            normalize_factors=normalize_factors,
            tol=tol,
            nn_modes=nn_modes,
            random_state=random_state,
            verbose=verbose,
            return_errors=return_errors,
            n_iter_parafac=n_iter_parafac,
            linesearch=linesearch,
        )

        if return_errors:
            decomposition, errors = decomposition_result
        else:
            decomposition = decomposition_result
            errors = None

        reconstruction = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            results = [decomposition, HyperData(reconstruction)]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction
    
    # Testing...
    def randomised_parafac(self, tensor, rank, n_samples, n_iter_max=100, init='random', 
                           svd='truncated_svd', tol=1e-08, max_stagnation=20, 
                           return_errors=False, random_state=None, verbose=0, 
                           callback=None, implementation='tensorly', 
                           return_decomposition=False, unfold_domain=None,
                           unfold_method='row_major', working_dtype=None):
        """Randomised CP decomposition via sampled ALS
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_samples : int
            Number of samples per ALS step.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'random'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        max_stagnation : int, optional, default is 20
            Maximum allowed number of iterations with no decrease in fit.
        
        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.
        
        return_errors : bool, optional, default is False
            If True, return a list of all errors.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        callback : callable, optional
            Function to call at the end of each iteration.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of shape (tensor.shape[i], rank).
        
        Notes
        -----
        Randomised CP decomposition via sampled ALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Randomised CP decomposition
            result = rand_parafac(
                tensor,
                rank=rank,
                n_samples=n_samples,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                max_stagnation=max_stagnation,
                return_errors=return_errors,
                random_state=random_state,
                verbose=verbose,
                callback=callback,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def parafac_power_iteration(self, tensor, rank, n_repeat=10, n_iteration=10, 
                                verbose=0, implementation='tensorly', 
                                return_decomposition=False, unfold_domain=None, unfold_method='row_major'):
        """CP Decomposition via Robust Tensor Power Iteration
        
        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose.
        
        rank : int
            Rank of the decomposition (number of rank-1 components).
        
        n_repeat : int, optional, default is 10
            Number of initializations to be tried.
        
        n_iteration : int, optional, default is 10
            Number of power iterations.
        
        verbose : bool, optional, default is 0
            Level of verbosity.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).
        
        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.
        
        factors : list of 2-D tl.tensor of shape (size, rank)
            Each column of each factor corresponds to one eigenvector.
        
        Notes
        -----
        CP Decomposition via Robust Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        
        # Conditional implementation
        if implementation == 'tensorly':
            # CP Decomposition via Robust Tensor Power Iteration
            weights, factors = parafac_power_iter(
                tensor,
                rank=rank,
                n_repeat=n_repeat,
                n_iteration=n_iteration,
                verbose=verbose,
            )
            
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((weights, factors))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            return weights, factors, reconstruction
        else:
            return reconstruction


    def symmetric_parafac_power_iteration(self, tensor, rank, n_repeat=10, 
                                          n_iteration=10, verbose=False, 
                                          implementation='tensorly', 
                                          return_decomposition=False, unfold_domain=None, unfold_method='row_major'):
        """Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration
        
        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose, must be symmetric of shape (size, )*order.
        
        rank : int
            Rank of the decomposition (number of rank-1 components).
        
        n_repeat : int, optional, default is 10
            Number of initializations to be tried.
        
        n_iteration : int, optional, default is 10
            Number of power iterations.
        
        verbose : bool, optional, default is False
            Level of verbosity.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
        
        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.
        
        factor : 2-D tl.tensor of shape (size, rank)
            Each column corresponds to one eigenvector.
        
        Notes
        -----
        Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration
            weights, factor = sym_parafac_power_iter(
                tensor,
                rank=rank,
                n_repeat=n_repeat,
                n_iteration=n_iteration,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # A symmetric CP decomposition uses the same factor for every mode.
        factors = [factor] * np.ndim(tensor)
        reconstruction = tl.cp_to_tensor((weights, factors))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            return weights, factor, reconstruction
        else:
            return reconstruction


    def robust_pca(self, tensor, mask=None, tol=1e-06, reg_E=1.0,
                   reg_J=1.0, mu_init=0.0001, mu_max=1e10,
                   learning_rate=1.1, n_iter_max=100, return_errors=False,
                   verbose=1, implementation='tensorly',
                   return_decomposition=False, unfold_domain=None,
                   unfold_method='row_major', working_dtype=None):
        """
        Denoise a tensor by separating low-rank signal and sparse corruption.

        This wraps :func:`tensorly.decomposition.robust_pca`. The low-rank
        component is returned as the denoised reconstruction by default. Use
        ``return_decomposition=True`` to also obtain the sparse component and,
        when requested, the per-iteration reconstruction errors.

        Parameters
        ----------
        tensor : ndarray
            Input tensor.
        mask : ndarray, optional
            Boolean array with the same shape as ``tensor``. It should be zero
            where values are missing and one elsewhere.
        tol : float, optional
            Convergence tolerance.
        reg_E : float, optional
            Regularization strength for the sparse component.
        reg_J : float, optional
            Regularization strength for the low-rank component.
        mu_init : float, optional
            Initial augmented-Lagrangian penalty.
        mu_max : float, optional
            Maximum augmented-Lagrangian penalty.
        learning_rate : float, optional
            Multiplicative increase applied to the penalty each iteration.
        n_iter_max : int, optional
            Maximum number of iterations.
        return_errors : bool, optional
            If True, return ``(low_rank, errors)`` by default, or append
            errors to the decomposition payload when requested.
        verbose : int, optional
            TensorLy verbosity level.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``[low_rank, sparse_component]`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold both components
            afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or list
            Low-rank denoised tensor, or decomposition details when
            ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != np.shape(tensor):
                raise ValueError(
                    "mask must have the same shape as the input tensor; "
                    f"got {mask.shape} and {np.shape(tensor)}."
                )

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
            if mask is not None:
                mask, _ = self._unfold_with_hyperdata(
                    mask, unfold_domain, unfold_method
                )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        result = robust_tensor_pca(
            tensor,
            mask=mask,
            tol=tol,
            reg_E=reg_E,
            reg_J=reg_J,
            mu_init=mu_init,
            mu_max=mu_max,
            learning_rate=learning_rate,
            n_iter_max=n_iter_max,
            return_errors=return_errors,
            verbose=verbose,
        )

        if return_errors:
            low_rank, sparse_component, errors = result
        else:
            low_rank, sparse_component = result
            errors = None

        if unfold_metadata is not None:
            low_rank = self._refold_with_hyperdata(
                low_rank, unfold_metadata
            )
            sparse_component = self._refold_with_hyperdata(
                sparse_component, unfold_metadata
            )

        if return_decomposition:
            results = [low_rank, sparse_component]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return low_rank, errors
        return low_rank



    def non_negative_parafac_hals(self, tensor, rank, n_iter_max=100, init='svd', 
                    svd='truncated_svd', tol=1e-07, random_state=None, 
                    sparsity_coefficients=None, fixed_modes=None, 
                    nn_modes='all', exact=False, normalize_factors=False, 
                    verbose=False, return_errors=False, cvg_criterion='abs_rec_error', 
                    implementation='tensorly', return_decomposition=False,
                    unfold_domain=None, unfold_method='row_major',
                    working_dtype=None):
        """Non-negative CP decomposition via HALS
        
        Uses Hierarchical ALS (Alternating Least Squares) which updates each factor 
        column-wise (one column at a time while keeping all other columns fixed).
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-07
            Tolerance: the algorithm stops when the variation in the reconstruction 
            error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.
        
        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients on each factor. If None, the algorithm is 
            computed without sparsity.
        
        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.
        
        nn_modes : None, 'all' or array of integers, optional, default is 'all'
            Specify which modes to impose non-negativity constraints on. If 'all', 
            then non-negativity is imposed on all modes.
        
        exact : bool, optional, default is False
            If True, the algorithm gives results with high precision but it needs 
            high computational cost. If False, the algorithm gives an approximate solution.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape 
            (rank, ), which will contain the norms of the factors.
        
        verbose : bool, optional, default is False
            Indicates whether the algorithm prints the successive reconstruction 
            errors or not.
        
        return_errors : bool, optional, default is False
            Indicates whether the algorithm should return all reconstruction 
            errors and computation time of each iteration.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', 
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol. 
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor 
                                        and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of 
            shape (tensor.shape[i], rank).
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithm 
            (if return_errors is True).
        
        Notes
        -----
        Non-negative CP decomposition via HALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition via HALS
            result = nn_parafac_hals(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                random_state=random_state,
                sparsity_coefficients=sparsity_coefficients,
                fixed_modes=fixed_modes,
                nn_modes=nn_modes,
                exact=exact,
                normalize_factors=normalize_factors,
                verbose=verbose,
                return_errors=return_errors,
                cvg_criterion=cvg_criterion,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    def non_negative_parafac(self, tensor, rank, n_iter_max=100, init='svd', 
                             svd='truncated_svd', tol=1e-06, random_state=None,
                             verbose=0, normalize_factors=False, return_errors=False, 
                             mask=None, cvg_criterion='abs_rec_error', fixed_modes=None, 
                             implementation='tensorly', return_decomposition=False,
                             unfold_domain=None, unfold_method='row_major',
                             working_dtype=None):
        """Non-negative CP decomposition using multiplicative updates
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : int
            Number of components.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-06
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional, default is 0
            Level of verbosity.
        
        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape (rank, ), which will contain the norms of the factors.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
        
        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', ALS stops at current iteration if (previous rec_error - current rec_error) < tol. If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
        
        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified. The last mode cannot be fixed due to error computation.
        
        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of shape (tensor.shape[i], rank).
        
        errors : list
            A list of reconstruction errors at each iteration of the algorithm (if return_errors is True).
        
        Notes
        -----
        Non-negative CP decomposition using multiplicative updates.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition using multiplicative updates
            result = nn_parafac(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                random_state=random_state,
                verbose=verbose,
                normalize_factors=normalize_factors,
                return_errors=return_errors,
                mask=mask,
                cvg_criterion=cvg_criterion,
                fixed_modes=fixed_modes,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def cp_constrained(self, tensor, rank, n_iter_max=100, n_iter_max_inner=10,
                       init='svd', svd='truncated_svd', tol_outer=1e-08,
                       tol_inner=1e-06, random_state=None, verbose=0,
                       return_errors=False, cvg_criterion='abs_rec_error',
                       fixed_modes=None, non_negative=None, l1_reg=None,
                       l2_reg=None, l2_square_reg=None, unimodality=None,
                       normalize=None, simplex=None, normalized_sparsity=None,
                       soft_sparsity=None, smoothness=None, monotonicity=None,
                       hard_sparsity=None, implementation='tensorly',
                       return_decomposition=False, unfold_domain=None,
                       unfold_method='row_major', working_dtype=None):
        """
        Apply constrained PARAFAC decomposition to an input tensor.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int
            Number of components.
        n_iter_max : int
            Maximum number of outer iterations.
        n_iter_max_inner : int
            Maximum number of inner ADMM iterations.
        init : {'svd', 'random', CPTensor}, optional
            TensorLy initialization method.
        svd : str, optional
            SVD function used by TensorLy.
        tol_outer : float, optional
            Relative reconstruction-error tolerance for the outer loop.
        tol_inner : float, optional
            Absolute reconstruction-error tolerance for inner ADMM updates.
        return_decomposition : bool, optional
            If True, return ``[weights, factors, reconstruction]`` and append
            errors when ``return_errors=True``.
        unfold_domain : any, optional
            Apply unfolding before decomposition.
        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        ndarray or list
            Reconstructed tensor, or decomposition details when
            ``return_decomposition=True``.

        Examples
        --------
        >>> my4Dobject = HyperData(data)
        >>> reconstructed_data = my4Dobject.denoise(method='cp_constrained',
                                                    unfold_domain='real',
                                                    rank=3, n_iter_max=50)
        """
        
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        if implementation == 'tensorly':
            result = constrained_parafac(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                n_iter_max_inner=n_iter_max_inner,
                init=init,
                svd=svd,
                tol_outer=tol_outer,
                tol_inner=tol_inner,
                random_state=random_state,
                verbose=verbose,
                return_errors=return_errors,
                cvg_criterion=cvg_criterion,
                fixed_modes=fixed_modes,
                non_negative=non_negative,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                l2_square_reg=l2_square_reg,
                unimodality=unimodality,
                normalize=normalize,
                simplex=simplex,
                normalized_sparsity=normalized_sparsity,
                soft_sparsity=soft_sparsity,
                smoothness=smoothness,
                monotonicity=monotonicity,
                hard_sparsity=hard_sparsity,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
            
        reconstruction = tl.cp_to_tensor(cp_tensor)
            
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
            
        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def tensor_ring_als(self, tensor, rank, ls_solve='lstsq',
                        n_iter_max=100, tol=1e-06, random_state=None,
                        verbose=False, callback=None, return_errors=False,
                        implementation='tensorly', return_decomposition=False,
                        unfold_domain=None, unfold_method='row_major',
                        working_dtype=None):
        """
        Denoise a tensor with Tensor-Ring alternating least squares.

        This wraps :func:`tensorly.decomposition.tensor_ring_als` and returns
        the dense Tensor-Ring reconstruction by default.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int or sequence of int
            Tensor-Ring rank. An integer applies the same rank to every core.
        ls_solve : {'lstsq', 'normal_eq'}, optional
            Least-squares solver. ``'lstsq'`` is more numerically stable;
            ``'normal_eq'`` can be faster but less accurate.
        n_iter_max : int, optional
            Maximum number of ALS iterations.
        tol : float, optional
            Stop when the relative reconstruction-error change is below this
            value.
        random_state : None, int, or numpy.random.RandomState, optional
            Random state used to initialize the Tensor-Ring cores.
        verbose : bool, optional
            If True, print TensorLy iteration information.
        callback : callable or None, optional
            TensorLy callback receiving the current ``TRTensor`` and relative
            reconstruction error after each iteration.
        return_errors : bool, optional
            Return the relative error sequence collected through TensorLy's
            iteration callback.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``(tr_decomposition, reconstruction)`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold the dense
            reconstruction afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or tuple
            Dense Tensor-Ring reconstruction, or the ``TRTensor`` and dense
            reconstruction when ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        errors = [] if return_errors else None
        if return_errors:
            def collect_error(decomposition, relative_error):
                errors.append(float(relative_error))
                if callback is not None:
                    return callback(decomposition, relative_error)
                return None
        else:
            collect_error = callback

        tr_decomposition = tr_als(
            tensor,
            rank=rank,
            ls_solve=ls_solve,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            callback=collect_error,
        )
        reconstruction = tl.tr_to_tensor(tr_decomposition)

        if unfold_metadata is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            if return_errors:
                return tr_decomposition, reconstruction, errors
            return tr_decomposition, reconstruction
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def tensor_ring_als_sampled(
            self, tensor, rank, n_samples, n_iter_max=100, tol=1e-06,
            uniform_sampling=False, randomized_error=False,
            random_state=None, verbose=False, callback=None,
            return_errors=False,
            implementation='tensorly', return_decomposition=False,
            unfold_domain=None, unfold_method='row_major',
            working_dtype=None):
        """
        Denoise a tensor with sampled Tensor-Ring alternating least squares.

        Sampling reduces the least-squares problem size and can make Tensor-Ring
        decomposition faster at the cost of approximation accuracy.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int or sequence of int
            Tensor-Ring rank. An integer applies the same rank to every core.
        n_samples : int or sequence of int
            Rows sampled while updating each core. An integer applies the same
            sample count to every mode.
        n_iter_max : int, optional
            Maximum number of sampled ALS iterations.
        tol : float, optional
            Stop when the relative reconstruction-error change is below this
            value.
        uniform_sampling : bool, optional
            Use uniform sampling instead of leverage-score sampling.
        randomized_error : bool, optional
            Estimate the residual using random sampling instead of computing
            the exact residual after each iteration.
        random_state : None, int, or numpy.random.RandomState, optional
            Random state used for initialization and sampling.
        verbose : bool, optional
            If True, print TensorLy iteration information.
        callback : callable or None, optional
            TensorLy callback receiving the current ``TRTensor`` and relative
            reconstruction error after each iteration.
        return_errors : bool, optional
            Return the relative error sequence collected through TensorLy's
            iteration callback.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``(tr_decomposition, reconstruction)`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold the dense
            reconstruction afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or tuple
            Dense Tensor-Ring reconstruction, or the ``TRTensor`` and dense
            reconstruction when ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        errors = [] if return_errors else None
        if return_errors:
            def collect_error(decomposition, relative_error):
                errors.append(float(relative_error))
                if callback is not None:
                    return callback(decomposition, relative_error)
                return None
        else:
            collect_error = callback

        tr_decomposition = tr_als_sampled(
            tensor,
            rank=rank,
            n_samples=n_samples,
            n_iter_max=n_iter_max,
            tol=tol,
            uniform_sampling=uniform_sampling,
            randomized_error=randomized_error,
            random_state=random_state,
            verbose=verbose,
            callback=collect_error,
        )
        reconstruction = tl.tr_to_tensor(tr_decomposition)

        if unfold_metadata is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            if return_errors:
                return tr_decomposition, reconstruction, errors
            return tr_decomposition, reconstruction
        if return_errors:
            return reconstruction, errors
        return reconstruction
        
    def tensor_train_matrix(self, tensor, rank, svd='truncated_svd', verbose=False,
                            implementation='tensorly', return_decomposition=False,
                            unfold_domain=None, unfold_method='row_major',
                            working_dtype=None):
        """Decompose a tensor into a matrix in tt-format
    
        Decomposes the input tensor into a matrix in Tensor Train (TT) format.
    
        Parameters
        ----------
        tensor : tensorized matrix
    
        rank : 'same', float or int tuple
            If 'same', creates a decomposition with the same number of parameters as tensor.
            If float, creates a decomposition with rank x the number of parameters of tensor.
            Otherwise, the actual rank to be used, e.g., (1, rank_2, ..., 1) of size tensor.ndim//2.
            Note that boundary conditions dictate that the first rank = last rank = 1.
    
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
    
        verbose : boolean, optional
            Level of verbosity.
    
        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.
    
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.
    
        Returns
        -------
        reconstruction :
        list containing 'reconstruction' and 'tt_matrix'
    
        Notes
        -----
        Tensor Train (TT) decomposition decomposes the input tensor into a sequence of matrices
        in TT-format by recursively applying SVD.
        """
        
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
    
        # Tensor Train matrix decomposition
        tt_matrix = tt_mat(tensor, rank=rank, svd=svd, verbose=verbose)
    
        # Reconstruct the tensorized matrix from TT-Matrix factors.
        reconstruction = tt_matrix_to_tensor(tt_matrix)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
    
        if return_decomposition:
            return tt_matrix, HyperData(reconstruction)
        else:
            return reconstruction
            

    def tensor_train(self, input_tensor, rank, svd='truncated_svd', verbose=False, 
                     return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                     implementation='tensorly', working_dtype=None):
        """TT decomposition via recursive SVD
    
        Decomposes input_tensor into a sequence of order-3 tensors (factors) – also known as Tensor-Train decomposition.
    
        Parameters
        ----------
        input_tensor : tensorly.tensor
            The input tensor to decompose.
        
        rank : {int, int list}
            Maximum allowable TT rank of the factors. If int, then this is the same for all the factors.
            If int list, then rank[k] is the rank of the kth factor.
    
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
    
        verbose : boolean, optional
            Level of verbosity.
    
        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.
    
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.
    
        Returns
        -------
        factors : TT factors
            Order-3 tensors of the TT decomposition.
    
        Notes
        -----
        Tensor-Train (TT) decomposition decomposes the input tensor into a sequence of order-3 tensors
        (factors) by recursively applying SVD.
        """
        
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            input_tensor, unfold_metadata = self._unfold_with_hyperdata(
                input_tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            input_tensor = np.asarray(input_tensor, dtype=working_dtype)
    
        # Tensor-Train decomposition
        factors = tt(input_tensor, rank=rank, svd=svd, verbose=verbose)
    
        # Reconstruct the tensor from TT factors.
        reconstruction = tt_to_tensor(factors)
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
    
        if return_decomposition:
            return factors, HyperData(reconstruction)
        else:
            return reconstruction
    
    def non_negative_tucker_hals(self, tensor, rank, n_iter_max=100, init='svd', 
                                 svd='truncated_svd', tol=1e-08, sparsity_coefficients=None, 
                                 core_sparsity_coefficient=None, fixed_modes=None, 
                                 random_state=None, verbose=False, normalize_factors=False, 
                                 return_errors=False, exact=False, algorithm='fista', 
                                 return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                                 implementation='tensorly', working_dtype=None):
        """Non-negative Tucker decomposition with HALS
        
        Uses HALS to update each factor column-wise and uses FISTA or active set algorithm to update the core.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Core size, with one rank per tensor axis. An integer applies the
            same rank to every axis.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients for each factor. If None, the algorithm is computed without sparsity.
        
        core_sparsity_coefficient : array of float, optional, default is None
            Coefficient imposing sparsity on the core when updated with FISTA.
        
        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.
        
        random_state : any, optional
            Random seed or state to initialize the random number generator.
        
        verbose : bool, optional, default is False
            Level of verbosity.
        
        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        exact : bool, optional, default is False
            If True, the HALS NNLS subroutines give results with high precision but with a higher computational cost.
            If False, the algorithm gives an approximate solution.
        
        algorithm : {'fista', 'active_set'}, optional, default is 'fista'
            Non-negative least square solution to update the core.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.
        
        Returns
        -------
        factors : ndarray list
            List of positive factors of the Tucker decomposition.
        
        errors : list
            List of reconstruction errors at each iteration of the algorithm (if return_errors is True).
        
        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """
        
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Non-negative Tucker decomposition with HALS
        result = nnth(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            sparsity_coefficients=sparsity_coefficients,
            core_sparsity_coefficient=core_sparsity_coefficient,
            fixed_modes=fixed_modes,
            random_state=random_state,
            verbose=verbose,
            normalize_factors=normalize_factors,
            return_errors=return_errors,
            exact=exact,
            algorithm=algorithm,
        )
        tucker_tensor, errors = self._split_tucker_result(result, return_errors)
        core, factors = tucker_tensor
        
        # Reconstruct the tensor from Tucker factors and core
        reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            results = [core, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction
    
    
    def non_negative_tucker(self, tensor, rank, n_iter_max=10, init='svd', tol=0.0001, 
                            random_state=None, verbose=False, return_errors=False, 
                            normalize_factors=False, return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                            implementation='tensorly', working_dtype=None):
        """Non-negative Tucker decomposition
        
        Iterative multiplicative update.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        n_iter_max : int, optional, default is 10
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional
            Level of verbosity.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.
        
        Returns
        -------
        core : ndarray
            Positive core of the Tucker decomposition, has shape ranks.
        
        factors : ndarray list
            List of factors of the Tucker decomposition, element i is of shape (tensor.shape[i], rank).
        
        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """
        
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)
        
        # Non-negative Tucker decomposition
        result = nnt(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            return_errors=return_errors,
            normalize_factors=normalize_factors,
        )
        tucker_tensor, errors = self._split_tucker_result(result, return_errors)
        core, factors = tucker_tensor
        
        # Reconstruct the tensor from Tucker factors and core
        reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))
        
        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )
        
        if return_decomposition:
            results = [core, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    def partial_tucker(self, tensor, rank, modes=None, n_iter_max=100,
                       init='svd', tol=0.0001, svd='truncated_svd', random_state=None,
                       verbose=False, mask=None, svd_mask_repeats=5,
                       return_errors=False, return_decomposition=False,
                       unfold_domain=None,
                       unfold_method='row_major', implementation='tensorly',
                       working_dtype=None):
        """Partial Tucker decomposition via Higher Order Orthogonal Iteration (HOI)
        
        Decomposes tensor into a Tucker decomposition exclusively along the provided modes.
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Rank for the selected modes. A sequence must have the same length
            as ``modes``; an integer applies to every selected mode.
        
        modes : None, int list, optional
            List of the modes on which to perform the decomposition.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, or TuckerTensor, optional, default is 'svd'
            Method to initialize the decomposition. If a TuckerTensor is provided, this is used for initialization.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.tenalg.svd.SVD_FUNS.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        verbose : int, optional
            Level of verbosity.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).
        
        svd_mask_repeats : int, optional, default is 5
            Number of repetitions for the SVD in case of masking.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        return_errors : bool, optional
            Return TensorLy's reconstruction errors for each iteration.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.
        
        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.
        
        factors : ndarray list
            One factor for each selected mode, in ``modes`` order.
        
        Notes
        -----
        Partial Tucker decomposition decomposes the input tensor into a core tensor and factor matrices
        along the specified modes.
        """
        
        if implementation == 'tensorly':
            
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                tensor, unfold_metadata = self._unfold_with_hyperdata(
                    tensor, unfold_domain, unfold_method
                )
            else:
                unfold_metadata = None

            if working_dtype is not None:
                tensor = np.asarray(tensor, dtype=working_dtype)
            
            # Partial Tucker decomposition
            result = partial_tuck(
                tensor,
                rank=rank,
                modes=modes,
                n_iter_max=n_iter_max,
                init=init,
                tol=tol,
                svd=svd,
                random_state=random_state,
                verbose=verbose,
                mask=mask,
                svd_mask_repeats=svd_mask_repeats,
            )
            tucker_tensor, errors = self._split_tucker_result(
                result,
                return_errors=True,
            )
            core, factors = tucker_tensor

            # Partial factors correspond to the selected modes, not axes 0..N.
            reconstruction_modes = (
                list(range(np.ndim(tensor))) if modes is None else list(modes)
            )
            reconstruction = multi_mode_dot(
                core, factors, modes=reconstruction_modes,
            )
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = self._refold_with_hyperdata(
                    reconstruction, unfold_metadata
                )
            
            if return_decomposition:
                if return_errors:
                    return core, factors, reconstruction, errors
                return core, factors, reconstruction
            if return_errors:
                return reconstruction, errors
            return reconstruction
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def tucker(self, tensor, rank, fixed_factors=None, n_iter_max=100, init='svd',
               return_errors=False, svd='truncated_svd', tol=0.0001, random_state=None,
               mask=None, verbose=False, return_decomposition=False,
               unfold_domain=None, unfold_method='row_major',
               implementation='tensorly', working_dtype=None):
        
        """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)
        
        Decomposes tensor into a Tucker decomposition: tensor = [| core; factors[0], ...factors[-1] |]
        
        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.
        
        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.
        
        fixed_factors : int list or None, optional, default is None
            If not None, list of modes for which to keep the factors fixed. Only valid if a Tucker tensor is provided as init.
        
        n_iter_max : int, optional, default is 100
            Maximum number of iterations.
        
        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.
        
        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.
        
        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.
        
        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.
        
        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.
        
        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).
        
        verbose : int, optional
            Level of verbosity.
        
        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        
        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).
            
        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        implementation : string, optional
            'tensorly' to use Tensorly package implementation or 
            'zhang' to use that in Zhang et al. (2020), see 'Notes' below.
        
        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.
        
        factors : ndarray list
            List of factors of the Tucker decomposition. Its i-th element is of shape (tensor.shape[i], ranks[i]).
        
        Notes
        -----
        Tucker decomposition decomposes the input tensor into a core tensor and factor matrices.
        
        The "zhang" implementation is based on that in Zhang et al. (2020):
            Zhang, C., Han, R., Zhang, A. R., & Voyles, P. M. (2020). 
            "Denoising atomic resolution 4D scanning transmission electron 
            microscopy data with tensor singular value decomposition." 
            Ultramicroscopy, 219, 113123.
        """
        
        if return_errors and fixed_factors is not None:
            raise NotImplementedError(
                "TensorLy 0.9 does not return convergence errors when "
                "tucker uses fixed_factors."
            )

        if implementation == 'tensorly':
            
            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                tensor, unfold_metadata = self._unfold_with_hyperdata(
                    tensor, unfold_domain, unfold_method
                )
            else:
                unfold_metadata = None

            if working_dtype is not None:
                tensor = np.asarray(tensor, dtype=working_dtype)
            
            # Tucker decomposition
            result = tuck(
                tensor,
                rank=rank,
                fixed_factors=fixed_factors,
                n_iter_max=n_iter_max,
                init=init,
                return_errors=return_errors,
                svd=svd,
                tol=tol,
                random_state=random_state,
                mask=mask,
                verbose=verbose,
            )
            tucker_tensor, errors = self._split_tucker_result(
                result,
                return_errors=return_errors,
            )
            core, factors = tucker_tensor
            
            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))
            
            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = self._refold_with_hyperdata(
                    reconstruction, unfold_metadata
                )
            
            if return_decomposition:
                results = [core, factors, reconstruction]
                if return_errors:
                    results.append(errors)
                return results
            if return_errors:
                return reconstruction, errors
            return reconstruction
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    
    # # =============================================================================
    # # Transform Domain Filtering
    # # =============================================================================
        
    def fourier_filter(self, target_data, mode='pass', r_inner=0, r_outer=None, sigma=10):
        """Filter a 2D image by a circular or annular Fourier-frequency mask.

        Frequency radii are measured in Fourier pixels from the DC component.
        With ``r_outer=None``, ``r_inner`` is a disk cutoff. The default
        ``r_inner=0`` uses a Gaussian low-pass with width ``sigma`` instead of
        a one-pixel disk. With ``r_outer`` specified, the pass region is the
        annulus from ``r_inner`` to ``r_outer``. ``mode='cut'`` uses the exact
        complement of the pass mask. ``sigma=0`` gives hard boundaries;
        positive ``sigma`` gives Gaussian-CDF edge transitions.

        The inverse transform restores the original Fourier axis order and
        retains signed values (or complex values for complex input).

        Parameters
        ----------
        target_data : ndarray
            Two-dimensional real or complex image.
        mode : {'pass', 'cut'}, optional
            Keep or reject the selected frequency region.
        r_inner : float, optional
            Nonnegative disk cutoff or annulus inner radius.
        r_outer : float or None, optional
            Annulus outer radius, greater than ``r_inner``.
        sigma : float, optional
            Nonnegative frequency-space edge width.

        Returns
        -------
        ndarray
            Filtered image with the same shape as the input.
        """
        data = np.asarray(target_data)
        if data.ndim != 2:
            raise ValueError("fourier_filter requires a 2D image.")
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("fourier_filter requires numeric image data.")
        if not isinstance(mode, str) or mode.lower() not in ('pass', 'cut'):
            raise ValueError("mode must be 'pass' or 'cut'.")
        mode = mode.lower()

        def finite_nonnegative(value, label):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{label} must be a nonnegative finite number.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} must be a nonnegative finite number."
                ) from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be a nonnegative finite number.")
            return value

        r_inner = finite_nonnegative(r_inner, 'r_inner')
        sigma = finite_nonnegative(sigma, 'sigma')
        if r_outer is not None:
            r_outer = finite_nonnegative(r_outer, 'r_outer')
            if r_outer <= r_inner:
                raise ValueError("r_outer must be greater than r_inner.")

        rows, cols = data.shape
        y = np.arange(rows) - rows // 2
        x = np.arange(cols) - cols // 2
        radius = np.hypot(y[:, None], x[None, :])

        if r_outer is None and r_inner == 0:
            pass_mask = (
                np.exp(-0.5 * (radius / sigma) ** 2)
                if sigma > 0 else (radius == 0).astype(float)
            )
        elif r_outer is None:
            pass_mask = (
                0.5 * erfc((radius - r_inner) / (np.sqrt(2) * sigma))
                if sigma > 0 else (radius <= r_inner).astype(float)
            )
        elif sigma == 0:
            pass_mask = ((radius >= r_inner) & (radius <= r_outer)).astype(float)
        else:
            inner_gate = (
                1.0 if r_inner == 0 else
                0.5 * erfc((r_inner - radius) / (np.sqrt(2) * sigma))
            )
            outer_gate = 0.5 * erfc((radius - r_outer) / (np.sqrt(2) * sigma))
            pass_mask = inner_gate * outer_gate

        mask = pass_mask if mode == 'pass' else 1.0 - pass_mask
        mask_dtype = (
            np.float32 if data.dtype in (np.float32, np.complex64)
            else np.float64
        )
        spectrum = fftshift(fft2(data))
        filtered = ifft2(ifftshift(spectrum * mask.astype(mask_dtype)))
        return filtered if np.iscomplexobj(data) else filtered.real
    
    # #
    # def wavelet_thresholding(self, other):
    #     """
    #     Fourier filtering.
        
    #     discussed in [reference].
    #     """
        
        
    #     return clip_values(filtered_data) 
    
    # #
    # def curvelet(self, other):
    #     """
        
    #     """
        
    #     return clip_values(filtered_data) 
            
#%%

@dataclass(frozen=True)
class _DenoisingMethodContract:
    """Input accepted by one numerical method, before HyperData routing."""

    family: str
    input_ndim: tuple[int, ...]
    input_layout: str
    constraints: str = ''


# This describes the numerical helpers as they exist today. HyperData's 4D
# slicing/unfolding choices are reported separately by _DenoiseEngine.
_DENOISING_METHOD_CONTRACTS = {
    'adaptive_median_filter': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        's and sMax must be odd positive window sizes.'),
    'anisotropic_diffusion': _DenoisingMethodContract(
        'tensor filter', (2, 3, 4), 'N-dimensional array',
        'Real-valued input; integer data are promoted to floating point.'),
    'bilateral': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        'sigma_color is in input intensity units; OpenCV filters a '
        'baseline-shifted float32 image.'),
    'bm3d': _DenoisingMethodContract(
        'image filter', (2, 3), '2D image or (N, Y, X) stack',
        'A 3D stack is filtered one 2D image at a time; requires bm3d.'),
    'bm4d': _DenoisingMethodContract(
        'volume filter', (3,), 'single 3D volume',
        'Requires bm4d; 4D data must first be unfolded to 3D.'),
    'cp_constrained': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'fourier_filter': _DenoisingMethodContract(
        'frequency filter', (2,), 'single 2D image',
        'Radii and sigma are measured in Fourier pixels.'),
    'gaussian': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        'Integer input is promoted to float; kernel_size is odd and positive.'),
    'median': _DenoisingMethodContract(
        'image filter', (2, 3, 4), 'image or tensor with selected axes',
        'axes=None filters every axis, including a 3D stack axis. '
        'HyperData selects real or reciprocal axis pairs for 4D domain routing.'),
    'nmf': _DenoisingMethodContract(
        'matrix factorization', (2,), 'nonnegative 2D matrix',
        'Negative input values are not accepted by NMF.'),
    'non_local_means': _DenoisingMethodContract(
        'image filter', (2, 3), '2D image or 3D volume',
        'h multiplies estimated noise; intensity range is preserved.'),
    'non_negative_parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_parafac_hals': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_tucker_hals': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'parafac2': _DenoisingMethodContract(
        'tensor decomposition', (3,), '3D stack of matrix slices',
        '4D data must first be unfolded to 3D.'),
    'parafac_power_iteration': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'partial_tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'randomised_parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'robust_pca': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor',
        'The default reconstruction is the low-rank component.'),
    'symmetric_parafac_power_iteration': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'symmetric N-dimensional tensor',
        'All axis lengths must match and the tensor should be symmetric.'),
    'tensor_ring_als': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_ring_als_sampled': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_train': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_train_matrix': _DenoisingMethodContract(
        'tensor decomposition', (2, 4), 'even-order tensorized matrix',
        'The input order must be even; odd-order 3D unfolding is invalid.'),
    'total_variation': _DenoisingMethodContract(
        'tensor filter', (2, 3, 4), 'N-dimensional image/volume',
        'Integer data are promoted to float without rescaling intensities.'),
    'tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
}


class _DenoiseEngine:
    """
    Private dispatcher for applying denoising methods to 2D, 3D, and 4D data.

    ``HyperData.denoise(...)`` is the public API. This helper owns method
    lookup, argument validation, and dimensional routing, including the
    whole-array axis-aware median path, while ``_DenoisingMethods`` stores the
    numerical algorithms.

    Attributes
    ----------
    target_data : ndarray
        The data to be denoised, which can be 2D, 3D, or 4D.

    methods : _DenoisingMethods
        The private numerical-method collection.
        
    Methods
    -------
    denoise(method_name, target_data=None, **kwargs)
        Applies the specified denoising method to the target data using the provided parameters.
    apply(method, domain='reciprocal', **kwargs)
        Applies a method directly to 2D/3D data, over a selected coordinate
        domain for 4D data, or directly to the whole array when ``domain=None``.

    Notes
    -----
    - The `denoise` method fetches the appropriate method from the _DenoisingMethods instance and applies it to 
      the data. It raises a ValueError if the specified method is not found.
    - The `apply` method supports direct 2D/3D denoising and 4D routing through
      real or reciprocal coordinates. Axis-aware median filtering uses one
      whole-array call; other image methods retain slice-wise routing.
    """
    
    def __init__(self, target_data):
        self.array = np.asarray(target_data)
        self.ndim = self.array.ndim

        self.methods = _DenoisingMethods()

        self.available_methods = sorted(
            name for name, _ in inspect.getmembers(
                self.methods, predicate=inspect.ismethod,
            ) if not name.startswith('_')
        )

    @staticmethod
    def _parameter_display(param):
        """Return a compact display string for a method parameter."""
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return f"*{param.name}"
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return f"**{param.name}"
        if param.default is inspect.Parameter.empty:
            return f"{param.name}=..."
        return f"{param.name}={repr(param.default)}"

    @staticmethod
    def _parameter_info(param):
        """Return structured information for one user-facing parameter."""
        default = None
        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            default = param.default

        annotation = None
        if param.annotation is not inspect.Parameter.empty:
            annotation = (
                param.annotation.__name__
                if hasattr(param.annotation, '__name__')
                else str(param.annotation)
            )

        return {
            'name': param.name,
            'kind': param.kind.description,
            'required': (
                not has_default
                and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ),
            'default': default,
            'has_default': has_default,
            'annotation': annotation,
        }

    @staticmethod
    def _doc_summary(docstring):
        """Return the first paragraph of a docstring."""
        if not docstring:
            return ''
        paragraphs = docstring.strip().split('\n\n')
        return ' '.join(paragraph.strip() for paragraph in paragraphs[0].splitlines())

    @staticmethod
    def _four_dimensional_routes(method_name, contract):
        """Describe existing 4D routes without changing the dispatcher."""
        if method_name == 'median':
            return (
                "domain='real' or 'reciprocal': filter the selected axis pair",
                "unfold_domain='both': filter the resulting 2D matrix",
            )

        routes = []
        if 2 in contract.input_ndim:
            routes.append(
                "domain='real' or 'reciprocal': filter each 2D image independently"
            )
            routes.append("unfold_domain='both': denoise a 2D matrix")
        if 3 in contract.input_ndim:
            routes.append(
                "unfold_domain='real' or 'reciprocal': denoise the 3D result"
            )
        if 4 in contract.input_ndim:
            routes.append("domain=None: denoise the full 4D tensor")
        return tuple(routes)

    def _contract_info(self, method_name, method):
        """Combine a stable input contract with current routing and flags."""
        contract = _DENOISING_METHOD_CONTRACTS.get(method_name)
        if contract is None:
            raise RuntimeError(
                f"Denoising method {method_name!r} has no registered input contract."
            )

        signature = inspect.signature(method)
        result_flags = tuple(
            name for name in ('return_decomposition', 'return_errors')
            if name in signature.parameters
        )
        return {
            'family': contract.family,
            'supported_input_ndim': contract.input_ndim,
            'input_layout': contract.input_layout,
            'constraints': contract.constraints,
            'four_dimensional_routes': self._four_dimensional_routes(
                method_name, contract
            ),
            'default_output': (
                'shape-preserving HyperData reconstruction; ndarray with '
                'return_array=True'
            ),
            'optional_result_flags': result_flags,
            'output_caveat': (
                'return_errors=True yields (reconstruction, errors); '
                'return_decomposition=True retains a method-specific payload.'
                if 'return_errors' in result_flags
                else (
                    'return_decomposition=True retains a method-specific payload.'
                    if result_flags else ''
                )
            ),
        }

    def method_info(self, method_name=None, include_doc=True, print_info=False):
        """
        Return signature, input contract, and routing information.

        The first numerical-method argument is the data array supplied by
        ``HyperData.denoise``. It is intentionally excluded from
        ``method_parameters`` because users should not pass it manually.
        The contracts describe the underlying numerical methods; they do not
        alter routing or promise that all algorithms have passed numerical
        validation. Optional result flags retain their existing return types.
        """
        if method_name is None:
            info = {
                'available_methods': tuple(self.available_methods),
                'method_contracts': {
                    name: self._contract_info(name, getattr(self.methods, name))
                    for name in self.available_methods
                },
            }
            if print_info:
                print("Available denoising methods:")
                print(', '.join(self.available_methods))
            return info

        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method_name must be a non-empty string or None.")

        method = getattr(self.methods, method_name, None)
        if method_name not in self.available_methods or method is None:
            raise ValueError(
                f"No such method '{method_name}'. Available methods are: "
                f"{', '.join(self.available_methods)}"
            )

        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())

        injected_data_parameter = None
        method_parameters = parameters
        if parameters and parameters[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            injected_data_parameter = parameters[0].name
            method_parameters = parameters[1:]

        parameter_info = [
            self._parameter_info(param)
            for param in method_parameters
        ]
        required = [
            param['name']
            for param in parameter_info
            if param['required']
        ]
        optional = [
            param
            for param in parameter_info
            if not param['required']
        ]
        method_kwargs_display = ', '.join(
            self._parameter_display(param)
            for param in method_parameters
        )
        example_call = f"my_dataset.denoise(method='{method_name}'"
        if method_kwargs_display:
            example_call += f", {method_kwargs_display}"
        example_call += ")"

        docstring = inspect.getdoc(method) or ''
        info = {
            'method': method_name,
            'full_signature': f"{method_name}{signature}",
            'method_kwargs_signature': f"{method_name}({method_kwargs_display})",
            'example_call': example_call,
            'injected_data_parameter': injected_data_parameter,
            'required_parameters': required,
            'optional_parameters': optional,
            'parameters': parameter_info,
            'doc_summary': self._doc_summary(docstring),
        }
        info.update(self._contract_info(method_name, method))
        if include_doc:
            info['docstring'] = docstring

        if print_info:
            print(f"Method: {method_name}")
            if info['doc_summary']:
                print(info['doc_summary'])
            print()
            print("Use:")
            print(f"  {example_call}")
            print()
            print(
                "Data input supplied automatically by HyperData.denoise: "
                f"{injected_data_parameter}"
            )
            print(
                f"Input: {info['input_layout']} "
                f"(ndim {', '.join(map(str, info['supported_input_ndim']))})"
            )
            print(f"Default output: {info['default_output']}")
            if info['constraints']:
                print(f"Constraints: {info['constraints']}")
            print("For 4D data:")
            for route in info['four_dimensional_routes']:
                print(f"  - {route}")
            if info['optional_result_flags']:
                print(
                    "Optional result flags: "
                    + ', '.join(info['optional_result_flags'])
                )
                print(f"  {info['output_caveat']}")
            if required:
                print("Required method arguments:")
                for name in required:
                    print(f"  - {name}")
            else:
                print("Required method arguments: none")

            if optional:
                print("Optional method arguments:")
                for param in optional:
                    if param['kind'] == 'variadic keyword':
                        print(f"  - **{param['name']}")
                    elif param['kind'] == 'variadic positional':
                        print(f"  - *{param['name']}")
                    elif param['has_default']:
                        print(f"  - {param['name']}={repr(param['default'])}")
                    else:
                        print(f"  - {param['name']}")

        return info

    def _resolve_method(self, method_name):
        """Return only public numerical methods, never private helpers."""
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method must be a non-empty string.")
        if method_name not in self.available_methods:
            raise ValueError(
                f"No such method '{method_name}'. Available methods are: "
                f"{', '.join(self.available_methods)}"
            )
        return getattr(self.methods, method_name)

    def _validate_call(self, method_name, method, target_data, kwargs):
        """Check the method contract and arguments before numerical work."""
        contract = _DENOISING_METHOD_CONTRACTS[method_name]
        if target_data.ndim not in contract.input_ndim:
            supported = ', '.join(f'{ndim}D' for ndim in contract.input_ndim)
            raise ValueError(
                f"Method '{method_name}' expects {contract.input_layout} "
                f"({supported}); got shape {target_data.shape}. "
                "See denoising_method_info for supported 4D routing."
            )
        if (
            method_name == 'symmetric_parafac_power_iteration'
            and len(set(target_data.shape)) != 1
        ):
            raise ValueError(
                "symmetric_parafac_power_iteration requires equal axis "
                f"lengths; got shape {target_data.shape}."
            )
        try:
            inspect.signature(method).bind(target_data, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid arguments for denoising method '{method_name}': {exc}"
            ) from exc

    @staticmethod
    def require_reconstruction(result, expected_shape, *, method, context='denoising'):
        """Unwrap and validate a reconstruction before it enters a dataset."""
        if isinstance(result, HyperData):
            result = result.array
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Method '{method}' returned {type(result).__name__} during "
                f"{context}; expected an ndarray or HyperData reconstruction. "
                "Optional decomposition results cannot be routed as images."
            )
        expected_shape = tuple(expected_shape)
        if result.shape != expected_shape:
            raise ValueError(
                f"Method '{method}' changed shape during {context}: "
                f"expected {expected_shape}, got {result.shape}."
            )
        return result

    @staticmethod
    def split_reconstruction_errors(result, method):
        """Separate a reconstruction from its convergence history."""
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"Method '{method}' with return_errors=True must return "
                "(reconstruction, errors)."
            )
        reconstruction, errors = result
        if errors is None:
            raise ValueError(
                f"Method '{method}' did not provide convergence errors."
            )
        return reconstruction, errors

    def denoise(self, method_name, target_data=None, **kwargs):
        """Invoke one method on a whole 2D, 3D, or 4D array."""
        method = self._resolve_method(method_name)
        target_data = self.array if target_data is None else np.asarray(target_data)
        self._validate_call(method_name, method, target_data, kwargs)
        return method(target_data, **kwargs)

    def _apply_to_slices(self, method_name, domain, kwargs):
        """Apply a 2D method to 4D slices without truncating its output dtype."""
        if kwargs.get('return_decomposition', False):
            raise ValueError(
                "Slice-wise 4D denoising requires one image per slice; use "
                "return_decomposition=False, or domain=None for a whole-tensor "
                "decomposition."
            )
        if kwargs.get('return_errors', False):
            raise ValueError(
                "Slice-wise 4D denoising has no single convergence history; "
                "use unfold_domain or domain=None with return_errors=True."
            )

        method = self._resolve_method(method_name)
        if domain == 'real':
            index_shape = self.array.shape[2:4]
            expected_shape = self.array.shape[:2]
            description = 'Filtering real-space images'

            def get_slice(index):
                return self.array[:, :, index[0], index[1]]

            def put_slice(output, index, values):
                output[:, :, index[0], index[1]] = values
        else:
            index_shape = self.array.shape[:2]
            expected_shape = self.array.shape[2:4]
            description = 'Filtering diffraction patterns'

            def get_slice(index):
                return self.array[index[0], index[1], :, :]

            def put_slice(output, index, values):
                output[index[0], index[1], :, :] = values

        if 0 in index_shape or 0 in expected_shape:
            raise ValueError("Cannot denoise 4D data with an empty axis.")
        first_index = (0, 0)
        first_slice = get_slice(first_index)
        self._validate_call(method_name, method, first_slice, kwargs)
        first = self.require_reconstruction(
            method(first_slice, **kwargs), expected_shape,
            method=method_name, context='slice-wise 4D denoising',
        )
        output = np.empty(self.array.shape, dtype=first.dtype)
        put_slice(output, first_index, first)

        indices = np.ndindex(index_shape)
        next(indices)
        for index in tqdm(indices, total=prod(index_shape) - 1, desc=description):
            values = self.require_reconstruction(
                method(get_slice(index), **kwargs), expected_shape,
                method=method_name, context='slice-wise 4D denoising',
            )
            if not np.can_cast(values.dtype, output.dtype, casting='safe'):
                raise TypeError(
                    f"Method '{method_name}' returned inconsistent slice dtypes "
                    f"({output.dtype} then {values.dtype}); refusing a lossy cast."
                )
            put_slice(output, index, values)
        return output

    def apply(self, method, domain='reciprocal', **kwargs):
        """Route 4D domains; apply 2D/3D arrays directly as before."""
        self._resolve_method(method)
        if domain is None:
            return self.denoise(method, **kwargs)

        if not isinstance(domain, str):
            raise ValueError("domain must be a string or None.")
        domain = domain.lower()
        if domain in ('real_space', 'r'):
            domain = 'real'
        elif domain in ('reciprocal_space', 'k', 'k_space'):
            domain = 'reciprocal'
        if domain not in ('real', 'reciprocal'):
            raise ValueError("domain must be 'real', 'reciprocal', or None.")

        if self.ndim in (2, 3):
            return self.denoise(method, **kwargs)
        if self.ndim != 4:
            raise ValueError("Unsupported image dimensionality; expected 2D, 3D, or 4D.")

        if method == 'median':
            if 'axes' in kwargs:
                raise TypeError(
                    "For 4D median denoising, axes are selected by domain; "
                    "do not pass axes explicitly."
                )
            axes = (0, 1) if domain == 'real' else (2, 3)
            return self.denoise(method, axes=axes, **kwargs)

        return self._apply_to_slices(method, domain, kwargs)
