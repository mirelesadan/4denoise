"""Pure coordinate and calibration helpers shared by 4Denoise objects."""

import numpy as np


def _normalize_real_spacing(spacing):
    """Validate signed ``(y, x)`` units per pixel for oriented scan axes."""
    if isinstance(spacing, (bool, np.bool_)):
        raise ValueError("Real-space pixel spacing must be nonzero and finite.")
    try:
        values = np.asarray(spacing, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Real-space pixel spacing must be numeric.") from exc
    if values.ndim == 0:
        if not np.isfinite(values) or values == 0:
            raise ValueError("Real-space pixel spacing must be nonzero and finite.")
        return float(values)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values == 0):
        raise ValueError("Real-space pixel spacing must be a nonzero finite (y, x) pair.")
    return tuple(float(value) for value in values)


def _real_spacing_pair(spacing):
    """Return ``(y, x)`` spacing without changing the public scalar form."""
    if spacing is None:
        return (1.0, 1.0)
    return (spacing, spacing) if np.isscalar(spacing) else tuple(spacing)


def _scaled_real_spacing(spacing, factors):
    """Scale each pixel axis, retaining scalar form for isotropic spacing."""
    result = tuple(
        float(value) * float(factor)
        for value, factor in zip(_real_spacing_pair(spacing), factors)
    )
    return result[0] if np.isclose(result[0], result[1]) else result


def _normalize_real_origin(origin):
    """Validate the physical coordinate of the first real-space pixel center."""
    try:
        values = np.asarray(origin, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("real_origin must be a finite (y, x) pair.") from exc
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("real_origin must be a finite (y, x) pair.")
    return tuple(float(value) for value in values)


def _parse_real_selection(value, max_len, name, mode, conv_factor,
                          origin=0.0):
    """Parse real-space scalar/range selections into half-open pixel slices."""
    if value is None:
        return 0, max_len, 'all'

    is_pair = (
        isinstance(value, (tuple, list, np.ndarray))
        and np.asarray(value).shape == (2,)
    )
    if is_pair:
        a, b = np.asarray(value, dtype=float)
        if not np.all(np.isfinite((a, b))):
            raise ValueError(f"{name} range must contain finite values.")
        if mode == 'calibrated':
            a = (a - origin) / conv_factor
            b = (b - origin) / conv_factor
            if conv_factor > 0:
                start = int(np.floor(a))
                stop = int(np.ceil(b))
            else:
                # Physical [low, high) runs opposite to pixel index order.
                start = int(np.floor(b)) + 1
                stop = int(np.floor(a)) + 1
        else:
            if not float(a).is_integer() or not float(b).is_integer():
                raise ValueError(
                    f"{name} range must contain integer pixel indices when "
                    "selection_units='pixels'."
                )
            start = int(a)
            stop = int(b)
        if not (0 <= start < stop <= max_len):
            raise ValueError(
                f"Invalid {name} range ({value[0]}, {value[1]}) for "
                f"length {max_len} using {mode} units."
            )
        return start, stop, 'range'

    if np.isscalar(value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} index must be finite.")
        if mode == 'calibrated':
            index = int(np.rint((value - origin) / conv_factor))
        else:
            if not value.is_integer():
                raise ValueError(
                    f"{name} index must be an integer when "
                    "selection_units='pixels'."
                )
            index = int(value)
        if not (0 <= index < max_len):
            raise ValueError(
                f"{name} index {value:g} out of bounds for length "
                f"{max_len} using {mode} units."
            )
        return index, index + 1, 'index'

    raise ValueError(f"{name} must be a scalar, a length-2 range, or None.")


def _normalize_unit_mode(unit_mode, label='axis_units'):
    """Normalize a unit-selection mode used by plotting/selection helpers."""
    if unit_mode is None:
        unit_mode = 'auto'
    if not isinstance(unit_mode, str):
        raise ValueError(f"{label} must be 'auto', 'pixels', or 'calibrated'.")

    normalized = unit_mode.strip().lower().replace('_', '-')
    if normalized in {'auto', 'default'}:
        return 'auto'
    if normalized in {'pixel', 'pixels', 'px'}:
        return 'pixels'
    if normalized in {'calibrated', 'calibration', 'physical', 'data', 'units'}:
        return 'calibrated'
    raise ValueError(f"{label} must be 'auto', 'pixels', or 'calibrated'.")


def _resolve_unit_mode(unit_mode, units, conv_factor, label='axis_units'):
    """Resolve ``auto``/pixel/calibrated mode to units and a pixel step."""
    mode = _normalize_unit_mode(unit_mode, label=label)
    if mode == 'pixels':
        return None, None, 'pixels'

    has_units = units is not None
    has_factor = conv_factor is not None
    if has_units != has_factor:
        raise ValueError(
            f"{label} cannot use a partial calibration. Define both units and "
            "conv_factor, or clear both."
        )

    if not has_units:
        if mode == 'calibrated':
            raise ValueError(
                f"{label}='calibrated' requires stored units and conv_factor."
            )
        return None, None, 'pixels'

    factor = _normalize_real_spacing(conv_factor)
    return str(units).strip(), factor, 'calibrated'


def _center_to_calibrated(center_px, shape, conv_factor):
    """Convert pixel ``(ky, kx)`` to reciprocal coords, positive ky upward."""
    center_y, center_x = tuple(float(v) for v in center_px)
    origin_y = (int(shape[0]) - 1) / 2.0
    origin_x = (int(shape[1]) - 1) / 2.0
    return (
        (origin_y - center_y) * float(conv_factor),
        (center_x - origin_x) * float(conv_factor),
    )


def _calibrated_center_to_pixels(center, conv_factor, shape):
    """Convert calibrated reciprocal ``(ky, kx)`` to pixel coordinates."""
    center = np.asarray(center, dtype=float)
    origin_y = (int(shape[0]) - 1) / 2.0
    origin_x = (int(shape[1]) - 1) / 2.0
    return (
        origin_y - center[0] / float(conv_factor),
        origin_x + center[1] / float(conv_factor),
    )
