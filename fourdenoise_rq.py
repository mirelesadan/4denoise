"""Real/detector orientation calibration and its interactive comparison viewer."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from fourdenoise_geometry import (
    _calibrated_center_to_pixels, _real_spacing_pair, _resolve_unit_mode,
)


def _finite_number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite.")
    return result


def _pair(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite (y, x) pair.") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite (y, x) pair.")
    return tuple(float(item) for item in result)


def _rotation_matrix(angle):
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    return np.array(((c, -s), (s, c)))


def _mirror_matrix(axis):
    if axis not in (None, 'x', 'y'):
        raise ValueError("mirror_axis must be None, 'x', or 'y'.")
    return np.diag((-1.0 if axis == 'y' else 1.0,
                    -1.0 if axis == 'x' else 1.0))


@dataclass(frozen=True)
class RQCalibration:
    """Orientation of scan directions in the detector frame.

    Directions use Cartesian ``(x, y)`` components, with x right and y up
    in each image. ``matrix`` maps real to reciprocal components by first
    reflecting about ``mirror_axis`` (if any), then rotating counterclockwise
    by ``rotation_deg``. Magnitudes keep their original units: this is an
    orientation change, not a conversion of lengths to inverse lengths.

    Arrays retain their usual ``(y, x)`` index order. Image registration scale,
    translation and pivot are deliberately separate from this calibration.
    Instances are immutable and their matrix properties return fresh arrays.
    """

    rotation_deg: float = 0.0
    mirror_axis: str | None = None
    source: str = 'manual'

    def __post_init__(self):
        angle = _finite_number(self.rotation_deg, 'rotation_deg') % 360.0
        _mirror_matrix(self.mirror_axis)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty string.")
        object.__setattr__(self, 'rotation_deg', angle)

    @property
    def matrix(self):
        """Orthogonal 2x2 matrix mapping real to reciprocal directions."""
        return _rotation_matrix(self.rotation_deg) @ _mirror_matrix(self.mirror_axis)

    @property
    def inverse_matrix(self):
        """Orthogonal 2x2 matrix mapping reciprocal to real directions."""
        return self.matrix.T

    def transform_vectors(self, vectors, direction='real_to_reciprocal'):
        """Transform Cartesian ``(..., 2)`` vectors without changing their units.

        Use centered Cartesian components, not raw ``(row, column)`` peak
        indices. ``direction`` may also be ``'reciprocal_to_real'``.
        """
        values = np.asarray(vectors, dtype=float)
        if values.ndim == 0 or values.shape[-1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError("vectors must have finite Cartesian (..., 2) components.")
        if direction == 'real_to_reciprocal':
            matrix = self.matrix
        elif direction == 'reciprocal_to_real':
            matrix = self.inverse_matrix
        else:
            raise ValueError("direction must be 'real_to_reciprocal' or 'reciprocal_to_real'.")
        return values @ matrix.T

    def to_dict(self):
        """Return versioned, HDF5-serializable orientation metadata."""
        return dict(version=1, convention='real_to_reciprocal_cartesian_xy',
                    rotation_deg=self.rotation_deg, mirror_axis=self.mirror_axis,
                    source=self.source)

    @classmethod
    def from_matrix(cls, matrix, *, mirror_axis='x', source='manual'):
        """Describe an orthogonal real-to-reciprocal matrix, including reflection."""
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.shape != (2, 2) or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix.T @ matrix, np.eye(2), atol=1e-8, rtol=0)):
            raise ValueError("RQ calibration matrix must be a finite orthogonal 2x2 matrix.")
        axis = mirror_axis if np.linalg.det(matrix) < 0 else None
        if axis is None and np.linalg.det(matrix) < 0:
            axis = 'x'
        rotation = matrix @ _mirror_matrix(axis)
        return cls(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])), axis, source)


def _coerce_rq_calibration(value):
    if value is None or isinstance(value, RQCalibration):
        return value
    if not isinstance(value, dict):
        raise TypeError("rq_calibration must be RQCalibration, its saved dictionary, or None.")
    if (value.get('version') != 1
            or value.get('convention') != 'real_to_reciprocal_cartesian_xy'):
        raise ValueError("Unsupported RQ calibration version or coordinate convention.")
    return RQCalibration(value['rotation_deg'], value.get('mirror_axis'),
                         value.get('source', 'manual'))


def _reframe_rq_calibration(value, *, real=None, reciprocal=None):
    calibration = _coerce_rq_calibration(value)
    if calibration is None:
        return None
    real = np.eye(2) if real is None else np.asarray(real)
    reciprocal = np.eye(2) if reciprocal is None else np.asarray(reciprocal)
    return RQCalibration.from_matrix(
        reciprocal @ calibration.matrix @ real.T, source='coordinate_transform',
    )


def _rq_center(shape, metadata=None, conv_factor=None):
    """Resolve a rotation pivot without rounding an even-sized image center."""
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("center_beam_metadata must be a dictionary or None.")
        if 'shape' in metadata:
            try:
                stored_shape = tuple(metadata['shape'])
            except TypeError as exc:
                raise ValueError("center_beam_metadata has an invalid shape.") from exc
            if stored_shape != tuple(shape):
                raise ValueError("center_beam_metadata has a stale pattern shape; update it first.")
        for key in ('center_px', 'mean_fit_center_px'):
            if metadata.get(key) is not None:
                return _pair(metadata[key], key), key
        if metadata.get('center_calibrated') is not None:
            factor = metadata.get('conv_factor', conv_factor)
            factor = _finite_number(factor, 'center calibration factor', positive=True)
            center = _pair(metadata['center_calibrated'], 'center_calibrated')
            return _calibrated_center_to_pixels(center, factor, shape), 'center_calibrated'
    return tuple((size - 1) / 2.0 for size in shape), 'image midpoint'


def _prepare_image(image, options, name, *, reciprocal=False):
    """Take only a 2D snapshot so the viewer cannot retain a full scan view."""
    from matplotlib import colormaps

    values = np.asarray(getattr(image, 'array', image))
    if (values.ndim != 2 or min(values.shape) < 2 or np.iscomplexobj(values)
            or not (np.issubdtype(values.dtype, np.number) or values.dtype == bool)):
        raise ValueError(f"{name} must be a real numeric 2D image at least 2x2.")
    allowed = {'cmap', 'vmin', 'vmax', 'power', 'logScale', 'percentiles',
               'symmetric', 'title', 'interpolation'}
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"Unsupported {name} show options: {', '.join(sorted(unknown))}.")
    data = np.array(values, dtype=np.result_type(values.dtype, np.float32), copy=True)
    power = _finite_number(options.get('power', 1), 'power')
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        data = power * np.log(data) if options.get('logScale', reciprocal) else data ** power
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError(f"{name} has no finite display values; check logScale and power.")
    low, high = float(finite.min()), float(finite.max())
    percentiles = options.get('percentiles')
    if percentiles is not None:
        percentiles = _pair(percentiles, 'percentiles')
        if not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
        low, high = np.percentile(finite, percentiles)
    low = low if options.get('vmin') is None else _finite_number(options['vmin'], 'vmin')
    high = high if options.get('vmax') is None else _finite_number(options['vmax'], 'vmax')
    if options.get('symmetric', False):
        high = max(abs(low), abs(high))
        low = -high
    if low > high:
        raise ValueError("vmin must not exceed vmax.")
    style = dict(cmap=colormaps.get_cmap(options.get('cmap', 'gray')), vmin=low, vmax=high,
                 interpolation=options.get('interpolation', 'bilinear'))
    return np.ma.masked_invalid(data), style


class RQComparison:
    """Interactive result returned by :meth:`fourdenoise.HyperData.compare_rq`.

    The real image is fixed. The diffraction image is mirrored, rotated, and
    scaled about ``center`` (a subpixel ``(y, x)`` pivot), then placed with
    that pivot at the real-image midpoint plus ``translation``. Translation
    uses real-image ``(dy, dx)`` pixels; positive dy is downward. ``scale`` is
    real horizontal pixels per diffraction pixel. Known unequal scan pixel
    spacings are respected independently of the displayed tick units.

    ``rotation_deg`` is the counterclockwise diffraction correction, Q to R.
    ``calibration`` reports its inverse, R to Q, including any reflection.
    Editing controls affects only the preview. Call ``apply`` or press Apply
    calibration to store the orientation in the owning HyperData object.
    """

    def __init__(self, real_image, reciprocal_image, *, calibration=None,
                 on_apply=None, layout='overlay', real_alpha=0.7,
                 reciprocal_alpha=0.7, interactive=True, scale=None,
                 translation=(0.0, 0.0), real_show_kwargs=None,
                 reciprocal_show_kwargs=None, figsize=(12, 8), show=True,
                 export_path='rq_alignment.png'):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        from matplotlib.transforms import Affine2D

        if layout not in ('overlay', 'side_by_side'):
            raise ValueError("layout must be 'overlay' or 'side_by_side'.")
        if getattr(reciprocal_image, 'is_polar', False):
            raise ValueError("compare_rq requires a Cartesian diffraction image; use to_cartesian first.")
        real_options = dict(real_show_kwargs or {})
        q_options = dict(reciprocal_show_kwargs or {})
        units, factor, _ = _resolve_unit_mode(
            real_options.pop('axis_units', 'auto'), getattr(real_image, 'units', None),
            getattr(real_image, 'conv_factor', None),
        )
        self._tick_spacing = _real_spacing_pair(factor)
        self._tick_origin = getattr(real_image, 'origin', (0., 0.)) if factor is not None else (0., 0.)
        physical_steps = _real_spacing_pair(getattr(real_image, 'conv_factor', None))
        self._aspect = abs(physical_steps[0] / physical_steps[1])
        axes_visible = real_options.pop('axes', True)
        grid = real_options.pop('grid', False)
        scale_bar = real_options.pop('scale_bar', None)
        bar_color = real_options.pop('scale_bar_color', 'white')
        bar_label = real_options.pop('scale_bar_label', True)
        if scale_bar is not None:
            scale_bar = _finite_number(scale_bar, 'scale_bar', positive=True)
        self._real, self._real_style = _prepare_image(real_image, real_options, 'real_image')
        self._q, self._q_style = _prepare_image(reciprocal_image, q_options, 'reciprocal_image', reciprocal=True)
        self.center, self.center_source = _rq_center(
            self._q.shape, getattr(reciprocal_image, 'center_beam_metadata', None),
            getattr(reciprocal_image, 'conv_factor', None),
        )
        self._real_center = tuple((size - 1) / 2.0 for size in self._real.shape)
        initial = _coerce_rq_calibration(calibration) or RQCalibration()
        correction = initial.inverse_matrix @ _mirror_matrix(initial.mirror_axis)
        rotation = np.degrees(np.arctan2(correction[1, 0], correction[0, 0]))
        if scale is None:
            scale = min(self._real.shape[1] / self._q.shape[1],
                        self._real.shape[0] * self._aspect / self._q.shape[0])
        self._state = self._validate_state(dict(
            rotation_deg=rotation, mirror_axis=initial.mirror_axis, scale=scale,
            translation=translation, real_alpha=real_alpha, reciprocal_alpha=reciprocal_alpha,
        ))
        self._initial = dict(self._state)
        self._on_apply = on_apply
        self.layout = layout
        self.controls = {}
        self._syncing = False
        self._closed = False
        self._affine = Affine2D()
        self.figure = plt.figure(figsize=figsize)
        columns = 1 if layout == 'overlay' else 2
        gs = self.figure.add_gridspec(1, columns, left=.07, right=.70 if interactive else .95,
                                     bottom=.23 if interactive else .14, top=.87, wspace=.25)
        self.axes = tuple(self.figure.add_subplot(gs[0, i]) for i in range(columns))
        self.real_artist = self.axes[0].imshow(self._real, origin='upper', zorder=1, **self._real_style)
        self.reciprocal_artist = self.axes[-1].imshow(self._q, origin='upper', zorder=2, **self._q_style)
        for ax in self.axes:
            ax.set_xlim(-.5, self._real.shape[1] - .5)
            ax.set_ylim(self._real.shape[0] - .5, -.5)
            ax.set_aspect(self._aspect)
            ax.set_facecolor('#20252a')
            ax.set_xlabel(f"Rx ({units or 'pixels'})")
            ax.set_ylabel(f"Ry ({units or 'pixels'})")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{self._tick_origin[1] + x * self._tick_spacing[1]:g}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{self._tick_origin[0] + y * self._tick_spacing[0]:g}"))
            if grid:
                ax.grid(True, alpha=.25)
            else:
                ax.grid(False)
            if not axes_visible:
                ax.set_axis_off()
            if scale_bar is not None:
                bar_pixels = scale_bar / abs(self._tick_spacing[1])
                if bar_pixels > .8 * self._real.shape[1]:
                    plt.close(self.figure)
                    raise ValueError("scale_bar is too long for the real-space field of view.")
                x0, y0 = .08 * self._real.shape[1], .91 * self._real.shape[0]
                ax.plot([x0, x0 + bar_pixels], [y0, y0], color=bar_color, lw=4, zorder=6)
                if bar_label:
                    ax.annotate(f"{scale_bar:g} {units or 'pixels'}", (x0 + bar_pixels / 2, y0),
                                xytext=(0, 6), textcoords='offset points', ha='center', color=bar_color)
        real_title = real_options.get('title', 'Real space')
        q_title = q_options.get('title', 'Diffraction in the real-space frame')
        self.axes[0].set_title(
            f"{real_title} + {q_options.get('title', 'diffraction')}" if layout == 'overlay' else real_title
        )
        if layout == 'side_by_side':
            self.axes[1].set_title(q_title)
        self.figure.suptitle('Real / diffraction orientation', fontsize=16)
        self._info = self.figure.text(.07, .91, '', fontsize=9)
        self._status = self.figure.text(.07, .012, 'Preview only. Apply calibration to store orientation.', fontsize=9)
        self._pivot_artist, = self.axes[-1].plot([], [], '+', color='#ffdc73', markersize=12, zorder=7)
        self._q_vectors = []
        for text, color in (('Qx', '#ffad42'), ('Qy', '#5edfe3')):
            artist = self.axes[-1].annotate('', xy=(0, 0), xytext=(0, 0), zorder=7,
                                           arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
            label = self.axes[-1].annotate(text, xy=(0, 0), xytext=(4, 4),
                                          textcoords='offset points', color=color, zorder=7)
            self._q_vectors.append((artist, label))
        length = .16 * min(self._real.shape[1], self._real.shape[0] * self._aspect)
        anchor = np.array((.12 * self._real.shape[1], .82 * self._real.shape[0]))
        for text, delta in (('Rx', (length, 0)), ('Ry', (0, -length / self._aspect))):
            endpoint = anchor + delta
            self.axes[0].annotate('', xy=endpoint, xytext=anchor,
                                  arrowprops=dict(arrowstyle='->', color='white'), zorder=6)
            self.axes[0].annotate(text, xy=endpoint, xytext=(4, 4),
                                  textcoords='offset points', color='white', zorder=6)
        self._control_axes = []
        if interactive:
            self._make_controls(export_path)
        # Keep Matplotlib widgets alive even when the return value is not assigned.
        self.figure._fourdenoise_rq_viewer = self
        self._close_cid = self.figure.canvas.mpl_connect('close_event', self._on_close)
        self._update()
        if interactive and show:
            backend = plt.get_backend().lower()
            if 'inline' in backend or backend == 'agg':
                warnings.warn(
                    "RQ controls need an interactive backend. In Jupyter install ipympl "
                    "and run %matplotlib widget before creating the viewer; in Spyder use Qt.",
                    UserWarning, stacklevel=2,
                )
        if show:
            plt.show()

    @staticmethod
    def _validate_state(state):
        state = dict(state)
        state['rotation_deg'] = (_finite_number(state['rotation_deg'], 'rotation_deg') + 180) % 360 - 180
        _mirror_matrix(state['mirror_axis'])
        state['scale'] = _finite_number(state['scale'], 'scale', positive=True)
        state['translation'] = _pair(state['translation'], 'translation')
        for key in ('real_alpha', 'reciprocal_alpha'):
            state[key] = _finite_number(state[key], key)
            if not 0 <= state[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")
        return state

    @property
    def parameters(self):
        """Return a copy of the current preview settings."""
        return dict(self._state)

    @property
    def calibration(self):
        """Current orientation as R-to-Q metadata, independent of preview scale/shift."""
        correction = _rotation_matrix(self._state['rotation_deg']) @ _mirror_matrix(self._state['mirror_axis'])
        return RQCalibration.from_matrix(correction.T, mirror_axis=self._state['mirror_axis'],
                                         source='interactive_comparison')

    @property
    def image_matrix(self):
        """3x3 forward map from diffraction ``(x, y, 1)`` pixels to reference pixels."""
        correction = _rotation_matrix(self._state['rotation_deg']) @ _mirror_matrix(self._state['mirror_axis'])
        # Convert y-down pixels to Cartesian components and back, respecting scan aspect.
        linear = (np.diag((1., -1. / self._aspect)) @ correction @ np.diag((1., -1.))) * self._state['scale']
        target = np.asarray(self._real_center[::-1]) + np.asarray(self._state['translation'][::-1])
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = target - linear @ np.asarray(self.center[::-1])
        return matrix

    def transform_points(self, coords):
        """Map diffraction ``(y, x)`` pixel positions into reference pixel coordinates."""
        points = np.asarray(coords, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("coords must have finite (..., 2) pixel coordinates in (y, x) order.")
        matrix = self.image_matrix
        return (points[..., ::-1] @ matrix[:2, :2].T + matrix[:2, 2])[..., ::-1]

    def set_parameters(self, **parameters):
        """Update preview settings atomically; the source data and calibration are unchanged."""
        if self._closed:
            raise RuntimeError("This comparison viewer is closed.")
        unknown = set(parameters) - set(self._state)
        if unknown:
            raise TypeError(f"Unknown comparison parameters: {', '.join(sorted(unknown))}.")
        self._state = self._validate_state({**self._state, **parameters})
        self._sync_controls()
        self._status.set_text('Preview changed. Apply calibration to store orientation.')
        self._update()
        return self

    def _update(self):
        self._affine.set_matrix(self.image_matrix)
        self.reciprocal_artist.set_transform(self._affine + self.axes[-1].transData)
        self.real_artist.set_alpha(self._state['real_alpha'])
        self.reciprocal_artist.set_alpha(self._state['reciprocal_alpha'])
        pivot = self.transform_points(self.center)[::-1]
        self._pivot_artist.set_data([pivot[0]], [pivot[1]])
        correction = self.calibration.inverse_matrix
        length = .16 * min(self._real.shape[1], self._real.shape[0] * self._aspect)
        for index, (arrow, label) in enumerate(self._q_vectors):
            delta = np.diag((1., -1. / self._aspect)) @ correction[:, index] * length
            arrow.xy = tuple(pivot + delta)
            arrow.set_position(tuple(pivot))
            label.xy = tuple(pivot + delta)
        self._info.set_text(
            f"Q pivot (y, x): ({self.center[0]:g}, {self.center[1]:g}) [{self.center_source}]\n"
            f"Q correction: {self._state['rotation_deg']:.3f} deg | "
            f"R to Q: {self.calibration.rotation_deg:.3f} deg | mirror: {self._state['mirror_axis'] or 'none'}"
        )
        if 'log_scale' in self.controls:
            self.controls['log_scale'].valtext.set_text(f"{self._state['scale']:.3g}x")
        self.figure.canvas.draw_idle()

    def _make_controls(self, export_path):
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

        def slider(key, label, ypos, low, high, initial, callback):
            self.figure.text(.77, ypos + .038, label, fontsize=9)
            ax = self.figure.add_axes((.78, ypos, .16, .025))
            control = Slider(ax, '', low, high, valinit=initial)
            control.on_changed(lambda value: callback(value) if not self._syncing else None)
            self.controls[key] = control
            self._control_axes.append(ax)

        slider('real_alpha', 'Real opacity', .80, 0, 1, self._state['real_alpha'],
               lambda value: self.set_parameters(real_alpha=value))
        slider('reciprocal_alpha', 'Diffraction opacity', .71, 0, 1, self._state['reciprocal_alpha'],
               lambda value: self.set_parameters(reciprocal_alpha=value))
        slider('rotation_deg', 'Diffraction correction (deg)', .62, -180, 180,
               self._state['rotation_deg'], lambda value: self.set_parameters(rotation_deg=value))
        angle = TextBox(self.figure.add_axes((.84, .566, .10, .032)), 'Angle ', initial=f"{self._state['rotation_deg']:g}")
        angle.on_submit(lambda text: self._safe_action(lambda: self.set_parameters(rotation_deg=float(text)))
                        if not self._syncing else None)
        self.controls['angle_entry'] = angle
        base_scale = np.log2(self._state['scale'])
        slider('log_scale', 'Diffraction scale', .48, base_scale - 4, base_scale + 4, base_scale,
               lambda value: self.set_parameters(scale=2 ** value))
        extent = max(self._real.shape)
        for index, (key, label, ypos) in enumerate((('shift_y', 'Shift down (R pixels)', .30),
                                                   ('shift_x', 'Shift right (R pixels)', .39))):
            value = self._state['translation'][index]
            def move(new_value, axis=index):
                translation = list(self._state['translation'])
                translation[axis] = new_value
                self.set_parameters(translation=translation)
            slider(key, label, ypos, min(-extent, value), max(extent, value), value, move)
        mirror = RadioButtons(self.figure.add_axes((.79, .15, .16, .10)),
                              ('No mirror', 'Mirror x', 'Mirror y'),
                              active=(None, 'x', 'y').index(self._state['mirror_axis']))
        mirror.on_clicked(lambda text: self.set_parameters(
            mirror_axis={'No mirror': None, 'Mirror x': 'x', 'Mirror y': 'y'}[text]
        ) if not self._syncing else None)
        self.controls['mirror_axis'] = mirror
        path = TextBox(self.figure.add_axes((.17, .065, .49, .035)), 'Export file ', initial=str(export_path))
        self.controls['export_path'] = path
        actions = [('reset', 'Reset', self.reset), ('apply', 'Apply calibration', self.apply),
                   ('export', 'Export figure', lambda: self.export(path.text))]
        for index, (key, label, action) in enumerate(actions):
            button = Button(self.figure.add_axes((.08 + index * .20, .125, .18, .045)), label)
            button.on_clicked(lambda event, action=action: self._safe_action(action))
            self.controls[key] = button

    def _safe_action(self, action):
        try:
            return action()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self._status.set_text(str(exc))
            self.figure.canvas.draw_idle()

    def _sync_controls(self):
        if not self.controls:
            return
        self._syncing = True
        try:
            values = {key: self._state[key] for key in ('real_alpha', 'reciprocal_alpha', 'rotation_deg')}
            values.update(log_scale=np.log2(self._state['scale']), shift_y=self._state['translation'][0],
                          shift_x=self._state['translation'][1])
            for key, value in values.items():
                slider = self.controls[key]
                if value < slider.valmin or value > slider.valmax:
                    slider.valmin = min(value, slider.valmin)
                    slider.valmax = max(value, slider.valmax)
                    slider.ax.set_xlim(slider.valmin, slider.valmax)
                slider.eventson = False
                slider.drawon = False
                slider.set_val(value)
                slider.eventson = True
                slider.drawon = True
            self.controls['angle_entry'].set_val(f"{self._state['rotation_deg']:.6g}")
            self.controls['mirror_axis'].set_active((None, 'x', 'y').index(self._state['mirror_axis']))
        finally:
            self._syncing = False

    def reset(self):
        """Restore the opening preview settings, without reverting an applied calibration."""
        return self.set_parameters(**self._initial)

    def apply(self):
        """Store the current orientation on the owning dataset and return it."""
        if self._closed:
            raise RuntimeError("This comparison viewer is closed.")
        calibration = self.calibration
        if self._on_apply is not None:
            self._on_apply(calibration)
        self._status.set_text('Calibration applied. Source pixels are unchanged.')
        self.figure.canvas.draw_idle()
        return calibration

    def export(self, filename, *, dpi=180, overwrite=False):
        """Save the image panels without controls; an existing file requires overwrite=True."""
        from matplotlib.transforms import Bbox

        path = Path(filename).expanduser()
        if not path.suffix:
            path = path.with_suffix('.png')
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; choose another filename.")
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()
        bounds = Bbox.union([ax.get_tightbbox(renderer) for ax in self.axes])
        bounds = bounds.transformed(self.figure.dpi_scale_trans.inverted()).padded(.10)
        self.figure.savefig(path, dpi=dpi, bbox_inches=bounds)
        self._status.set_text(f"Exported {path}")
        self.figure.canvas.draw_idle()
        return str(path)

    def _on_close(self, event=None):
        if self._closed:
            return
        self._closed = True
        for control in self.controls.values():
            control.disconnect_events()
        self.figure.canvas.mpl_disconnect(self._close_cid)
        self._on_apply = None
        if hasattr(self.figure, '_fourdenoise_rq_viewer'):
            del self.figure._fourdenoise_rq_viewer

    def close(self):
        """Disconnect GUI callbacks and close the comparison figure."""
        import matplotlib.pyplot as plt

        self._on_close()
        plt.close(self.figure)
