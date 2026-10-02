Views and Results API
=====================

These objects retain the calibration or analysis context needed to interpret
their arrays. Shape-changing methods generally return new objects; inspect
the individual method contracts before relying on that behavior.

Real-space images
-----------------

.. autoclass:: fourdenoise.RealSpace

   .. automethod:: __init__

.. automethod:: fourdenoise.RealSpace.show

.. automethod:: fourdenoise.RealSpace.set_scale

.. automethod:: fourdenoise.RealSpace.crop

.. automethod:: fourdenoise.RealSpace.resize

.. automethod:: fourdenoise.RealSpace.copy

Diffraction patterns
--------------------

.. autoclass:: fourdenoise.ReciprocalSpace

   .. automethod:: __init__

.. automethod:: fourdenoise.ReciprocalSpace.show

.. automethod:: fourdenoise.ReciprocalSpace.set_scale

.. automethod:: fourdenoise.ReciprocalSpace.get_peaks

.. automethod:: fourdenoise.ReciprocalSpace.get_centers

.. automethod:: fourdenoise.ReciprocalSpace.get_intensities

.. automethod:: fourdenoise.ReciprocalSpace.block_direct_beam

.. automethod:: fourdenoise.ReciprocalSpace.inpaint_background

.. automethod:: fourdenoise.ReciprocalSpace.copy

Analysis results
----------------

.. autoclass:: fourdenoise.RQCalibration
   :members: matrix, inverse_matrix, transform_vectors, to_dict, from_matrix

.. autoclass:: fourdenoise.RQComparison
   :members: parameters, calibration, image_matrix, transform_points, set_parameters, apply, reset, export, close

.. autoclass:: fourdenoise.PeakDetectionResult
   :members: observed_mask

.. autoclass:: fourdenoise.StrainResult
   :members: valid_mask, as_real_space
