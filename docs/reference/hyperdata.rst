HyperData API
=============

``HyperData`` wraps a 2D image, 3D pattern stack, or 4D scan. The public
methods below are selected entry points, not an inventory of implementation
helpers. See the :doc:`data model <../concepts/data-model>` and task guides
before using a method on a new dataset.

.. autoclass:: fourdenoise.HyperData

   .. automethod:: __init__

Loading and metadata
--------------------

.. automethod:: fourdenoise.HyperData.open_hdf5

.. automethod:: fourdenoise.HyperData.to_polar_hdf5

.. automethod:: fourdenoise.HyperData.save

.. automethod:: fourdenoise.HyperData.copy

Viewing and geometry
--------------------

.. automethod:: fourdenoise.HyperData.compare_rq

.. automethod:: fourdenoise.HyperData.set_rq_calibration

.. automethod:: fourdenoise.HyperData.get_dp

.. automethod:: fourdenoise.HyperData.virtual_image

.. automethod:: fourdenoise.HyperData.visualize

.. automethod:: fourdenoise.HyperData.crop

.. automethod:: fourdenoise.HyperData.resize

.. automethod:: fourdenoise.HyperData.to_polar

.. automethod:: fourdenoise.HyperData.to_cartesian

Preprocessing and denoising
---------------------------

.. automethod:: fourdenoise.HyperData.clip

.. automethod:: fourdenoise.HyperData.alignment

.. automethod:: fourdenoise.HyperData.block_direct_beam

.. automethod:: fourdenoise.HyperData.unfold

.. automethod:: fourdenoise.HyperData.reshape

.. automethod:: fourdenoise.HyperData.denoise

.. automethod:: fourdenoise.HyperData.denoise_info

.. automethod:: fourdenoise.HyperData.rank_scree

Peak and strain analysis
------------------------

.. automethod:: fourdenoise.HyperData.get_peaks

.. automethod:: fourdenoise.HyperData.get_centers

.. automethod:: fourdenoise.HyperData.get_intensities

.. automethod:: fourdenoise.HyperData.get_strains

.. automethod:: fourdenoise.HyperData.get_clusters
