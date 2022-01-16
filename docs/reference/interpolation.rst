========================
Trajectory Interpolation
========================

Use interpolators to obtain trajectory functions where you input a time (or array of times) and return the trajectory at that timepoint (or timepoints).

Interpolator Class
------------------

This is the primary way to create an interpolation function. This handles many edge-cases to ensure behavior is the same as the real-life `Ctrl-P <https://github.com/cbteeple/ctrlp>`_ control system.


.. autoclass:: sorotraj.interpolator.Interpolator
   :members: 



Custom Back-End Interpolators
-----------------------------

	Several custom interpolation classes are used under the hood to make the functions behave like the physical control system. Below you can find documentation for these classes

.. automodule:: sorotraj.interpolator
   :members: 
   :exclude-members: Interpolator