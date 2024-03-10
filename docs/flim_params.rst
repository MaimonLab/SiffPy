FLIMParams classes
==================

The ``FLIMParams`` class, and their variants,
stores parameters for fitting FLIM data.

Initialization of the ``FLIMParams`` class
-------------------------------------------

The :ref:`flim_params<``FLIMParams``>` class can be initialized
either using the ``from_tuple`` class method or by
passing a list of :ref:`flim_parameter<``FLIMParameter``>` instances.
The :ref:`flim_parameter<``FLIMParameter``>` classes are UNITFUL --
for some purposes it's easiest to work with indices of the arrival
time bins and sometimes it's easiest to work with real SI units.

.. code-block:: python

    from siffpy.core.flim import FLIMParams, Exp, Irf

    fp = FLIMParams.from_tuple(
        (
            0.5, 0.3, # tau, fraction in state 1
            2.2, 0.5, # tau, fraction in state 2
            6.2, 0.2, # tau, fraction in state 3
            4.1, 0.03 # IRF mean offset and breadth
        ),
        units = 'nanoseconds'
    )
    
    # Equivalent
    fp = FLIMParams(
        # any of these params could be defined
        # with their own units if desired
        Exp(0.5, 0.3, units = 'nanoseconds'),
        Exp(2.2, 0.5, units = 'nanoseconds'),
        Exp(6.2, 0.2, units = 'nanoseconds'), 
        Irf(4.1, 0.03, units = 'nanoseconds'),
    )

``FlimUnits`` and the ``as_units`` context manager
--------------------------------------------------

The :ref:`flim_params<``FLIMParams``>` class provides a context
manager for its units internally so that most
methods (other than ``convert_units``) do not
cryptically change the units (and thus the numerical
values used for the various parameters). To use the
context manager, use ``with fp.as_units('unit'):``
where ``fp`` is a :ref:`flim_params<``FLIMParams``>` instance and
``unit`` is the desired unit. ``unit`` may be a 
string or a :ref:`flimunit<``FlimUnits``>` unit.

.. code-block:: python

    from siffpy.core.flim import FLIMParams, Exp, Irf

    fp = FLIMParams(

    )

.. _flim_units:
.. autoclass:: siffpy.core.flim.flimunits.FlimUnits
    :members:

.. _flim_params:
.. autoclass:: siffpy.core.flim.flimparams.FLIMParams
    :members:

.. _flim_parameter:
.. autoclass:: siffpy.core.flim.flimparams.FLIMParameter
    :members:

.. autoclass:: siffpy.core.flim.flimparams.Exp:
    :members:

.. autoclass:: siffpy.core.flim.flimparams.Irf:
    :members: