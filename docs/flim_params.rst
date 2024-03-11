FLIMParams classes
================================================

The ``FLIMParams`` class, and their variants,
stores parameters for fitting FLIM data.

Initialization of the ``FLIMParams`` class
-------------------------------------------

The :ref:`flim params<FLIMParams>` class can be initialized
either using the ``from_tuple`` class method or by
passing a list of :ref:`flim parameter<FLIMParameter>` instances.
The :ref:`flim parameter<FLIMParameter>` classes are UNITFUL --
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

The :ref:`flim params<FLIMParams>` class provides a context
manager for its units internally so that most
methods (other than ``convert_units``) do not
cryptically change the units (and thus the numerical
values used for the various parameters). To use the
context manager, use ``with fp.as_units('unit'):``
where ``fp`` is a :ref:`flim params<FLIMParams>` instance and
``unit`` is the desired unit. ``unit`` may be a 
string or a :ref:`flim units<FlimUnits>` unit.

.. code-block:: python

    from siffpy.core.flim import FLIMParams, Exp, Irf

    fp = FLIMParams(
        Exp(0.5, 0.3, units = 'nanoseconds'),
        Exp(2.6, 0.7, units = 'nanoseconds'),
        Irf(4.1, 0.03, units = 'nanoseconds'),
    )

    print(fp.units) # nanoseconds

    with fp.as_units('countbins'):
        # do some processing
        analyses_with_fp(fp, ...)

    print(fp.units) # still nanoseconds

Saving ``FLIMParams``
---------------------

The :ref:`flim params<FLIMParams>` class can be saved to a file
using the ``save`` method. The file format is a simple JSON file, but
uses the suffix ``.flimparams``. These data can be loaded as a dict
and used to instantiate a new :ref:`flim params<FLIMParams>` instance
with the ``from_dict`` class method, or the ``load`` class method with
a path to file.

.. code-block:: python

    from siffpy.core.flim import FLIMParams, Exp, Irf

    fp = FLIMParams(
        Exp(0.5, 0.3, units = 'nanoseconds'),
        Exp(2.6, 0.7, units = 'nanoseconds'),
        Irf(4.1, 0.03, units = 'nanoseconds'),
    )

    fp.save('my_params.flimparams')

    # later
    fp = FLIMParams.load('my_params.flimparams')

    # OR:
    import json
    with open('my_params.flimparams', 'r') as f:
        data = json.load(f)
    fp = FLIMParams.from_dict(data)


.. _flim units:

.. autoclass:: siffpy.core.flim.flimunits.FlimUnits
    :members:

.. _flim params:

.. autoclass:: siffpy.core.flim.flimparams.FLIMParams
    :members:

.. _flim parameter:

.. autoclass:: siffpy.core.flim.flimparams.FLIMParameter
    :members:

.. autoclass:: siffpy.core.flim.flimparams.Exp:
    :members:

.. autoclass:: siffpy.core.flim.flimparams.Irf:
    :members: