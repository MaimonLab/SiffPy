.. _registration:
``RegistrationInfo`` classes
=============================

``RegistrationInfo`` classes are the interface to registration that are
used in ``SiffPy``. The point of this class is that there are many registration methods
one *might* want to use, but most of them don't accept a ``.siff`` file. If you convert
it to a ``.tiff`` file, you usually end up duplicating your data and you also lose the
FLIM data. So these classes are used to bridge your ``.siff`` file's intensity data to
a format palatable for registration and then retain the way that the images were warped
so that the transformations can be applied to photons in the ``.siff`` file.

Core implementation
-------------------

``SiffIO`` frame objects perform registration *during reads*, meaning they accept
an argument ``registration_dict : Dict[int, Tuple[int,int]]`` to functions like
``get_frames(...)``, where each key is the index of a frame and each value is the
pixelwise shifts in the y and x direction respectively. This means that the current
implementations **only perform rigid registration** -- future implementations may
permit non-rigid registration, but it will be more complicated to store (especially
without duplicating the whole file the way many pipelines do). The additional latency
in rigid registration of frames as they are read is nearly-negligible, so I have not
spent time worrying about implementing duplicative-type registration methods that are
fixed. TODO: provide this option?

Any ``RegistrationInfo`` class **must** define the following methods:

- ``register(self, siffio : SiffIO, *args, alignment_color_channel : int =0, **kwargs)->None:``
    This function is usually the one called on a file opened by a :ref:`_siffio<``SiffIO``>`
    object. It accepts whatever other arguments and keyword arguments are needed to
    perform the registration. It should then modify the ``SiffIO`` object *in place* to
    store the registration parameters, e.g. in a ``reference_frames`` or ``yx_shifts``
    parameter.

Any ``RegistrationInfo`` class *can*, but **is not required** to define the following
class attributes:

- ``multithreaading_compatible : bool``
   ``multithreaading_compatible`` defines whether or not you can run the ``register``
   method outside of the main thread. Some packages, e.g. ``suite2p``, do not permit
   this, because they use multithreading internally already and some of their tools
   fail in nested threads

- ``backend : RegistrationType``
    The :ref:`registration_type<``RegistrationType``>` enum defined in ``registration_info.py`` enumerates the
    list of implemented core ``RegistrationInfo`` classes and provides typical ``str``
    names for them, making it easier for automated code inspection to generate GUI
    tools (as in ``siff-napari``) and providing aliases that can be stored in metadata
    to reconstruct the class of tool being used (as in the base class's ``load`` method).

Saving and loading
------------------

:ref:`registration_info_abc<``RegistrationInfo``>` classes are designed to be saved and loaded from disk. The
``RegistrationInfo`` base class provides a ``save`` and ``load_as_dict`` method that stores
the basic paremeters in a ``.h5`` file (with no custom suffix, in this case). The stored
attributes are:

- ``filename``
    The filename of the source ``.siff`` file. This only saves the **stem** (e.g.
    a file named ``a_long_custom_path/in_a_private_directory/on_a_machine_that_deanonymizes_the_user/imaging.siff``
    will be stored as ``imaging``) -- it's not intended to perfectly uniquely identify
    a file, but provide a reminder.

-  ``registration_type``
    A string that can be mapped using the :ref:`registration_type<``RegistrationType``>`
    ``Enum`` to the class of ``RegistrationInfo`` that was used to register the file.

-  ``registration_color``
    The color channel used for registration

- ``yx_shifts``
    The framewise shifts in the y and x direction for that frame, stored as a
    ``Group`` with ``h5py``. The ``Datasets`` stored are:
    - ``frame_index`` the indices for stored frames
    - ``shift_values`` tuples of (y,x) shifts
    - ``reference_frames`` the reference frames used for registration, if any

Rather than a direct ``load`` class function or ``staticmethod``, this package uses
``load_as_dict`` to return a ``dict`` object which can be parsed (and passed through
inspection tools) to determine the class of ``RegistrationInfo`` (and reinstantiate it
if needed). This is partly to get around needing a ``SiffIO`` object if you don't need
to work directly with a file anymore and partly to make it easy for others to use the
objects without learning to use this specific framework if they want...

.. _registration_type:

.. autoclass:: siffpy.core.utils.registration_tools.registration_info.RegistrationType
   :members:
   :undoc-members:
   :show-inheritance:

.. _registration_info_abc:

.. autoclass:: siffpy.core.utils.registration_tools.registration_info.RegistrationInfo
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: siffpy.core.utils.registration_tools.suite2p.Suite2pRegistrationInfo
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: siffpy.core.utils.registration_tools.siffpy.SiffpyRegistrationInfo
   :members:
   :undoc-members:
   :show-inheritance:

Calling with ``SiffReader``
---------------------------

You can call the ``register`` method of a ``SiffReader`` object to
perform registration:

.. code-block:: python

    from siffpy import SiffReader
    from siffpy.core.utils.registration_tools import RegistrationType

    reader = SiffReader("path/to/file.siff")
    reader.register(
        registration_method = 'suite2p',
        alignment_color_channel = 0,
        batch_size = 200,
        do_bidiphase = True,
        smooth_sigma_time = 5,
        norm_frames = False
    )

The convention and syntax for calling ``register`` is:

.. autoclass:: siffpy.core.siffreader.SiffReader
    :members: register