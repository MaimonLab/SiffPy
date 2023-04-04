# Registration

This section contains material on image registration. Most image registration
pipelines operate on either a `.tiff` file or some sort of array-like structure.
Most types of `.siff` files do not contain a directly-readable array of that
sort, and so piping data from `.siff` files into them takes a little bit more
wrangling.

Notably, this backend is specifically for I/O using a `SiffIO` or `SiffReader`
class, so it tracks lists or dictionaries of pixelwise shifts. It is only able
to perform rigid registration, for this reason (because individual photons
can only be assigned to one pixel at a time, this gets rid of me needing to
think about linear mixing of pixels, though that is probably the right
future direction).

## RegistrationInfo

The core interface for various registration algorithms is the `RegistrationInfo`
tool. This is a base class, and each registration tool implemented subclasses it.

A `RegistrationInfo` has a few attributes:

- `registration_type`, a `RegistrationType` `Enum` class that can be used
to specifiy a particular `RegistrationInfo` on initialization.

- `filename` : `str` the name of the file being processed (for confirmation on
loading)

- `yx_shifts` : `dict[int, tuple[int,int]]` every individual frame's shift in `y` and `x`.

- `reference_frames` : `np.ndarray` the template image for each z plane

- `im_params` : `siffpy.core.utils.ImParams` the parameters of the image

And a few methods:

- `register(self, siffio : SiffIO, *args, **kwargs)->None` which updates
the `yx_shifts` attrribute.

- `align_to_reference(self, images : np.ndarray, z_plane : int)->list[tuple[int,int]]` called during registration, aligns a set of frames to a template.

- `save(self, save_path)` stores the `RegistrationInfo` object as an `.hdf5`. Does not store the `ImParams` or `SiffIO` object

- `assign_siffio(self, siffio: SiffIO)` stores a reference to the passed
`SiffIO` object, so that when you load with a new `SiffReader` you don't need to create a new `SiffIO` and have it open the file.

- `load(path, siffio = None, im_params = None)` a `classmethod` to create a `RegistrationInfo` from an `.hdf5` file.
If the `siffio` or `im_params` argument are passed, it will
add these to your `RegistrationInfo` returned.