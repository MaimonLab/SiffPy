# Napari Viewers

`napari` `Viewer` type objects are not extensible, but this section creates families of classes
that use `Viewer` objects for different purposes. So I've created an object, a `NapariInterface`,
that keeps track of a `Viewer` object and behaves a lot *like* a `Viewer` without actually *being*
a `Viewer`. Any method you can call with a `Viewer`, you can call with a `NapariInterface` and it will
call the function of the `Viewer`, but each subtype of `NapariInterface` will have its own functionality
that accompanying `SiffVisualizer` objects can interact with if they're using their `napari` backends.

This organization is basically necessary to keep `HoloViews` and `napari` separated while still allowing
both functionalities to share the same `SiffVisualizer` objects.
