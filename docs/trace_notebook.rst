Analysis with ``Trace``-type objects
====================================

Most of the time, you’ll probably just want a ``numpy`` array and to run
off and do your own analyses. But this can be a little bit tricky if
you’re new to working with FLIM data, because you can’t do simply
operations like addition or multiplication. Even worse, pipelines that
you pass your FLIM data into will not know that it has different rules!
So the ``Trace`` classes extend ``numpy`` arrays to overwrite some of
their functions and make sure that they behave acceptably.

There are also ``FluorescenceTrace`` type ``Trace`` classes that are
mostly for convenience: they store metadata and make sure that things
transform properly, but impose very few changes to the overall
structure. This page documents a few operations you can do with the
``Trace`` classes.


