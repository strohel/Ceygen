=================
Ceygen Change Log
=================

This file mentions changes between Ceygen versions that are important for its users. Most
recent versions and changes are mentioned on top.

.. currentmodule:: ceygen

Changes in 0.2 since 0.1
========================

*  Internal Ceygen .pxd files (e.g. ``eigen_cython.pxd``) are no longer installed.
*  ``-fopenmp`` is now added by default to `build_ext` ``cflags`` and ``ldflags`` to
   enable parellelising :func:`core.dot_mm` in Eigen; speedups are noticeable for matrices
   64\*64 and bigger. Can be easily disabled.
*  :func:`dtype.vector` and :func:`dtype.matrix` convenience functions added; their usage
   in other modules leads to speedups because it circumvents Cython shortcoming.
*  :func:`core.set_is_malloc_allowed` added to aid in debugging and tests.
*  :func:`core.dot_mm` was optimized for C-contiguous and F-contiguous matrices resulting
   in roughly 40% speed gains.
*  Simple benchmark was added, define ``BENCHMARK`` environment variable during test
   execution to run it.
