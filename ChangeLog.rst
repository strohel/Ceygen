=================
Ceygen Change Log
=================

This file mentions changes between Ceygen versions that are important for its users. Most
recent versions and changes are mentioned on top.

.. currentmodule:: ceygen

Changes in 0.2 since 0.1
========================

*  Added code paths optimized for C-contiguous and F-contiguous matrices and vectors using
   fancy C++ dispatching code. Rougly 40% speed gains in :func:`core.dot_mm` (for common
   matrix sizes), 300% gains for :func:`core.dot_mv` and :func:`core.dot_vm` starting with
   16\*16, 30% gains for vector-vector operations and slight gains at other places.
*  Internal Ceygen .pxd files (e.g. ``eigen_cython.pxd``) are no longer installed.
*  ``-fopenmp`` is now added by default to `build_ext` ``cflags`` and ``ldflags`` to
   enable parellelising :func:`core.dot_mm` in Eigen; speedups are noticeable for matrices
   64\*64 and bigger. Can be easily disabled.
*  :func:`dtype.vector` and :func:`dtype.matrix` convenience functions added; their usage
   in other modules leads to speedups because it circumvents Cython shortcoming.
*  :func:`core.set_is_malloc_allowed` added to aid in debugging and tests.
*  Simple benchmark was added, define ``BENCHMARK`` environment variable during test
   execution to run it.
