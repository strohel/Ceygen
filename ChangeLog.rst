=================
Ceygen Change Log
=================

This file mentions changes between Ceygen versions that are important for its users. Most
recent versions and changes are mentioned on top.

.. currentmodule:: ceygen

Changes in 0.4 since 0.3
========================

Changes in 0.3 since 0.2
========================

*  :func:`~core.eigen_version` function introduced to get Eigen version.
*  :mod:`~ceygen.llt` module introduced with Cholesky matrix decomposition.
*  :obj:`~dtype.dtype` enhanced to provide C char, short, int, long and float types in
   addition to C double type. :obj:`~dtype.nonint_dtype` introduced for non-integer
   numeric types. If you get *no suitable method found* or *Invalid use of fused types,
   type cannot be specialized* Cython errors, specify the specialization explicitly:
   ``ceygen.elemwise.add_vv[double](np.array(...), np.array(...))``. This unfortunately
   slows down compilation and makes resulting modules bigger, but doesn't affect
   performance and makes Ceygen more generic.
*  :func:`~elemwise.power_vs` and :func:`~elemwise.power_ms` functions were added to the
   :mod:`~ceygen.reductions` module.

Changes in 0.2 since 0.1
========================

*  :mod:`~ceygen.reductions` module was added with vector, matrix, row-wise and column-wise
   sums.
*  Simple benchmarks for many functions have been added, define ``BENCHMARK`` or
   ``BENCHMARK_NUMPY`` environment variable during test execution to run them; define
   ``SAVE`` environment variable to save timings into ``.pickle`` files that can be
   visualized by ``support/visualize_stats.py``.
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
