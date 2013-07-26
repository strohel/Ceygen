==================
Ceygen Development
==================

This document should serve as a reminder to me and other possible Ceygen
hackers about Ceygen coding style and conventions.

Development Guidelines
======================

Some special and important files:

 * ``eigen_cpp.h`` - low-level implementation of a tiny C++ Eigen subclass that
   is used to create wrappers around Cython arrays.
 * ``eigen_cython.pxd`` - exports BaseMap C++ class defined in `eigen_cpp.h`
   to Cython along with other Eigen methods.
 * ``dtype.{pxd,pyx}`` - defines the base scalar fused (template-like) type
   that all other functions use, along with functions to create vectors and
   matrices.
 * ``dispatch.{h,pxd}`` - contains fancy code and Cython declarations for
   so-called dispatchers: tiny helpers that call more optimized Eigen functions
   (in fact, the same functions with different template parameters) for
   column-contiguous, row-contiguous matrices and contiguous vectors.

All other \*.{pxd,pyx} are public Ceygen modules.

Please always use appropriate \*Dispatcher from `dispatch.pxd` instead of
calling methods from `eigen_cython.pxd` directly, because declarations from
`eigen_cython.pxd` don't contain ``except +`` keyword for performance reasons
(i.e. you would leak C++ exceptions raised by Eigen code without converting
them to Python exceptions).

Tests and Stress Tests
======================

All public functions should have a unit test. Suppose you have a module
``ceygen/modname.pyx``, then unit tests for all functions in ``modname.pyx``
should go into ``ceygen/tests/test_modname.py``. There is a couple of
"standard" environment variables recognized in tests:

 * ``BENCHMARK`` - run potentially time-consuming benchmarks of Ceygen code
 * ``BENCHMARK_NUMPY`` - also run some benchmarks with NumPy backend to see
   difference
 * ``SAVE`` - save timings into ``.pickle`` files that can be visualized by
   ``support/visualize_stats.py``.

Releasing Ceygen
================

Things to do when releasing new version (let it be **X.Y**) of Ceygen:

Before Tagging
--------------

1. Set version to **X.Y** in `setup.py` (around line 37)
#. Ensure `ChangeLog.rst` mentions all important changes
#. Ensure that `README.rst` is up-to-date
#. (Optional) update **short description** in `setup.py`
#. (Optional) update **long description** `README.rst`

Tagging & Publishing
--------------------

1. Do ``./setup.py sdist`` and check contents, unpack somewhere, run tests incl.
   benchmarks
#. git tag -s **vX.Y**
#. ./setup.py register sdist upload --sign
#. Build and upload docs: ``cd ../ceygen-doc && ./synchronise.sh``
#. If **short description** changed, update it manually at following places:

   * https://github.com/strohel/Ceygen
#. If **long description** changed, update it manually at following places:

   * http://scipy.org/Topical_Software
   * http://www.ohloh.net/p/ceygen

After
-----

1. Set version to **$NEXT_VERSION-pre** in `setup.py`
#. Add header for the next version into `ChangeLog.rst`
