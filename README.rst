======
Ceygen
======

About
=====

Ceygen is a binary Python extension module helper for linear algebra with Cython_
`typed memoryviews`_. Cython is built atop of `Eigen C++ library`_. Ceygen is **not**
a Cython wrapper or an interface to Eigen!

The name Ceygen is a rather poor wordplay on Cython + Eigen; it has nothing to do
with software piracy.

.. _Cython: http://cython.org/
.. _`typed memoryviews`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Eigen C++ library`: http://eigen.tuxfamily.org/

Licensing
---------

Ceygen is currently distributed under GNU GPL v2+ license. The authors of
Ceygen are however open to other licensing suggestions. (Do you want to use
Ceygen in e.g. a BSD-licensed project? Ask!)

Features
========

Ceygen...

 * **is fast** - Ceygen's primary raison d'être is to provide overhead-free algebraic
   operations for Cython projects that work with `typed memoryviews`_ (especially
   small-sized). For every function there is a code-path where no Python function is
   called, no memory is allocated on heap and no data is copied.
   `Eigen itself performs rather well`_, too.
 * **is documented** - see :ref:`Documentation`
 * **supports various data types** - Ceygen uses Cython `fused types`_ (a.k.a. wannabe
   templates) along with Eigen's template nature to support various data types without
   duplicating code. While just a few types are pre-defined (float, double, ...), adding
   a new type is a matter of adding 3 lines and rebuilding Ceygen.
 * **is extensively tested** - Ceygen's test suite validates every public Cython method,
   including errors raised on invalid input.
 * **is multithreading-friendly** - Every Ceygen function doesn't acquire the GIL_
   unless it needs to create a Python object (always avoidable); all functions are
   declared nogil_ so that you can call them in prange_ blocks without losing parallelism.
 * **provides descriptive error messages** - Care is taken to propagate all errors
   properly (down from Eigen) so that you are not stuck debugging your program. Ceygen
   functions don't crash on invalid input but rather raise reasonable errors.
 * works well with NumPy_, but doesn't depend on it. You don't need NumPy to build or run
   Ceygen, but thanks to Cython, `Cython memoryviews and NumPy arrays`_ are fully
   interchangeable without copying the data (where it is possible). The test suite
   currently makes use of NumPy because of our laziness. :-)

.. _`Eigen itself performs rather well`: http://eigen.tuxfamily.org/index.php?title=Benchmark
.. _`fused types`: http://docs.cython.org/src/userguide/fusedtypes.html
.. _GIL: http://docs.python.org/glossary.html#term-global-interpreter-lock
.. _nogil: http://docs.cython.org/src/userguide/external_C_code.html#declaring-a-function-as-callable-without-the-gil
.. _prange: http://docs.cython.org/src/userguide/parallelism.html
.. _NumPy: http://www.numpy.org/
.. _`Cython memoryviews and NumPy arrays`: http://docs.cython.org/src/userguide/memoryviews.html#coercion-to-numpy

On the other hand, Ceygen...

 * **depends on Eigen build-time**. Ceygen expects *Eigen 3* headers to be installed under
   ``/usr/lib/eigen3`` when it is being built. Installing Eigen is a matter of unpacking
   it, because it is a pure template library defined solely in the headers. Ceygen doesn't
   reference Eigen at all at runtime because all code is complited in.
 * **still provides a very little subset of Eigen functionality**. We add new functions
   only as we need them in another projects, but we believe that the hard part is the
   infrastructure - implementing a new function should be rather straightforward (with
   decent Cython and C++ knowledge). We're very open to pull requests!
   (do include unit tests in them)
 * **needs recent Cython** to compile. [#cythonvers]_ If this is a problem, you can
   distribute .cpp files or final Python extension module instead.

Building
========

Ceygen uses standard Distutils to build, test and install itself, simply run
``./setup.py build`` to build Ceygen, ``./setup.py test`` to test it (inside build
directory) and ``./setup.py install`` to install it.

.. _Documentation:

Documentation
=============

Ceygen documentation is maintained in reStructuredText_ format under ``doc/`` directory
and can be exported into a variety of formats using Sphinx_ (version at least 1.0 needed).
Just type ``make`` in that directory to see a list of supported formats and for example
``make html`` to build HTML pages with the documentation.

On-line documentation is in the works.

.. _reStructuredText: http://sphinx-doc.org/rest.html
.. _Sphinx: http://sphinx-doc.org/

.. rubric:: Footnotes

.. [#cythonvers] currently this is at least Cython 0.18 rc1.
