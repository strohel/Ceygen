======
Ceygen
======

About
=====

Ceygen is a binary Python extension module for linear algebra with Cython_ `typed
memoryviews`_. Ceygen is built atop the `Eigen C++ library`_. Ceygen is **not** a Cython
wrapper or an interface to Eigen!

The name Ceygen is a rather poor wordplay on Cython + Eigen; it has nothing to do
with software piracy. Ceygen is currently distributed under GNU GPL v2+ license. The
authors of Ceygen are however open to other licensing suggestions. (Do you want to use
Ceygen in e.g. a BSD-licensed project? Ask!)

Cython is being developed by Matěj Laitl with support from the `Institute of Information
Theory and Automation, Academy of Sciences of the Czech Republic`_. Feel free to send me
a mail to matej at laitl dot cz.

.. _Cython: http://cython.org/
.. _`typed memoryviews`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Eigen C++ library`: http://eigen.tuxfamily.org/
.. _`Institute of Information Theory and Automation, Academy of Sciences of the Czech Republic`:
   http://www.utia.cas.cz/

Features
========

Ceygen...

* **is fast** - Ceygen's primary raison d'être is to provide overhead-free algebraic
  operations for Cython projects that work with `typed memoryviews`_ (especially
  small-sized). For every function there is a code-path where no Python function is
  called, no memory is allocated on heap and no data is copied.
  `Eigen itself performs rather well`_, too.
* **is documented** - see `Documentation`_ or hop directly to `on-line docs`_.
* **supports various data types** - Ceygen uses Cython `fused types`_ (a.k.a. wannabe
  templates) along with Eigen's template nature to support various data types without
  duplicating code. While just a few types are pre-defined (float, double, ...), adding
  a new type is a matter of adding 3 lines and rebuilding Ceygen.
* **is extensively tested** - Ceygen's test suite validates every its public method,
  including errors raised on invalid input. Thanks to Travis CI, `every push is
  automatically tested`_ against **Python 2.6**, **2.7**, **3.2** and **3.3**.
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
.. _`on-line docs`: http://strohel.github.com/Ceygen-doc/
.. _`fused types`: http://docs.cython.org/src/userguide/fusedtypes.html
.. _`every push is automatically tested`: https://travis-ci.org/strohel/Ceygen
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
* **needs recent Cython** (currently at least 0.19.1) to compile. If this is a problem,
  you can distribute .cpp files or final Python extension module instead.
* **doesn't bring Eigen's elegance to Cython** - if you think of lazy evaluation and
  advanced expessions, stop dreaming. Ceygen will make your code faster, not nicer.
  `Array expessions`_ will help here.

.. _`Array expessions`: https://github.com/cython/cython/pull/144

A simple example to compute matrix product within a big matrix may look like

>>> cdef double[:, :] big = np.array([[1.,  2.,   2.,  0.,   0.,  0.],
>>>                                   [3.,  4.,   0., -2.,   0.,  0.]])
>>> ceygen.core.dot_mm(big[:, 0:2], big[:, 2:4], big[:, 4:6])
[[ 2. -4.]
 [ 6. -8.]]
>>> big
[[ 1.  2.   2.  0.   2. -4.]
 [ 3.  4.   0. -2.   6. -8.]],

where the `dot_mm`_ call above doesn't copy any data, allocates no memory on heap, doesn't
need the GIL_ and uses vectorization (SSE, AltiVec...) to get the best out of your
processor.

.. _`dot_mm`: http://strohel.github.com/Ceygen-doc/core.html#ceygen.core.dot_mm

Obtaining
=========

Ceygen development happens in `its github repository`_, ``git clone
git@github.com:strohel/Ceygen.git`` -ing is the preferred way to get it as you'll have
the latest & greatest version (which shouldn't break thanks to continuous integration).
Released versions are available from `Ceygen's PyPI page`_.

.. _`its github repository`: https://github.com/strohel/Ceygen
.. _`Ceygen's PyPI page`: http://pypi.python.org/pypi/Ceygen

Building
========

Ceygen uses standard Distutils to build, test and install itself, simply run:

* ``python setup.py build`` to build Ceygen
* ``python setup.py test`` to test it (inside build directory)
* ``python setup.py install`` to install it
* ``python setup.py clean`` to clean generated object, .cpp and .html files (perhaps to
  force recompilation)

Commands can be combined, automatically call dependent commands and can take options,
the recommended combo to safely install Ceygen is therefore ``python setup.py -v test install``.

Building Options
----------------

You can set various build options as it is usual with distutils, see
``python setup.py --help``. Notable is the `build_ext` command and its `--include-dirs`
(standard) and following additional options (whose are Ceygen extensions):

--include-dirs
   defaults to `/usr/include/eigen3` and must be specified if you've installed Eigen 3
   to a non-standard directory,

--cflags
   defaults to `-O2 -march=native -fopenmp`. Please note that it is important to enable
   optimizations and generation of appropriate MMX/SSE/altivec-enabled code as the actual
   computation code from Eigen is built along with the boilerplate Ceygen code,

--ldflags
   additional flags to pass to linker, defaults to `-fopenmp`. Use standard `--libraries`
   for specifying extra libraries to link against,

--annotate
   pass `--annotate` to Cython to produce annotated HTML files during compiling. Only
   useful during Ceygen development.

You may want to remove `-fopenmp` from `cflags` and `ldflags` if you are already
parallelising above Ceygen. The resulting command could look like ``python setup.py -v
build_ext --include-dirs=/usr/local/include/eigen3 --cflags="-O3 -march=core2" --ldflags=
test``. The same could be achieved by putting the options to a `setup.cfg` file::

   [build_ext]
   include_dirs = /usr/local/include/eigen3
   cflags = -O3 -march=core2
   ldflags =

Documentation
=============

Ceygen documentation is maintained in reStructuredText_ format under ``doc/`` directory
and can be exported into a variety of formats using Sphinx_ (version at least 1.0 needed).
Just type ``make`` in that directory to see a list of supported formats and for example
``make html`` to build HTML pages with the documentation.

See ``ChangeLog.rst`` file for changes between versions or `view it online`_.

**On-line documentation** is available at http://strohel.github.com/Ceygen-doc/

.. _reStructuredText: http://sphinx-doc.org/rest.html
.. _Sphinx: http://sphinx-doc.org/
.. _`view it online`: http://strohel.github.com/Ceygen-doc/ChangeLog.html

Bugs
====

Please report any bugs you find and suggestions you may have to `Ceygen's github Issue
Tracker`_.

.. _`Ceygen's github Issue Tracker`: https://github.com/strohel/Ceygen/issues
