=============================
Core Data Types and Functions
=============================

This module provides basic linear algebra operations such as vector and matrix
products as provided by the <`Eigen/Core`_> include.

Core Data Types
===============

.. module:: ceygen.dtype

.. data:: dtype

   Cython `fused type`_, a selection of C char, short, int, long, float and double
   (Python :obj:`float`).

.. data:: nonint_dtype

   Cython `fused type`_ for methods that cannot work with integer types (such as
   :func:`~ceygen.lu.inv`).

.. function:: vector(size, like)

   Convenience function to create a new vector (*cython.view.array*) and return a
   memoryview of it. This function is declared *with gil* (it can be called without the
   GIL_ held, but acquires it during execution) and is rather expensive (as many Python
   calls are done).

   :param int size: number of elements of the desired vector
   :param like: dummy pointer to desired data type; value not used
   :type like: :obj:`dtype * <ceygen.dtype.dtype>`
   :rtype: |vector|

.. function:: matrix(rows, col, like)

   Convenience function to create a new matrix (*cython.view.array*) and return a
   memoryview of it. This function is declared *with gil* (it can be called without the
   GIL_ held, but acquires it during execution) and is rather expensive (as many Python
   calls are done).

   :param int rows: number of rows of the desired matrix
   :param int cols: number of columns of the desired matrix
   :param like: dummy pointer to desired data type; value not used
   :type like: :obj:`dtype * <ceygen.dtype.dtype>`
   :rtype: |matrix|

Linear Algebra Functions
========================

.. module:: ceygen.core

.. function:: dot_vv(x, y)

   Vector-vector dot product, returns a scalar of appropriate type.

   :param x: first factor
   :type x: |vector|
   :param y: second factor
   :type y: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |scalar|

.. function:: dot_mv(x, y[, out=None])

   Matrix-(column) vector product, returns a vector of appropriate type.

   :param x: first factor (matrix)
   :type x: |matrix|
   :param y: second factor (vector)
   :type y: |vector|
   :param out: |out|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: dot_vm(x, y[, out=None])

   (Row) vector-matrix product, returns a vector of appropriate type. This is equivalent
   to dotvm(*y*.T, *x*) because there's no distinction between row and column vectors in
   Cython memoryviews, but calling this function directly may incur slightly less
   overhead.

   :param x: first factor (vector)
   :type x: |vector|
   :param y: second factor (matrix)
   :type y: |matrix|
   :param out: |out|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: dot_mm(x, y[, out=None])

   Matrix-matrix product, returns a matrix of appropriate type and dimensions. You may of
   course use this function to multiply matrices that are in fact vectors, you just need
   to pay attention to column-vector vs. row-vector distinction this time.

   If both *x* and *y* are contiguous in some way (either C or Fortran, independently),
   this function takes optimized code path that doesn't involve memory allocation in
   Eigen; speed gains are around 40% for matrices around 2\*2 -- 24\*24 size. No special
   markup is needed to trigger this. See also :func:`set_is_malloc_allowed`.

   :param x: first factor
   :type x: |matrix|
   :param y: second factor
   :type y: |matrix|
   :param out: |out|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

Miscellaneous Functions
=======================

.. function:: set_is_malloc_allowed(allowed)

   Set the internal Eigen flag whether it is allowed to allocate memory on heap.

   If this flag is :obj:`False` and Eigen will try to allocate memory on heap, it will
   assert which causes :obj:`~exceptions.ValueError` to be raised by Ceygen. This is
   useful to ensure you use the most optimized code path. Defaults to :obj:`True`.
   Note: for this to work, Ceygen defines *EIGEN_RUNTIME_NO_MALLOC* preprocessor
   directive before including Eigen.

   See http://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html

.. function:: eigen_version()

   Return version of Eigen which Ceygen was compiled against as a tuple of three integers,
   for example (3, 1, 2).

   :rtype: :obj:`tuple` of 3 :obj:`ints <int>`

.. _`Eigen/Core`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers
.. _`fused type`: http://docs.cython.org/src/userguide/fusedtypes.html

.. include:: definitions.rst
