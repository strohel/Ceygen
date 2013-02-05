==============
Core Functions
==============

This module provides basic linear algebra operations such as vector and matrix
products as provided by the <`Eigen/Core`_> include.

.. module:: ceygen.dtype

.. data:: dtype

   Cython `fused type`_, currently just a C double (Python :obj:`float`).

.. module:: ceygen.core

.. function:: set_is_malloc_allowed(allowed)

   Set the internal Eigen flag whether it is allowed to allocate memory on heap.

   If this flag is :obj:`False` and Eigen will try to allocate memory on heap, it will
   assert which causes :obj:`~exceptions.ValueError` to be raised by Ceygen. This is
   useful to ensure you use the most optimized code path. Defaults to :obj:`True`.
   Note: for this to work, Ceygen defines *EIGEN_RUNTIME_NO_MALLOC* preprocessor
   directive before including Eigen.

   See http://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html

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
   Eigen; speed gains are noticeable, but not dramatic. No special markup is needed to
   trigger this. See also :func:`set_is_malloc_allowed`.

   :param x: first factor
   :type x: |matrix|
   :param y: second factor
   :type y: |matrix|
   :param out: |out|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. _`Eigen/Core`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers
.. _`fused type`: http://docs.cython.org/src/userguide/fusedtypes.html

.. include:: definitions.rst
