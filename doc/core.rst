==============
Core Functions
==============

This module provides basic linear algebra operations such as vector and matrix
products as provided by the <`Eigen/Core`_> include.

.. module:: ceygen.dtype

.. data:: dtype

   Cython `fused type`_, currently just a C double (Python :obj:`float`).

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
