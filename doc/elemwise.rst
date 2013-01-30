=======================
Element-wise operations
=======================

This module implements some basic element-wise operations such as addition or division.

.. note:: This module exists only as a stop-gap until support for element-wise operations
   with memoryviews is implemented in Cython. It will be phased out once Cython with Mark
   Florisson's `array expressions`_ `pull request`_ merged is released.

.. module:: ceygen.elemwise

.. function:: add_vv(x, y[, out=None])

   Vector-vector addition: *x* + *y*

   :param x: first addend
   :type x: |vector|
   :param y: second addend
   :type y: |vector|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: subtract_vv(x, y[, out=None])

   Vector-vector subtraction: *x* - *y*

   :param x: minuend
   :type x: |vector|
   :param y: subtrahend
   :type y: |vector|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: add_mm(x, y[, out=None])

   Matrix-matrix addition: *x* + *y*

   :param x: first addend
   :type x: |matrix|
   :param y: second addend
   :type y: |matrix|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. function:: subtract_mm(x, y[, out=None])

   Matrix-matrix subtraction: *x* - *y*

   :param x: minuend
   :type x: |matrix|
   :param y: subtrahend
   :type y: |matrix|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. _`array expressions`: https://github.com/markflorisson88/minivect/raw/master/thesis/thesis.pdf
.. _`pull request`: https://github.com/cython/cython/pull/144

.. include:: definitions.rst
