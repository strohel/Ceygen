=======================
Element-wise operations
=======================

This module implements some basic element-wise operations such as addition or division.

.. note:: This module exists only as a stop-gap until support for element-wise operations
   with memoryviews is implemented in Cython. It will be phased out once Cython with Mark
   Florisson's `array expressions`_ `pull request`_ merged is released.

.. module:: ceygen.elemwise

.. function:: add_mm(x, y[, out=None])

   Matrix-matrix addition: *x* + *y*

   :param x: first addend
   :type x: |matrix|
   :param y: second addend
   :type y: |matrix|
   :param out: |out|
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
   :param out: |out|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. _`array expressions`: https://github.com/markflorisson88/minivect/raw/master/thesis/thesis.pdf
.. _`pull request`: https://github.com/cython/cython/pull/144

.. include:: definitions.rst
