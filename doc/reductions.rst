==========
Reductions
==========

This module provides various reductions from matrices and vectors to scalars and from
matrices to to vectors.

.. module:: ceygen.reductions

.. function:: sum_v(x)

   Return sum of the vector *x*.

   :param x: vector to sum up
   :type x: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |scalar|

.. function:: sum_m(x)

   Return sum of the matrix *x*.

   :param x: matrix to sum up
   :type x: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |scalar|

.. function:: rowwise_sum(x[, out])

   Compute sum of the invidual rows of matrix *x*.

   :param x: matrix to sum up
   :type x: |matrix|
   :param out: |out|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: colwise_sum(x[, out])

   Compute sum of the invidual columns of matrix *x*.

   :param x: matrix to sum up
   :type x: |matrix|
   :param out: |out|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. include:: definitions.rst
