==================================
LU Decomposition-powered Functions
==================================

This module contains algebraic functions powered by the LU matrix decomposition (as
provided by the <`Eigen/LU`_> include), most notably matrix inverse and determinant.

.. module:: ceygen.lu

.. function:: inv(x[, out=None])

   Return matrix inverse computed using LU decomposition with partial pivoting. It is your
   responsibility to ensure that *x* is invertible, otherwise you get undefined result
   without any warning.

   :param x: matrix to invert
   :type x: |nonint_matrix|
   :param out: |out|
   :type out: |nonint_matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |nonint_matrix|

.. function:: iinv(x)

   Compte matrix inverse using LU decomposition with partial pivoting in-place. Equivalent
   to *x* = :obj:`inv(x) <inv>`, but without overhead. It is your responsibility to ensure
   that *x* is invertible, otherwise you get undefined result without any warning.

   :param x: matrix to invert in-place
   :type x: |nonint_matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :returns: |alwaystrue|

.. function:: det(x)

   Compute determinant of a square matrix *x* using LU decomposition.

   :param x: matrix whose determimant to compute
   :type x: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |scalar|

.. _`Eigen/LU`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers

.. include:: definitions.rst
