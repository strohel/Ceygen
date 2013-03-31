========================================
Cholesky Decomposition-powered Functions
========================================

This module contains algebraic functions powered by the Cholesky matrix decomposition (as
provided by the <`Eigen/Cholesky`_> include).

.. module:: ceygen.llt

.. function:: cholesky(x[, out=None])

   Compute Cholesky decomposition of matrix *x* (which must be square, Hermitian and
   positive-definite) so that *x* = *out* \* *out*.H (*out*.H being conjugate transpose of
   *out*)

   :param x: matrix to decompose
   :type x: |nonint_matrix|
   :param out: |out|
   :type out: |nonint_matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |nonint_matrix|

.. _`Eigen/Cholesky`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers

.. include:: definitions.rst
