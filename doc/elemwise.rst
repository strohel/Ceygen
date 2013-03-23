=======================
Element-wise Operations
=======================

This module implements some basic element-wise operations such as addition or division.

Because aliasing is not a problem for element-wise operations, you can make the operations
in-place simply by repeating *x* or *y* in *out*. Following examples are therefore valid
and produce expected results::

   ceygen.elemwise.add_mm(x, y, x)
   ceygen.elemwise.multiply_vv(a, b, b)

.. note:: |arrayexprs|

.. module:: ceygen.elemwise

Vector-scalar Operations
========================

.. function:: add_vs(x, y[, out=None])

   Add scalar *y* to each coefficient of vector *x* and return the resulting vector.

   Note: there's no **subtract_vs**, just add opposite number.

   :param x: first addend (vector)
   :type x: |vector|
   :param y: second addend (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: multiply_vs(x, y[, out=None])

   Multiply each coefficient of vector *x* by scalar *y* and return the resulting vector.

   Note: there's no **divide_vs**, just multiply by inverse number.

   :param x: first factor (vector)
   :type x: |vector|
   :param y: second factor (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: power_vs(x, y[, out=None])

   Compute *y*-th power of each coefficient of vector *x*.

   :param x: base (vector)
   :type x: |vector|
   :param y: exponent (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

Vector-vector Operations
========================

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

.. function:: multiply_vv(x, y[, out=None])

   Vector-vector element-wise multiplication: *x* \* *y*

   :param x: first factor
   :type x: |vector|
   :param y: second factor
   :type y: |vector|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

.. function:: divide_vv(x, y[, out=None])

   Vector-vector element-wise division: *x* / *y*

   :param x: numerator
   :type x: |vector|
   :param y: denominator
   :type y: |vector|
   :param out: |out_elemwise|
   :type out: |vector|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |vector|

Matrix-scalar Operations
========================

.. function:: add_ms(x, y[, out=None])

   Add scalar *y* to each coefficient of matrix *x* and return the resulting matrix.

   Note: there's no **subtract_ms**, just add opposite number.

   :param x: first addend (matrix)
   :type x: |matrix|
   :param y: second addend (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. function:: multiply_ms(x, y[, out=None])

   Multiply each coefficient of matrix *x* by scalar *y* and return the resulting matrix.

   Note: there's no **divide_ms**, just multiply by inverse number.

   :param x: first factor (vector)
   :type x: |matrix|
   :param y: second factor (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. function:: power_ms(x, y[, out=None])

   Compute *y*-th power of each coefficient of matrix *x*.

   :param x: base (matrix)
   :type x: |matrix|
   :param y: exponent (scalar)
   :type y: |scalar|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

Matrix-matrix Operations
========================

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

.. function:: multiply_mm(x, y[, out=None])

   Matrix-matrix element-wise multiplication: *x* \* *y*

   :param x: first factor
   :type x: |matrix|
   :param y: second factor
   :type y: |matrix|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. function:: divide_mm(x, y[, out=None])

   Matrix-matrix element-wise division: *x* / *y*

   :param x: numerator
   :type x: |matrix|
   :param y: denominator
   :type y: |matrix|
   :param out: |out_elemwise|
   :type out: |matrix|
   :raises: |valueerror|
   :raises: |typeerror|
   :rtype: |matrix|

.. include:: definitions.rst
