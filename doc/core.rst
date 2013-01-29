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


.. |scalar| replace:: :obj:`~ceygen.dtype.dtype`
.. |vector| replace:: :obj:`dtype[:] <ceygen.dtype.dtype>`
.. |matrix| replace:: :obj:`dtype[:, :] <ceygen.dtype.dtype>`
.. |out| replace:: memory view to write the result to. Specifying this optional argument
   means that Ceygen doesn't have to allocate memory for the result (allocating memory
   involves acquiring the GIL_ and calling many expensive Python functions). Once
   specified, it must must have correct dimensions to store the result of this operation
   (otherwise you get :obj:`~exceptions.ValueError`). **Warning**: don't repeat *x* or *y*
   here, it would give incorrect result without any error. Use (or implement) :-) in-place
   variant of this function instead.
.. |valueerror| replace:: :obj:`~exceptions.ValueError` if argument dimensions aren't
   appropriate for this operation or if arguments are otherwise invalid.
.. |typeerror| replace:: :obj:`~exceptions.TypeError` if you pass an argument that doesn't
   support buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
   and resort to :obj:`Python array <array>`, `Cython array`_ or a
   :obj:`NumPy array <numpy.ndarray>`.

.. _`Eigen/Core`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers
.. _`fused type`: http://docs.cython.org/src/userguide/fusedtypes.html
.. _`Cython memoryview`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Cython array`: http://docs.cython.org/src/userguide/memoryviews.html#cython-arrays
.. _`GIL`: http://docs.python.org/glossary.html#term-global-interpreter-lock
