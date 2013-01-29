==============
Core Functions
==============

This module provides basic linear algebra operations such as vector and matrix
products as provided by the <`Eigen/Core`_> include.

.. module:: ceygen.core

.. data:: dtype

   Cython `fused type`_, currently just a C double (Python :obj:`float`).

.. function:: dotvv(x, y)

   Vector-vector dot product, returns a scalar of appropriate type.

   :param x: first factor
   :type x: :obj:`dtype[:] <dtype>`
   :param y: second factor
   :type y: :obj:`dtype[:] <dtype>`
   :raises: :obj:`~exceptions.ValueError` if argument dimensions don't match or are
            otherwise invalid.
   :raises: :obj:`~exceptions.TypeError` if you pass an argument that doesn't support
            buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
            and resort to :obj:`Python array <array>`, `Cython array`_ or a
            :obj:`NumPy array <numpy.ndarray>`.
   :rtype: :obj:`dtype`

.. function:: dotmv(x, y[, out=None])

   Matrix-(column) vector product, returns a vector of appropriate type.

   :param x: first factor (matrix)
   :type x: :obj:`dtype[:, :] <dtype>`
   :param y: second factor (vector)
   :type y: :obj:`dtype[:] <dtype>`
   :param out: memory view to write the result to. Specifying this optional argument means
               that Ceygen doesn't have to allocate memory for the result (allocating memory
               involves acquiring the GIL_ and calling many expensive Python functions).
               Once specified, it must must have correct dimensions (number of rows of *x*).
   :type out: :obj:`dtype[:] <dtype>`
   :raises: :obj:`~exceptions.ValueError` if argument dimensions don't match or are
            otherwise invalid.
   :raises: :obj:`~exceptions.TypeError` if you pass an argument that doesn't support
            buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
            and resort to :obj:`Python array <array>`, `Cython array`_ or a
            :obj:`NumPy array <numpy.ndarray>`.
   :rtype: :obj:`dtype[:] <dtype>`

.. function:: dotvm(x, y[, out=None])

   (Row) vector-matrix product, returns a vector of appropriate type. This is equivalent
   to dotvm(*y*.T, *x*) because there's no distinction between row and column vectors in
   Cython memoryviews, but calling this function directly may incur slightly less
   overhead.

   :param x: first factor (vector)
   :type x: :obj:`dtype[:] <dtype>`
   :param y: second factor (matrix)
   :type y: :obj:`dtype[:, :] <dtype>`
   :param out: memory view to write the result to. Specifying this optional argument means
               that Ceygen doesn't have to allocate memory for the result (allocating memory
               involves acquiring the GIL_ and calling many expensive Python functions).
               Once specified, it must must have correct dimensions (number of columns
               of *y*).
   :type out: :obj:`dtype[:] <dtype>`
   :raises: :obj:`~exceptions.ValueError` if argument dimensions don't match or are
            otherwise invalid.
   :raises: :obj:`~exceptions.TypeError` if you pass an argument that doesn't support
            buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
            and resort to :obj:`Python array <array>`, `Cython array`_ or a
            :obj:`NumPy array <numpy.ndarray>`.
   :rtype: :obj:`dtype[:] <dtype>`

.. function:: dotmm(x, y[, out=None])

   Matrix-matrix product, returns a matrix of appropriate type and dimensions. You may of
   course use this function to multiply matrices that are in fact vectors, you just need
   to pay attention to column-vector vs. row-vector distinction this time.

   :param x: first factor
   :type x: :obj:`dtype[:, :] <dtype>`
   :param y: second factor
   :type y: :obj:`dtype[:, :] <dtype>`
   :param out: memory view to write the result to. Specifying this optional argument means
               that Ceygen doesn't have to allocate memory for the result (allocating memory
               involves acquiring the GIL_ and calling many expensive Python functions).
               Once specified, it must must have correct dimensions (number of rows
               of *x* x number of columns of *y*).
   :type out: :obj:`dtype[:] <dtype>`
   :raises: :obj:`~exceptions.ValueError` if argument dimensions don't match or are
            otherwise invalid.
   :raises: :obj:`~exceptions.TypeError` if you pass an argument that doesn't support
            buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
            and resort to :obj:`Python array <array>`, `Cython array`_ or a
            :obj:`NumPy array <numpy.ndarray>`.
   :rtype: :obj:`dtype[:] <dtype>`

.. _`Eigen/Core`: http://eigen.tuxfamily.org/dox/QuickRefPage.html#QuickRef_Headers
.. _`fused type`: http://docs.cython.org/src/userguide/fusedtypes.html
.. _`Cython memoryview`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Cython array`: http://docs.cython.org/src/userguide/memoryviews.html#cython-arrays
.. _GIL: http://docs.python.org/glossary.html#term-global-interpreter-lock
