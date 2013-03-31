.. Definitions to be shared by other documentation documents.

.. |scalar| replace:: :obj:`~ceygen.dtype.dtype`
.. |nonint_scalar| replace:: :obj:`~ceygen.dtype.nonint_dtype`
.. |vector| replace:: :obj:`dtype[:] <ceygen.dtype.dtype>`
.. |nonint_vector| replace:: :obj:`nonint_dtype[:] <ceygen.dtype.nonint_dtype>`
.. |matrix| replace:: :obj:`dtype[:, :] <ceygen.dtype.dtype>`
.. |nonint_matrix| replace:: :obj:`nonint_dtype[:, :] <ceygen.dtype.nonint_dtype>`
.. |out| replace:: memory view to write the result to. Specifying this optional argument
   means that Ceygen doesn't have to allocate memory for the result (allocating memory
   involves acquiring the GIL_ and calling many expensive Python functions). Once
   specified, it must must have correct dimensions to store the result of this operation
   (otherwise you get :obj:`~exceptions.ValueError`); the same *out* instance will be also
   returned. **Warning**: don't repeat *x* (or *y*) here, it `would give incorrect result
   without any error`_. Perhaps there's an in-place variant instead?
.. |out_elemwise| replace:: memory view to write the result to. Specifying this optional
   argument means that Ceygen doesn't have to allocate memory for the result (allocating
   memory involves acquiring the GIL_ and calling many expensive Python functions). Once
   specified, it must must have correct dimensions to store the result of this operation
   (otherwise you get :obj:`~exceptions.ValueError`); the same *out* instance will be also
   returned. *As an exception from the general rule*, you **may repeat** *x* (or *y*) here
   `for this element-wise operation`_.
.. |valueerror| replace:: :obj:`~exceptions.ValueError` if argument dimensions aren't
   appropriate for this operation or if arguments are otherwise invalid.
.. |typeerror| replace:: :obj:`~exceptions.TypeError` if you pass an argument that doesn't
   support buffer interface (e.g. a plain list). Use preferrably a `Cython memoryview`_
   and resort to :obj:`Python array <array>`, `Cython array`_ or a
   :obj:`NumPy array <numpy.ndarray>`.
.. |alwaystrue| replace:: Always :obj:`True` to allow fast exception propagation.
.. |arrayexprs| replace:: This module exists only as a stop-gap until support for
   element-wise operations with memoryviews are implemented in Cython. It will be phased
   out once Cython with Mark Florisson's `array expressions`_ `pull request`_ merged is
   released.

.. _`would give incorrect result without any error`: http://eigen.tuxfamily.org/dox/TopicAliasing.html
.. _`for this element-wise operation`: http://eigen.tuxfamily.org/dox/TopicAliasing.html
.. _`Cython memoryview`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Cython array`: http://docs.cython.org/src/userguide/memoryviews.html#cython-arrays
.. _`GIL`: http://docs.python.org/glossary.html#term-global-interpreter-lock
.. _`array expressions`: https://github.com/markflorisson88/minivect/raw/master/thesis/thesis.pdf
.. _`pull request`: https://github.com/cython/cython/pull/144
