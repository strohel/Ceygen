.. Definitions to be shared by other documentation documents.

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

.. _`Cython memoryview`: http://docs.cython.org/src/userguide/memoryviews.html
.. _`Cython array`: http://docs.cython.org/src/userguide/memoryviews.html#cython-arrays
.. _`GIL`: http://docs.python.org/glossary.html#term-global-interpreter-lock
