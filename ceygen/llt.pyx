# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport matrix


cdef void cholesky_worker(
        nonint_dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        nonint_dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        nonint_dtype *y) nogil:
    cdef MatrixMap[nonint_dtype, XMatrixContiguity] x
    cdef MatrixMap[nonint_dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.llt_matrixL())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef nonint_dtype[:, :] cholesky(nonint_dtype[:, :] x, nonint_dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[nonint_dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    # we pass dummy scalar so that we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, <nonint_dtype *> 0,
            cholesky_worker, cholesky_worker, cholesky_worker,
            cholesky_worker, cholesky_worker, cholesky_worker,
            cholesky_worker, cholesky_worker, cholesky_worker)
    return out
