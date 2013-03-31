# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport matrix


cdef void cholesky_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        dtype *y) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    cdef MatrixMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.llt_matrixL())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] cholesky(dtype[:, :] x, dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    # we pass dummy scalar so that we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, <dtype *> 0,
            cholesky_worker, cholesky_worker, cholesky_worker,
            cholesky_worker, cholesky_worker, cholesky_worker,
            cholesky_worker, cholesky_worker, cholesky_worker)
    return out
