# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport matrix


cdef void inv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        dtype *y) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    cdef MatrixMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign_inverse(x)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] inv(dtype[:, :] x, dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    # we pass dummy scalar so that we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, <dtype *> 0,
            inv_worker, inv_worker, inv_worker,
            inv_worker, inv_worker, inv_worker,
            inv_worker, inv_worker, inv_worker)
    return out


cdef void iinv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    x.init(x_data, x_shape, x_strides)
    # why this doesn't exhibit aliasing problems?
    x.assign_inverse(x)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint iinv(dtype[:, :] x) nogil except False:
    cdef MSDispatcher[dtype] dispatcher
    # we pass dummy scalar so that we can reuse MSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, <dtype *> 0,
            iinv_worker, iinv_worker, iinv_worker)
    return True


cdef void det_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    x.init(x_data, x_shape, x_strides)
    o[0] = x.determinant()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype det(dtype[:, :] x) nogil except *:
    cdef MSDispatcher[dtype] dispatcher
    cdef dtype out
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out, det_worker, det_worker, det_worker)
    return out
