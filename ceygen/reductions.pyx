# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport vector


cdef void sum_v_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *o) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    x.init(x_data, x_shape, x_strides)
    o[0] = x.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype sum_v(dtype[:] x) nogil except *:
    cdef VSDispatcher[dtype] dispatcher
    cdef dtype out
    dispatcher.run(&x[0], x.shape, x.strides, &out, sum_v_worker, sum_v_worker)
    return out


cdef void sum_m_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    x.init(x_data, x_shape, x_strides)
    o[0] = x.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype sum_m(dtype[:, :] x) nogil except *:
    cdef MSDispatcher[dtype] dispatcher
    cdef dtype out
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out, sum_m_worker, sum_m_worker, sum_m_worker)
    return out


cdef void rowwise_sum_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.rowwise_sum())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] rowwise_sum(dtype[:, :] x, dtype[:] out = None) nogil:
    cdef MVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0], out.shape, out.strides,
            rowwise_sum_worker, rowwise_sum_worker, rowwise_sum_worker,
            rowwise_sum_worker, rowwise_sum_worker, rowwise_sum_worker)
    return out


cdef void colwise_sum_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.colwise_sum())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] colwise_sum(dtype[:, :] x, dtype[:] out = None) nogil:
    cdef MVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0], out.shape, out.strides,
            colwise_sum_worker, colwise_sum_worker, colwise_sum_worker,
            colwise_sum_worker, colwise_sum_worker, colwise_sum_worker)
    return out
