# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport vector, matrix


cdef void add_vs_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy,
        dtype *y) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x + y[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] add_vs(dtype[:] x, dtype y, dtype[:] out = None) nogil:
    cdef VVSDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &y)
    # y, out is swapped here so that we can share VVSDispatcher
    dispatcher.run(&x[0], x.shape, x.strides, &out[0], out.shape, out.strides, &y,
            add_vs_worker, add_vs_worker, add_vs_worker, add_vs_worker)
    return out


cdef void multiply_vs_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy,
        dtype *y) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x * y[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] multiply_vs(dtype[:] x, dtype y, dtype[:] out = None) nogil:
    cdef VVSDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &y)
    # y, out is swapped here so that we can share VVSDispatcher
    dispatcher.run(&x[0], x.shape, x.strides, &out[0], out.shape, out.strides, &y,
            multiply_vs_worker, multiply_vs_worker, multiply_vs_worker, multiply_vs_worker)
    return out


cdef void power_vs_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy,
        dtype *y) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.pow(y[0]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] power_vs(dtype[:] x, dtype y, dtype[:] out = None) nogil:
    cdef VVSDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &y)
    # y, out is swapped here so that we can share VVSDispatcher
    dispatcher.run(&x[0], x.shape, x.strides, &out[0], out.shape, out.strides, &y,
            power_vs_worker, power_vs_worker, power_vs_worker, power_vs_worker)
    return out


cdef void add_vv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, YVectorContiguity] y
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x + y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] add_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef VVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &x[0])
    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, &out[0], out.shape, out.strides,
            add_vv_worker, add_vv_worker, add_vv_worker, add_vv_worker,
            add_vv_worker, add_vv_worker, add_vv_worker, add_vv_worker)
    return out


cdef void subtract_vv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, YVectorContiguity] y
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x - y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] subtract_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef VVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &x[0])
    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, &out[0], out.shape, out.strides,
            subtract_vv_worker, subtract_vv_worker, subtract_vv_worker, subtract_vv_worker,
            subtract_vv_worker, subtract_vv_worker, subtract_vv_worker, subtract_vv_worker)
    return out


cdef void multiply_vv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, YVectorContiguity] y
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x * y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] multiply_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef VVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &x[0])
    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, &out[0], out.shape, out.strides,
            multiply_vv_worker, multiply_vv_worker, multiply_vv_worker, multiply_vv_worker,
            multiply_vv_worker, multiply_vv_worker, multiply_vv_worker, multiply_vv_worker)
    return out


cdef void divide_vv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef Array1DMap[dtype, XVectorContiguity] x
    cdef Array1DMap[dtype, YVectorContiguity] y
    cdef Array1DMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x / y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] divide_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef VVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &x[0])
    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, &out[0], out.shape, out.strides,
            divide_vv_worker, divide_vv_worker, divide_vv_worker, divide_vv_worker,
            divide_vv_worker, divide_vv_worker, divide_vv_worker, divide_vv_worker)
    return out


cdef void add_ms_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        dtype *y) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x + y[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] add_ms(dtype[:, :] x, dtype y, dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &y)
    # we swap out and y so tat we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, &y,
            add_ms_worker, add_ms_worker, add_ms_worker,
            add_ms_worker, add_ms_worker, add_ms_worker,
            add_ms_worker, add_ms_worker, add_ms_worker)
    return out


cdef void multiply_ms_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        dtype *y) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x * y[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] multiply_ms(dtype[:, :] x, dtype y, dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &y)
    # we swap out and y so tat we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, &y,
            multiply_ms_worker, multiply_ms_worker, multiply_ms_worker,
            multiply_ms_worker, multiply_ms_worker, multiply_ms_worker,
            multiply_ms_worker, multiply_ms_worker, multiply_ms_worker)
    return out


cdef void power_ms_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy,
        dtype *y) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x.pow(y[0]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] power_ms(dtype[:, :] x, dtype y, dtype[:, :] out = None) nogil:
    cdef MMSDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &y)
    # we swap out and y so tat we can reuse MMSDispatcher
    dispatcher.run(&x[0, 0], x.shape, x.strides, &out[0, 0], out.shape, out.strides, &y,
            power_ms_worker, power_ms_worker, power_ms_worker,
            power_ms_worker, power_ms_worker, power_ms_worker,
            power_ms_worker, power_ms_worker, power_ms_worker)
    return out


cdef void add_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, YMatrixContiguity] y
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x + y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] add_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MMMDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, add_mm_worker, add_mm_worker, add_mm_worker,
            add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker,
            add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker,
            add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker,
            add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker, add_mm_worker)
    return out


cdef void subtract_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, YMatrixContiguity] y
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x - y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] subtract_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MMMDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker,
            subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker,
            subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker,
            subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker,
            subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker, subtract_mm_worker)
    return out


cdef void multiply_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, YMatrixContiguity] y
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x * y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] multiply_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MMMDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker,
            multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker,
            multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker,
            multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker,
            multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker, multiply_mm_worker)
    return out


cdef void divide_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil:
    cdef Array2DMap[dtype, XMatrixContiguity] x
    cdef Array2DMap[dtype, YMatrixContiguity] y
    cdef Array2DMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.assign(x / y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] divide_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MMMDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], x.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, divide_mm_worker, divide_mm_worker, divide_mm_worker,
            divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker,
            divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker,
            divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker,
            divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker, divide_mm_worker)
    return out
