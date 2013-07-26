# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dispatch cimport *
from dtype cimport vector, matrix


cpdef bint set_is_malloc_allowed(bint allowed) nogil:
    c_set_is_malloc_allowed(allowed)

cpdef tuple eigen_version():
    return (EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION)


cdef void dot_vv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o) nogil:
    cdef VectorMap[dtype, XVectorContiguity] x
    cdef VectorMap[dtype, YVectorContiguity] y
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o[0] = x.dot(y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype dot_vv(dtype[:] x, dtype[:] y) nogil except *:
    cdef VVSDispatcher[dtype] dispatcher
    cdef dtype out
    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, &out,
            dot_vv_worker, dot_vv_worker, dot_vv_worker, dot_vv_worker)
    return out


cdef void dot_mv_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YVectorContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    cdef VectorMap[dtype, YVectorContiguity] y
    cdef VectorMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.noalias_assign(x * y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] dot_mv(dtype[:, :] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef MVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(x.shape[0], &y[0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0], y.shape, y.strides,
            &out[0], out.shape, out.strides,
            dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker,
            dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker, dot_mv_worker)
    return out


cdef void dot_vm_worker(
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XVectorContiguity x_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OVectorContiguity o_dummy) nogil:
    cdef RowVectorMap[dtype, XVectorContiguity] x
    cdef MatrixMap[dtype, YMatrixContiguity] y
    cdef RowVectorMap[dtype, OVectorContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.noalias_assign(x * y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] dot_vm(dtype[:] x, dtype[:, :] y, dtype[:] out = None) nogil:
    # we coul've just called dotmv(y.T, x, out), but y.T segfaults if y is uninitialized
    # memoryview, also we fear overhead in memoryview.transpose() and another function call
    cdef MVVDispatcher[dtype] dispatcher
    if out is None:
        out = vector(y.shape[1], &x[0])
    # warning: we swap x and y here so that we can share dispather with dot_mv!
    dispatcher.run(&y[0, 0], y.shape, y.strides, &x[0], x.shape, x.strides,
            &out[0], out.shape, out.strides,
            dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker,
            dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker, dot_vm_worker)
    return out


cdef void dot_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil:
    cdef MatrixMap[dtype, XMatrixContiguity] x
    cdef MatrixMap[dtype, YMatrixContiguity] y
    cdef MatrixMap[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.noalias_assign(x * y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] dot_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MMMDispatcher[dtype] dispatcher
    if out is None:
        out = matrix(x.shape[0], y.shape[1], &x[0, 0])
    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker)
    return out
