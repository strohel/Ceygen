# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from eigen_cython cimport *
from dtype cimport vector, matrix


cpdef bint set_is_malloc_allowed(bint allowed) nogil:
    c_set_is_malloc_allowed(allowed)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype dot_vv(dtype[:] x, dtype[:] y) nogil except *:
    cdef VectorMap[dtype] x_map, y_map
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    return x_map.dot(y_map)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] dot_mv(dtype[:, :] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef MatrixMap[dtype] x_map
    cdef VectorMap[dtype] y_map, out_map
    if out is None:
        out = vector(x.shape[0], &y[0])
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.noalias_assign(x_map * y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] dot_vm(dtype[:] x, dtype[:, :] y, dtype[:] out = None) nogil:
    # we coul've just called dotmv(y.T, x, out), but y.T segfaults if y is uninitialized
    # memoryview, also we fear overhead in memoryview.transpose() and another function call
    cdef RowVectorMap[dtype] x_map, out_map
    cdef MatrixMap[dtype] y_map
    if out is None:
        out = vector(y.shape[1], &x[0])
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.noalias_assign(x_map * y_map)
    return out

cdef bint dot_mm_worker(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) nogil except False:
    cdef MatrixMapTODO[dtype, XMatrixContiguity] x
    cdef MatrixMapTODO[dtype, YMatrixContiguity] y
    cdef MatrixMapTODO[dtype, OMatrixContiguity] o
    x.init(x_data, x_shape, x_strides)
    y.init(y_data, y_shape, y_strides)
    o.init(o_data, o_shape, o_strides)
    o.noalias_assign(x * y)
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] dot_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    if out is None:
        out = matrix(x.shape[0], y.shape[1], &x[0, 0])
    dispatch_mmm(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
            &out[0, 0], out.shape, out.strides, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
            dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker, dot_mm_worker,
    )
    return out
