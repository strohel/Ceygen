# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython
from cython cimport view

from eigen_cython cimport *
from dtype cimport get_format


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
    cdef VectorMap[dtype] y_map
    cdef VectorMap[dtype] out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
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
    cdef VectorMap[dtype] x_map
    cdef MatrixMap[dtype] y_map
    cdef VectorMap[dtype] out_map
    if out is None:
        with gil:
            out = view.array(shape=(y.shape[1],), itemsize=sizeof(dtype), format=get_format(&y[0, 0]))
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.noalias_assign(x_map.transpose() * y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] dot_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef MatrixMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],y.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.noalias_assign(x_map * y_map)
    return out
