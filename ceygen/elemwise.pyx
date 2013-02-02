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
cdef dtype[:] add_vs(dtype[:] x, dtype y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&y))
    x_map.init(&x[0], x.shape, x.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map + y)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] multiply_vs(dtype[:] x, dtype y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&y))
    x_map.init(&x[0], x.shape, x.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map * y)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] add_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0]))
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map + y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] subtract_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0]))
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map - y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] multiply_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0]))
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map * y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] divide_vv(dtype[:] x, dtype[:] y, dtype[:] out = None) nogil:
    cdef Array1DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0]))
    x_map.init(&x[0], x.shape, x.strides)
    y_map.init(&y[0], y.shape, y.strides)
    out_map.init(&out[0], out.shape, out.strides)
    out_map.assign(x_map / y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] add_ms(dtype[:, :] x, dtype y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&y))
    x_map.init(&x[0, 0], x.shape, x.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map + y)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] multiply_ms(dtype[:, :] x, dtype y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&y))
    x_map.init(&x[0, 0], x.shape, x.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map * y)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] add_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map + y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] subtract_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map - y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] multiply_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map * y_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:, :] divide_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = None) nogil:
    cdef Array2DMap[dtype] x_map, y_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    y_map.init(&y[0, 0], y.shape, y.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign(x_map / y_map)
    return out
