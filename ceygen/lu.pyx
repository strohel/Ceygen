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
cdef dtype[:, :] inv(dtype[:, :] x, dtype[:, :] out = None) nogil:
    cdef MatrixMap[dtype] x_map, out_map
    if out is None:
        with gil:
            out = view.array(shape=(x.shape[0],x.shape[1]), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape, x.strides)
    out_map.init(&out[0, 0], out.shape, out.strides)
    out_map.assign_inverse(x_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint iinv(dtype[:, :] x) nogil except False:
    cdef MatrixMap[dtype] x_map
    x_map.init(&x[0, 0], x.shape, x.strides)
    # how comes this doesn't exhibit aliasing problems?
    x_map.assign_inverse(x_map)
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype det(dtype[:, :] x) nogil except *:
    cdef MatrixMap[dtype] x_map
    x_map.init(&x[0, 0], x.shape, x.strides)
    return x_map.determinant()
