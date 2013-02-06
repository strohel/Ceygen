# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from cython cimport view


cdef inline str get_format(dtype *dummy):
    """
    This function must return Type code for all data types in `dtype` as described in
    table at http://docs.python.org/library/array.html
    """
    #if dtype is float:
        #return 'f'
    if dtype is double:
        return 'd'

cdef dtype[:] vector(int size, dtype *like) with gil:
    return view.array(shape=(size,), itemsize=sizeof(dtype), format=get_format(like))

cdef dtype[:, :] matrix(int rows, int cols, dtype *like) with gil:
    return view.array(shape=(rows, cols), itemsize=sizeof(dtype), format=get_format(like))
