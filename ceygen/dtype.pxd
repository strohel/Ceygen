# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# When adding new type:
#  * add it to ctypedef fused dtype
#  * add it to get_format() in dtype.pyx
#  * update documentation in doc/core.rst
#  * rebuild Ceygen!

ctypedef fused dtype:
    char
    short
    int
    long
    float
    double

# some methods such as inv() cannot really work with non-integer types
ctypedef fused nonint_dtype:
    float
    double

cdef dtype[:] vector(int size, dtype *like) with gil
cdef dtype[:, :] matrix(int rows, int cols, dtype *like) with gil
