# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# When adding new type:
#  * add it to ctypedef fused dtype
#  * add it to get_format()
#  * update documentation in doc/core.rst
#  * rebuild Ceygen!

ctypedef fused dtype:
#    float
    double

cdef inline str get_format(dtype *dummy):
    """
    This function must return Type code for all data types in `dtype` as described in
    table at http://docs.python.org/library/array.html
    """
    #if dtype is float:
        #return 'f'
    if dtype is double:
        return 'd'
