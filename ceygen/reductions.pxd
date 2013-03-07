# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport dtype


cdef dtype sum_v(dtype[:] x) nogil except *
cdef dtype sum_m(dtype[:, :] x) nogil except *
cdef dtype[:] rowwise_sum(dtype[:, :] x, dtype[:] out = *) nogil
cdef dtype[:] colwise_sum(dtype[:, :] x, dtype[:] out = *) nogil
