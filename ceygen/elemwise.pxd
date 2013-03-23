# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport dtype


cdef dtype[:] add_vs(dtype[:] x, dtype y, dtype[:] out = *) nogil
cdef dtype[:] multiply_vs(dtype[:] x, dtype y, dtype[:] out = *) nogil
cdef dtype[:] power_vs(dtype[:] x, dtype y, dtype[:] out = *) nogil

cdef dtype[:] add_vv(dtype[:] x, dtype[:] y, dtype[:] out = *) nogil
cdef dtype[:] subtract_vv(dtype[:] x, dtype[:] y, dtype[:] out = *) nogil
cdef dtype[:] multiply_vv(dtype[:] x, dtype[:] y, dtype[:] out = *) nogil
cdef dtype[:] divide_vv(dtype[:] x, dtype[:] y, dtype[:] out = *) nogil

cdef dtype[:, :] add_ms(dtype[:, :] x, dtype y, dtype[:, :] out = *) nogil
cdef dtype[:, :] multiply_ms(dtype[:, :] x, dtype y, dtype[:, :] out = *) nogil
cdef dtype[:, :] power_ms(dtype[:, :] x, dtype y, dtype[:, :] out = *) nogil

cdef dtype[:, :] add_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
cdef dtype[:, :] subtract_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
cdef dtype[:, :] multiply_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
cdef dtype[:, :] divide_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
