# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport dtype


cpdef bint set_is_malloc_allowed(bint allowed) nogil
cpdef tuple eigen_version()

cdef dtype dot_vv(dtype[:] x, dtype[:] y) nogil except *
cdef dtype[:] dot_mv(dtype[:, :] x, dtype[:] y, dtype[:] out = *) nogil
cdef dtype[:] dot_vm(dtype[:] x, dtype[:, :] y, dtype[:] out = *) nogil
cdef dtype[:, :] dot_mm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
