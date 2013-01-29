# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport dtype

cdef dtype dotvv(dtype[:] x, dtype[:] y) nogil except *
cdef dtype[:] dotmv(dtype[:, :] x, dtype[:] y, dtype[:] out = *) nogil
cdef dtype[:] dotvm(dtype[:] x, dtype[:, :] y, dtype[:] out = *) nogil
cdef dtype[:, :] dotmm(dtype[:, :] x, dtype[:, :] y, dtype[:, :] out = *) nogil
