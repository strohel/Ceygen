# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport dtype


cdef dtype[:, :] inv(dtype[:, :] x, dtype[:, :] out = *) nogil
cdef bint iinv(dtype[:, :] x) nogil except False

cdef dtype det(dtype[:, :] x) nogil except *
