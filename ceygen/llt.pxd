# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from dtype cimport nonint_dtype


cdef nonint_dtype[:, :] cholesky(nonint_dtype[:, :] x, nonint_dtype[:, :] out = *) nogil
