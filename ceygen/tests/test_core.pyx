# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for core"""

import numpy as np

import unittest as ut

cimport ceygen.core as c
import ceygen.core as c

class TestWrappersNumpy(ut.TestCase):

    def test_dotvv(self):
        cdef double[:] x = np.array([1., 2., 3.])
        cdef double[:] y = np.array([4., 5., 6.])
        self.assertAlmostEqual(c.dotvv(x, y), 32.)
