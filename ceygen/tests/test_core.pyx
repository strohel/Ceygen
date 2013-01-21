# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for core"""

import numpy as np

import unittest as ut

cimport ceygen.core as c

class TestCore(ut.TestCase):

    def test_dotvv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([4., 5., 6.])
        self.assertAlmostEqual(c.dotvv(x_np, y_np), 32.)
        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertAlmostEqual(c.dotvv(x, y), 32.)

    def test_dotvv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([4., 5.])
        z = np.array([[1., 2.], [3., 4.]])
        def dotvv(x, y):
            # wrap up because c.dotvv is cython-only (not callable from Python)
            return c.dotvv(x, y)

        self.assertRaises(StandardError, dotvv, x, y)
        self.assertRaises(StandardError, dotvv, x, z)

    def test_dotvv_none(self):
        x = np.array([1., 2., 3.])
        def dotvv(x, y):
            return c.dotvv(x, y)
        self.assertRaises(StandardError, dotvv, x, None)
        self.assertRaises(StandardError, dotvv, x, "Hello")
        self.assertRaises(StandardError, dotvv, None, x)
        self.assertRaises(StandardError, dotvv, "Hello", x)
