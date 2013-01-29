# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for core"""

import numpy as np

from support import CeygenTestCase
cimport ceygen.elemwise as e

class TestElemwise(CeygenTestCase):

    def test_subtract(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])
        self.assertApproxEqual(e.subtract_mm(x, y), [[-2., 0., 2.]])
        self.assertApproxEqual(e.subtract_mm(y, x), [[2., 0., -2.]])
