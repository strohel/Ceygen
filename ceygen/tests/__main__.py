# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Ceygen test-suite runner. Used when user calls `python -m ceygen.tests"""

import unittest as ut

from ceygen.tests import *


if __name__ == '__main__':
    ut.main()
