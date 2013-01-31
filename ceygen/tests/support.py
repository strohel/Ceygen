# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

import numpy as np

import functools
import unittest as ut
try:
    from unittest import skip
except ImportError:
    skip = None


class _AssertRaisesContext(object):
    """A context manager used to implement TestCase.assertRaises method, stolen from Python 2.7"""

    def __init__(self, expected, test_case):
        self.expected = expected
        self.failureException = test_case.failureException

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            try:
                exc_name = self.expected.__name__
            except AttributeError:
                exc_name = str(self.expected)
            raise self.failureException(
                "{0} not raised".format(exc_name))
        if not issubclass(exc_type, self.expected):
            # let unexpected exceptions pass through
            return False
        self.exception = exc_value # store for later retrieval
        return True


class CeygenTestCase(ut.TestCase):
    """Test case that adds some numeric assert functions"""

    def assertApproxEqual(self, X, Y):
        """Return true if X = Y to within machine precision

        Function for checking that different matrices from different
        computations are in some sense "equal" in the verification tests.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        fuzz = 1.0e-8
        self.assertEqual(X.ndim, Y.ndim)
        self.assertEqual(X.shape, Y.shape)

        if np.all(abs(X - Y) < fuzz):
            return
        else:
            self.fail("NumPy arrays {0} and {1} are not fuzzy equal (+- {2})".format(X, Y, fuzz))

    def assertRaises(self, excClass, callableObj=None, *args, **kwargs):
        """Python 2.6 doesn't support with assertRaises(Exception): syntax, steal it from 2.7"""
        context = _AssertRaisesContext(excClass, self)
        if callableObj is None:
            return context
        with context:
            callableObj(*args, **kwargs)


if skip is None:
    def skip(reason):
        """Implementation of the @skip decorator from Python 2.7 for Python 2.6"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                pass
            wrapper.__doc__ = wrapper.__name__ + " [skipped '{0}']".format(reason)
            return wrapper
        return decorator
