# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

import numpy as np

import functools
import os
import unittest as ut
# Python 2.6 compatibility
try:
    from unittest import skip, skipIf, skipUnless
except ImportError:
    skip = None

from ceygen.core import set_is_malloc_allowed, eigen_version


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


class NoMallocTestCase(CeygenTestCase):
    """
    CeygenTestCase sublass that by default runs with Eigen memory allocation disallowed.

    Use "with malloc_allowed:" context manager to suppress it temporarily
    """

    def setUp(self):
        # assure that no heap memory allocation in Eigen happens during this test class
        set_is_malloc_allowed(False)

    def tearDown(self):
        set_is_malloc_allowed(True)

class malloc_allowed:
    """Context manager to write with malloc_allowed: ... and be sure that after execution
    the state is reset to diwallowed no matter whether exception occured"""

    def __enter__(self):
        set_is_malloc_allowed(True)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        set_is_malloc_allowed(False)
        return False  # let the exceptions fall through


def _id(obj):
        return obj

if skip is None:
    def skip(reason):
        """Implementation of the @skip decorator from Python 2.7 for Python 2.6"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                pass
            origdoc = wrapper.__doc__ or wrapper.__name__
            wrapper.__doc__ = wrapper.__name__ + " [skipped '{0}']".format(reason)
            return wrapper
        return decorator

    def skipIf(condition, reason):
        """Implementation of the @skipIf decorator from Python 2.7 for Python 2.6"""
        if condition:
            return skip(reason)
        return _id

    def skipUnless(condition, reason):
        """Implementation of the @skipUnless decorator from Python 2.7 for Python 2.6"""
        if not condition:
            return skip(reason)
        return _id

def skipIfEigenOlderThan(world, major, minor):
    ev = eigen_version()
    reason = 'because this test only passes with Eigen >= {0}.{1}.{2}'.format(world, major, minor)
    reason += ', but tested Ceygen was compiled against {0}.{1}.{2}'.format(ev[0], ev[1], ev[2])
    for (expected, actual) in zip((world, major, minor), ev):
        if actual < expected:
            return skip(reason)
        if actual > expected:  # strictly greater, don't check more minor versions
            return _id
        # else check more minor version
    return _id  # actual == expected

def benchmark(func):
    """Decorator to mark functions as benchmarks so that they aren't run by default"""
    reason = 'because neither BENCHMARK or BENCHMARK_NUMPY environment variable is set'
    return skipUnless('BENCHMARK' in os.environ or 'BENCHMARK_NUMPY' in os.environ, reason)(func)
