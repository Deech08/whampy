# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys

__minimum_python_version__ = "3.6"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("whampy does not support Python < {}".format(__minimum_python_version__))


from .skySurvey import *
from .whampyTableMixin import *
