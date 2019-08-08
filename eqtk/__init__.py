# -*- coding: utf-8 -*-

"""Top-level package for bebi103."""

# Force showing deprecation warnings.
import re
import warnings

# warnings.filterwarnings(
#     "always", category=DeprecationWarning, module="^{}\.".format(re.escape(__name__))
# )

from .eqtk import *
from .parsers import *

__author__ = """Justin Bois"""
__email__ = "bois@caltech.edu"
__version__ = "0.0.1"
