===================
PyBayes Development
===================

This document should serve as a reminder to me and other possible PyBayes
hackers about PyBayes coding style and conventions.

General Layout and Principles
=============================

PyBayes is developed with special dual-mode technique - it is both perfectly
valid pure Python library and optimised cython-built binary python module.

PyBayes modules are laid out with following rules:

* all modules go directly into ``pybayes/<module>.py`` (pure Python file) with
  cython augmentation file in ``pybayes/module.pxd``
* in future, bigger independent units can form subpackages
* ``pybayes/numpywrap.{pyx,py,pxd}`` are special, it is the only module that
  has different implementation for cython and for python.

Tests and Stress Tests
======================

All methods of all PyBayes classes should have a unit test. Suppose you have
a module ``pybayes/modname.py``, then unit tests for all classes in
``modname.py`` should go into ``pybayes/tests/test_modname.py``. You can also
write stress test (something that runs considerably longer than a test and
perhaps provides a simple benchmark) that would go into
``pybayes/tests/stress_modname.py``.

Imports and cimports
====================

**No internal module** can ``import pybayes``! That would result in an infinite
recursion. External PyBayes clients can and should, however, only ``import pybayes``
(and in future also ``import pybayes.subpackage``). From insibe PyBayes just
import relevant pybayes modules, e.g. ``import pdfs``.

Imports in \*.py files should adhere to following rules:

* import first system modules (sys, io..), then external modules (matplotlib..)
  and then pybayes modules
* **never import numpy directly**, import numpywrap instead (and perhaps extend
  symbols that numpywrap.{py,pyx} imports) [TODO: numpywrap handling should be
  refactored a bit]
* ``from module import symbol1, symbol2`` syntax is the preferred one
* ``from module import *`` is prohibited

Following rules apply to \*.pxd (cython augmentation) files:

* no imports, just cimports.
* use ``from module cimport symbol1, symbol2`` syntax
* ``from module cimport *`` is forbidden, only exception is ``from numpywrap
  cimport *``, which is mandatory

*Above rules do not apply to* ``pybayes/tests``. *These modules are considered
external and should behave as a client script.*
