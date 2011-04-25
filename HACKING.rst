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
import relevant pybayes modules, e.g. ``import pdfs``. Notable exception from this rule is cimport,
where (presumable due to a cython bug) ``from a.b cimport c`` sometimes doesn't work and one has
to type ``from pybayes.a.b cimport c``.

Imports in \*.py files should adhere to following rules:

* import first system modules (sys, io..), then external modules (matplotlib..)
  and then pybayes modules.
* **instead of** importing **numpy** directly use ``import wrappers._numpy as np``. This ensures
  that fast C alternatives are used in compiled mode.
* **instead of** importing **numpy.linalg** directly use ``import wrappers._linalg as linalg``.
* use ``import module [as abbrev]`` or, for commonly used symbols ``from module import symbol``.
* ``from module import *`` shouldn't be used.

Following rules apply to \*.pxd (cython augmentation) files:

* no imports, just cimports.
* use same import styles as in associated .py file. (``from module cimport`` vs.
  ``cimport module [as abbrev]``)
* for numpy use ``cimport pybayes.wrappers._numpy as np``
* for numpy.linalg use ``cimport pybayes.wrappers._linalg as linalg``

*Above rules do not apply to* ``pybayes/tests``. *These modules are considered
external and should behave as a client script.*
