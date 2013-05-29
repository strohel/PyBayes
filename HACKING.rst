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
* ``pybayes/wrappers/`` subpackage is special, it is the only package whose
  modules have different implementation for cython and for python. It is
  accomplished by .py (Python) and .pyx, .pxd (Cython) files.

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
recursion. External PyBayes clients can and should, however, only ``import
pybayes`` (and in future also ``import pybayes.subpackage``). From inside
PyBayes just import relevant pybayes modules, e.g. ``import pdfs``. Notable
exception from this rule is cimport, where (presumable due to a cython bug)
``from a.b cimport c`` sometimes doesn't work and one has to type ``from
pybayes.a.b cimport c``.

Imports in \*.py files should adhere to following rules:

* import first system modules (sys, io..), then external modules (matplotlib..)
  and then pybayes modules.
* **instead of** importing **numpy** directly use ``import wrappers._numpy as np``. 
  This ensures that fast C alternatives are used in compiled mode.
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

Releasing PyBayes
=================

Things to do when releasing new version (let it be **X.Y**) of PyBayes:

Before Tagging
--------------

1. Set fallback version to **X.Y** in `setup.py` (around line 15)
#. Set version to **X.Y** in `support/python-pybayes.spec`
#. Ensure `ChangeLog.rst` mentions all important changes
#. (Optional) update **short description** in `setup.py` **AND** `support/python-pybayes.spec`
#. (Optional) update **long description** `README.rst` **AND** `support/python-pybayes.spec`

Tagging
-------

1. Check everything, run tests and stresses for Python 2.7, 3.2 in both pure/Cython mode
#. git tag -s **vX.Y**
#. git-archive-all.sh --format tar --prefix PyBayes-**X.Y/** dist/PyBayes-**X.Y**.tar
#. gzip dist/PyBayes-**X.Y**.tar
#. ./setup.py register

(do not use `./setup.py upload`, it does not work as some files are not in MANIFEST etc.)

Publishing
----------

1. Upload PyBayes-**X.Y**.tar.gz to https://github.com/strohel/PyBayes/downloads and
   http://pypi.python.org/pypi/PyBayes
#. Build and upload docs: ``cd ../pybayes-doc && ./synchronize.sh``
#. Upload updated `python-pybayes.spec` file to
   https://build.opensuse.org/package/files?package=python-pybayes&project=home%3Astrohel
#. If **short description** of PyBayes changed, update it manually at following places:

   * https://github.com/strohel/PyBayes
#. If **long description** of PyBayes changed, update it manually at following places:

   * https://build.opensuse.org/package/show?package=python-pybayes&project=home%3Astrohel
   * http://scipy.org/Topical_Software
   * http://www.ohloh.net/p/pybayes
