=======
PyBayes
=======

About
=====

PyBayes is a Python library in early stage of development. It's aim is to
create universal framework for Bayesian filtering and decision-making in
Python.

Licensing
---------

PyBayes is currently distributed under GNU GPL v2+ license. The authors of
PyBayes are however open to other licensing suggestions. (Do you want to use
PyBayes in e.g. BSD-licensed project? Ask!)

Obtaining PyBayes
=================

Development of PyBayes happens on http://github.com/strohel/PyBayes
The most fresh development source is available from there.

Installing PyBayes
==================

PyBayes uses standard Python distutils for building and installation. Follow
these steps in order to install PyBayes:

* download PyBayes, let's assume PyBayes-0.1.tar.gz filename
* unpack it:

  ``tar -xvf PyBayes-0.1.tar.gz``
* change directory into PyBayes source:

  ``cd Pybayes-0.1``
* build and install (either run as root or install to a user-writeable
  directory [#alternate_install]_):

  ``./setup.py install``

.. [#alternate_install] http://docs.python.org/install/#alternate-installation

**And you're done!** However, if you want PyBayes to be *considerably
faster*, please read following section.

Advanced installation options
-----------------------------

PyBayes can use Cython to build itself into binary Python
module. Such binary modules are transparent to Python in a way that Python
treats then as any other modules (you can ``import`` them as usual).
Interpreter overhead is avoided and many other optimisation options arise this
way.

In order to build optimised PyBayes, you'll additionally need:

* Cython_ Python to C compiler
* working C compiler (GCC on Unix-like systems, MinGW or Microsoft Visual C on
  Windows [#install_cython]_)
* NumPy_ numerical library for Python

.. _Cython: http://www.cython.org
.. [#install_cython] http://docs.cython.org/src/quickstart/install.html
.. _NumPy: http://numpy.scipy.org/

Proceed with following steps:

1. Install all required dependencies. They should be already available in your
   package manager if you use a modern Linux Distribution.

#. Unpack and install PyBayes as described above, you should see following
   messages during build:

      ``Notice: Cython found.``

      ``Notice: NumPy found.``

   * in order to be 100% sure that optimised build is used, you can add
     ``--use=cython=yes`` option to the ``./setup.py`` call. You can force pure
     Python mode even when Cython is installed, pass ``--use=cython=no``. By
     default, PyBayes auto-detects Cython and NumPy presence on system.
   * if you plan to profile code that uses optimised PyBayes, you may want to
     embed profiling information into PyBayes. This can be accomplished by
     passing ``--profile=yes`` to ``./setup.py``. The default is to omit
     profiling information in order to avoid performance penalties.


Testing PyBayes
===============

Once PyBayes is installed, you may want to run its tests in order to ensure
proper functionality. The ``examples`` directory contains ``run_tests.py`` and
``run_stresses.py`` scripts that execute all PyBayes tests and stress tests
respectively. Run these scripts with ``-h`` option to see usage.

   *Note: running tests from within source directory is discouraged and
   unsupported.*

For even greater convenience, ``examples/install_test_stress.py`` python
script can clear, build, install, test, stress both Python and Cython build in
one go. It is especially suitable for PyBayes hackers. Run
``install_test_stress.py -h`` to get usage information. Please be sure to add
``--clean`` or ``-c`` flag when you mix Python and Cython builds.
