=======
PyBayes
=======

About
=====

A long-term goal of PyBayes is to be the preferred python library for
implementing Bayesian filtering (recursive estimation) and decision-making
systems.

Already done are classes for both basic static and conditional probability
densities ([c]pdfs) and a special cpdf representing a chain rule. Particle
filter that uses cpdfs extensively is implemented, Kalman filter is also present.

Future plans include more specialised variants of Kalman/particle filters and
speed optimisations. The project is also interesting technically as it is
dual-mode: can be be used without cython at all or compiled to gain more speed
- with nearly no code duplication.

Automatically generated **documentation** can be found at
http://strohel.github.com/PyBayes-doc/

Licensing
---------

PyBayes is currently distributed under GNU GPL v2+ license. The authors of
PyBayes are however open to other licensing suggestions. (Do you want to use
PyBayes in e.g. BSD-licensed project? Ask!)

Obtaining PyBayes
=================

Development of PyBayes happens on http://github.com/strohel/PyBayes using git VCS
and the most fresh development sources can be obtained using git. It should be noted that
PyBayes uses git submodule to bundle Tokyo library, so the proper way to clone
PyBayes repository would be::

   # cd path/to/projects
   # git clone git://github.com/strohel/PyBayes.git
   Cloning into PyBayes...
   (...)
   # cd PyBayes
   # git submodule update --init
   Submodule 'tokyo' (git://github.com/strohel/Tokyo.git) registered for path 'tokyo'
   Cloning into tokyo...
   (...)
   Submodule path 'tokyo': checked out '896d046b62cf50faf7faa7e58a8705fb2f22f19a'

When updating your repository (using ``git pull``), git should inform you that
some submodules have became outdated. In that case you should issue
``git submodule update``.


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
faster*, please read the following section.

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
* NumPy_ numerical library for Python, version 1.5 or greater (NumPy is needed
  also in Python build, but older version suffice in that case)
* On some Debian-based Linux distributions (Ubuntu) you'll need python-dev
  package that contains ``Python.h`` file that is needed by PyBayes

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

Building Documentation
----------------------

*There is no need to build documentation yourself, an online version is at*
http://strohel.github.com/PyBayes-doc/

PyBayes uses Sphinx_ to prepare documentation, version 1.0 or greater is required.
The documentation is built separately from the python build process.
In order to build it, change directory to `doc/` under PyBayes source directory
(``cd [path_to_pybayes]/doc``) and issue ``make`` command. This will present you
with a list of available documentation formats. To generate html documentation,
for example, run ``make html`` and then point your browser to
`[path_to_pybayes]/doc/_build/html/index.html`.

   PyBayes docs contain many mathematical expressions; Sphinx_ can use LaTeX_ to
   embed them as images into resulting HTML pages. Be sure to have LaTeX-enabled
   Sphinx if you want to see such nice things.

.. _Sphinx: http://sphinx.pocoo.org/
.. _LaTeX: http://www.latex-project.org/

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
