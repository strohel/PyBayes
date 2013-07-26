=======
PyBayes
=======

About
=====

PyBayes is an object-oriented Python library for recursive Bayesian
estimation (Bayesian filtering) that is convenient to use. Already implemented are
Kalman filter, particle filter and marginalized particle filter, all built atop of
a light framework of probability density functions. PyBayes can optionally use Cython
for large speed gains (Cython build can be several times faster in some situations).

PyBayes is tested with Python 2.7, 3.2 and 3.3 (using 2to3). Future plans include
more specialised variants of Kalman/particle filters and speed optimisations.

PyBayes is being developed by Matěj Laitl, feel free to send me a mail to matej at laitl dot cz.
See ChangeLog.rst file to review a list of most important changes in recent versions.

Automatically generated **documentation** can be found at
http://strohel.github.com/PyBayes-doc/

Licensing
---------

PyBayes is currently distributed under GNU GPL v2+ license. The authors of
PyBayes are however open to other licensing suggestions. (Do you want to use
PyBayes in e.g. BSD-licensed project? Ask!)

Obtaining PyBayes
=================

PyBayes releases can be found in .tar.gz format at github_ or PyPI_. Binary packages for
CentOS, Debian, Fedora, RHEL, OpenSUSE and Ubuntu can be downloaded from the
`OpenSUSE Build Service`_; these packages are fast Cython builds. (with no requirement to
install Cython for building)

.. _github: https://github.com/strohel/PyBayes/downloads
.. _PyPI: http://pypi.python.org/pypi/PyBayes
.. _`OpenSUSE Build Service`: https://build.opensuse.org/package/show?package=python-pybayes&project=home%3Astrohel

Development of PyBayes happens on http://github.com/strohel/PyBayes using git VCS
and the most fresh development sources can be obtained using git::

   # cd path/to/projects
   # git clone git://github.com/strohel/PyBayes.git

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

PyBayes can use Cython to build itself into binary Python module. Such binary modules are
transparent to Python in a way that Python treats then as any other modules (you can
``import`` them as usual). Interpreter overhead is avoided and many other optimisation
options arise this way.

In order to build optimised PyBayes, you'll additionally need:

* Cython_ Python to C compiler, version **0.18** or newer is recommended
* working C compiler (GCC on Unix-like systems, MinGW or Microsoft Visual C on
  Windows [#install_cython]_)
* NumPy_ numerical library for Python, version 1.5 or greater (NumPy is needed
  also in Python build, but older version suffice in that case)
* Ceygen_ Python package 0.3 or greater installed to a standard location
* On some Debian-based Linux distributions (Ubuntu) you'll need python-dev
  package that contains ``Python.h`` file that is needed by PyBayes

.. _Cython: http://www.cython.org/
.. [#install_cython] http://docs.cython.org/src/quickstart/install.html
.. _NumPy: http://numpy.scipy.org/
.. _Ceygen: https://github.com/strohel/Ceygen

Proceed with following steps:

1. Install all required dependencies. They should be already available in your
   package manager if you use a modern Linux Distribution.

#. Unpack and install PyBayes as described above, you should see following
   message during build:

      ``Cython and NumPy found, enabling optimised Cython build.``

   * in order to be 100% sure that optimised build is used, you can add
     ``--use-cython=yes`` option to the ``./setup.py`` call. You can force pure
     Python mode even when Cython is installed, pass ``--use-cython=no``. By
     default, PyBayes auto-detects Cython and NumPy presence on system.
   * if you plan to profile code that uses optimised PyBayes, you may want to
     embed profiling information into PyBayes. This can be accomplished by
     passing ``--profile=yes`` to ``./setup.py``. The default is to omit
     profiling information in order to avoid performance penalties.
   * all standard and custom build parameters can be listed using ``./setup.py --help``

The best results performance-wise are achieved when also your code that uses or extends PyBayes is
compiled by Cython and uses static typing where appropriate. Remember to
``cimport pybayes[.something]`` everytime you ``import pybayes[.something]`` so that fast Cython
calling convention is used.

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

Testing
=======

PyBayes comes with a comprehensive test and stress-suite that can and should be used to verify that
your PyBayes build works as expected.

Since version 0.4, testing is integrated into the `setup.py` script and can be run without
installing PyBayes. In order to run PyBayes test-suite, simply issue ``./setup.py test`` from within
the source directory. To run tests during installation procedure, simply install like this:
``./setup.py build test install``. With this command, failing tests prevent installation.

If you want to test your already installed PyBayes instance, simply issue
``python -m pybayes.tests`` anytime, anywhere. :-)

Stress-testing
--------------

Stress-testing works similarly to unit testing since version 0.4, run it using ``./setup.py
stress`` from the source directory. Already installed PyBayes can be stress-tested using
``python -m pybayes.stresses``.
