#
# spec file for package python-pybayes
#
# Copyright (c) 2011 SUSE LINUX Products GmbH, Nuernberg, Germany.
#
# All modifications and additions to the file contributed by third parties
# remain the property of their copyright owners, unless otherwise agreed
# upon. The license for this file, and modifications and additions to the
# file, is the same license as for the pristine package itself (unless the
# license for the pristine package is not an Open Source License, in which
# case the license is the MIT License). An "Open Source License" is a
# license that conforms to the Open Source Definition (Version 1.9)
# published by the Open Source Initiative.

# Please submit bugfixes or comments via http://bugs.opensuse.org/
#


Name:           python-pybayes
Version:        0.3.9999
Release:        0
Url:            https://github.com/strohel/PyBayes
Summary:        Python library for recursive Bayesian estimation (Bayesian filtering)
License:        GPL-2.0
Group:          Development/Languages/Python
Source:         https://github.com/downloads/strohel/PyBayes/PyBayes-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-build

BuildRequires:  python-devel >= 2.6
BuildRequires:  python-cython >= 0.14.1

%if 0%{?fedora}
BuildRequires:  numpy >= 1.5.0
%else
BuildRequires:  python-numpy-devel >= 1.5.0
%endif

# following provides both clapack.h, cblas.h, cblas.so and lapack.so
BuildRequires:  atlas-devel

%if 0%{?suse_version}
%py_requires
%endif

%{!?python_sitelib: %global python_sitelib %(%{__python} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()")}

%description
PyBayes is an object-oriented Python library for recursive Bayesian estimation (Bayesian filtering)
that is convenient to use. Already implemented are Kalman filter, particle filter and marginalized
particle filter, all built atop of a light framework of probability density functions. PyBayes can
optionally use Cython for lage speed gains (Cython build can be several times faster in some
situations).

%prep
%setup -q -n PyBayes-%{version}

%build
python setup.py --use-cython=yes build build_prepare --blas-lib cblas --lapack-lib lapack_atlas

%check
python setup.py --use-cython=yes test

%install
python setup.py install --prefix=%{_prefix} --root=%{buildroot}

%files
%defattr(-,root,root,-)
%doc README.rst HACKING.rst ChangeLog.rst
%{python_sitelib}/*

%changelog
