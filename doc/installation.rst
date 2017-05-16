Installation
============



Preparation
-----------

This package requires Python 3.5 or later.
The package is tested on Linux (using Travis continuous integration).

Not all of the package's functionality is available for Microsoft Windows.
As some of the multi-core parallelizations rely on forking, these won't work
on Windows.
However, most other parts of the package should work on Windows
as well.



PIP
---

The package can be installed via pip.::

    pip install pyabc


GIT
---

If you want the bleeding edge verision, install directly from github::

   pip install git+https://github.com/neuralyzer/pyabc.git




If you want to upgrade from a previous pyABC version, use
``pip install --upgrade`` instead of ``pip install``.
You can also consult the `pip documentation <https://pip.pypa.io/en/stable/>`_
on how to manage packages.
