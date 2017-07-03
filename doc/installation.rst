Installation und Upgrading
==========================



Preparation
-----------

This package requires Python 3.5 or later.
The package is tested on Linux (using Travis continuous integration).

Not all of the package's functionality is available for Microsoft Windows.
As some of the multi-core parallelizations rely on forking,
these won't work on Windows.
However, most other parts of the
package should work on Windows
as well.


My system's Python distribution is outdated, what now?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several Python distributions can coexist on a single system.
If you don't have access to a recent Python version via your
system's package manager (this might be the case for old 
Debian or Ubuntu operating systems),
it is recommended to install the latest version of the
`Anaconda Python 3 distribution <https://www.continuum.io/downloads>`_.


PIP Installation
----------------

Install with root rights into you system's Python distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package can be installed via pip.::

    pip install pyabc


into your system's Python distribution. This requires usually root access.


Install as user into your home directory (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing pyABC into you system's Python
distribution can be problematic as you might not want to
change your system's Python installation or you
don't have root rights.
The recommended alternative is to install pyABC into your
home directory with::

   pip install --user pyabc



GIT Installation
----------------

If you want the bleeding edge version, install directly from github::

   pip install git+https://github.com/neuralyzer/pyabc.git



.. _upgrading:


Upgrading
---------

If you want to upgrade from a previous
pyABC version, use::

    pip install --upgrade pyabc


instead of ``pip install``.
You can also consult the `pip documentation <https://pip.pypa.io/en/stable/>`_
on how to manage packages.
If you installed pyABC into your
home directory with
``pip install --user pyabc``, then upgrade also with the ``--user`` flag::


    pip install --upgrade --user pyabc





