.. _installation:

Install und Upgrade
===================

Preparation
-----------

This package requires Python 3.8 or later.
The package is continuously tested on Linux, and in parts on iOS,
via GitHub Actions.

While many parts of the package should work on Microsoft Windows
as well, in particular the multi-core parallelizations rely on forking,
which won't be available.
Still, the main development focus is on Linux.

My system's Python distribution is outdated, what now?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several Python distributions can coexist on a system.
If you don't have access to a recent Python version via your
system's package manager (may be the case for old systems),
we recommend to install the latest version of the
`Anaconda Python 3 distribution <https://www.continuum.io/downloads>`_.
See also: :ref:`anacondaCluster`.

PIP Installation
----------------

Install with root rights into you system's Python distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package can be installed via pip.::

    pip install pyabc

into your system's Python distribution. This requires usually root access.

Install as user into your home directory (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing pyABC into your system's Python
distribution can be problematic as you might not want to
change your system's Python installation or you
don't have root rights.
The recommended alternative is to install pyABC into your
home directory with::

   pip install --user pyabc

GIT Installation
----------------

If you want the bleeding edge version, install directly from github::

   pip install git+https://github.com/icb-dcm/pyabc.git

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

.. _anacondaCluster:

Installing Anaconda on a Cluster environment
--------------------------------------------

To install `Anaconda <https://anaconda.com/products/distribution>`_, run::

   wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
   bash Anaconda3-2021.11-Linux-x86_64.sh

and follow the installation guide.
Replace the "2021.11" by the most recent version of Anaconda, see
https://repo.anaconda.com/archive.

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
provides an alternative, minimal installer for conda, including
only conda, Python, and some core and useful packages. Install the latest
version via::

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh

.. _install-optional:

Optional dependencies
---------------------

pyABC has various optional dependencies, see `setup.cfg`.

In particular, pyABC has optional interfaces to the :ref:`R <api_external_r>`
and :ref:`Julia <api_external_julia>` languages, see the API documentation
for details.

pyABC optionally uses git to store commit hashes in its database,
leveraging the gitpython package. This feature can be installed via
``pip install pyabc[git]``.
