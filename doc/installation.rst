.. _installation:

Installation und Upgrading
==========================


Preparation
-----------

This package requires Python 3.6 or later.
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

We're assuming you're on a Linux environment.
Use the most recent Anaconda Python 3.x distribution.
As of writing this documentation, this is the
`Anaconda Python 3.6 <https://www.continuum.io/downloads>`_ distribution.
To install it, run::

   wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh

to download the installer. To execute the installer run::

   bash Anaconda3-4.4.0-Linux-x86_64.sh

and follow the guided installation process (i.e. approve the license
and tell the installer where to install it to). You might want to replace
the "4.4.0" by the most recent version of Anaconda.
Find out on the `Anaconda Download <https://www.continuum.io/downloads>`_
page which one it is.


.. note::

    The Anaconda installer asks you at the end of the installation whether
    you want to use Anaconda Python as your default Python:: bash

       Do you wish the installer to prepend the Anaconda3 install location
       to PATH in your /home/username/.bashrc ? [yes|no]
       [no] >>>

    If you answer yes, the path to the Anaconda installation is prepended to
    your ``PATH`` environment variable and subsequent calls to ``pip``
    (see below) use the Anaconda Python pip (check with the command
    ``which pip``).
    If you answer no, you need to ensure manually, that the correct Python
    installation is used.
    Just saying "yes" here might safe you from some difficulties later on.


.. _install-optional:

Optional dependencies
---------------------

pyABC has an optional interface to the R language. To enable it install
pyabc via ``pip install pyabc[R]``. All Python based features will work just
fine if R is not installed. See also
:ref:`pyABC's external API <api_external>`.

pyABC optionally uses git to store commit hashed in its database leveraging
the gitpython package. This feature can be installed via
``pip install pyabc[git]``.
