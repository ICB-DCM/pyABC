.. _contribute:


Contribute documentation
========================

Documentation is an essential part of the software development process.
We want to provide a useful piece of software. It is therefore necessary to
have a good documentation, such that the user knows how to use our package.
Contributions to the documentation are as welcome as contributions to the code.

Docstrings
----------

We follow the numpy docstring standard.
Check `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ for a
detailed explanation.


Contribute tests
================

We're committed to testing our code. Tests that are required to pass are located in the
``test`` folder. All files starting with ``test_`` contain tests and are automatically run
on Travis CI. To run them manually, type::

    python3 -m pytest test

You can also run specific tests only.

We encourage to test whatever possible. However, it might not always be easy to
test code which is based on random sampling. We still encourage to provide general sanity
and integration tests. We highly encourage a
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_ style.

Writing tests
-------------

Tests can be written with `pytest <http://docs.pytest.org/en/latest/>`_
or the `unittest <https://docs.python.org/3/library/unittest.html>`_ module.

PEP8 Style Guide
----------------

We try to respect the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard.
We run `flake8 <http://flake8.pycqa.org/en/latest/>`_ as part of the test
suite. The tests won't pass if flake8 complains.


Contribute code
===============

If you start working on a new feature or a fix, if not already done, please
create an issue on github, shortly describing your plans, and assign it to
yourself. Your starting point should not be the master branch, but the
develop branch, which contains the latest updates.

Create an own branch or fork, on which you can implement your changes. To
get your work merged, please:

1. create a pull request to the develop branch,
2. check that all tests on travis pass,
3. check that the documentation is up-to-date,
4. request a code review from the main developers.

Document all your changes in the pull request, and make sure to appropriately
resolve issues, and delete stale branches after a successful merge.
