Contributing Code
=================

Testing
-------

We're commited to testing our code. Tests are run on Travis CI.
We encourage to test whatever possible. However, it might not always be easy to
test code which is based on random sampling. We still encourage to provide general sanity
and intergation tests. We highly encourage a
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_ style.

Python (PEP8)
~~~~~~~~~~~~~

We try to respect the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard.
In the future we might add automated flake8 runs.

Writing tests
-------------

Test can be written with `pytest <http://docs.pytest.org/en/latest/>`_
or the `unittest <https://docs.python.org/3/library/unittest.html>`_ module.


Versioning scheme
-----------------

For version numbers, we use ``A.B.C``, where

* ``C`` is increased for bug fixes
* ``B`` is increased for new features
* ``A`` for API breaking, backwards incompatible changes.

That is, we follow the versioning scheme suggested
by the `Python packaging guide <https://packaging.python.org>`_.