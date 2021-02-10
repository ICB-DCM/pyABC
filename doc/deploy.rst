.. _deploy:


Deploy
======

The develop branch contains the working version of the package with
new features and bug fixes. Regularly, also new production versions
of the package should be released. The latest release is represented
in the main branch.

Versioning scheme
-----------------

For version numbers, we use ``A.B.C``, where

* ``C`` is increased for bug fixes,
* ``B`` is increased for new features and minor API breaking changes,
* ``A`` is increased for major API breaking changes.

Thus, we roughly follow the versioning scheme suggested
by the `Python packaging guide <https://packaging.python.org>`_.

Create a new release
--------------------

After new commits have been added via pull requests to the develop branch,
changes can be merged to main and a new version of pyABC can be released.
Every merge to main should coincide with an incremented version number
and a git tag on the respective merge commit.

Merge into main
~~~~~~~~~~~~~~~

1. create a pull request from develop to main,
2. check that all tests pass,
3. check that the documentation is up-to-date,
4. adapt the version number in ``pyabc/version.py`` (see above),
5. update the release notes in ``doc/releasenotes.rst``,
6. request a code review,
7. merge into the origin main branch.

To be able to actually perform the merge, sufficient rights may be required.
Also, at least one review is required.

Create a release on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~

After merging into main, create a new release on GitHub. This can be done
either directly on the project GitHub website, or via the CLI as described
in
`Git Basics - Tagging <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_.
In the release form,

* specify a tag with the new version as specified in ``pyabc/version.py``,
* include the latest additions to ``doc/releasenotes.rst`` in the release
  description.

Upload to PyPI
--------------

The upload to the python package index PyPI has been automatized via GitHub
Actions and is triggered whenever a new release tag is created.

Should it be necessary to manually upload a new version to PyPI,
proceed as follows: First, a so called "wheel" is created via::

    python setup.py bdist_wheel

A wheel is essentially a zip archive which contains the source code
and the binaries (if any).

This archive is uploaded using twine::

    twine upload dist/pyabc-x.y.z-py3-non-any.wheel

replacing x.y.z by the respective version number.

For a more in-depth discussion see also the
`section on distributing packages 
<https://packaging.python.org/tutorials/distributing-packages>`_
of the Python packaging guide.
