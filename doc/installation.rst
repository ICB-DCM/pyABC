Installation
============


.. todo::
    Put the real install URLs here.


Preparation
-----------

This package requires Python 3.5 or later.


PIP
---

The package can be installed via pip.::

    pip install pyabc


GIT
---

If you want the bleeding edge verision, install directly from github::

   pip install git+https://github.com/neuralyzer/connectome.git


Running the tests
-----------------

Run the tests with::

  python -m pyabc.test

This could take a while, depending on your machine.



.. note::

   Some of the tests are stochastic
   and may therefore fail from time to time although there is actually no problem. If you get something like a quantity is, e.g.
   0.053 but it should be less than 0.05 it is usually safe to ignore this. In any case the test suite will make sure everything
   is correctly installed.
