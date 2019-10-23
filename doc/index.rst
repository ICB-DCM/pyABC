Welcome to pyABC's documentation!
=================================

.. image:: https://travis-ci.org/ICB-DCM/pyABC.svg?branch=master
   :target: https://travis-ci.org/ICB-DCM/pyABC
.. image:: https://readthedocs.org/projects/pyabc/badge/?version=latest
   :target: https://pyabc.readthedocs.io/en/latest
.. image:: https://api.codacy.com/project/badge/Grade/923a9ab160e6420b9fc468701be60a98
   :target: https://www.codacy.com/app/yannikschaelte/pyABC?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ICB-DCM/pyABC&amp;utm_campaign=Badge_Grade
.. image:: https://codecov.io/gh/ICB-DCM/pyABC/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/ICB-DCM/pyABC
.. image:: https://badge.fury.io/py/pyabc.svg
   :target: https://badge.fury.io/py/pyabc
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3364560.svg
   :target: https://doi.org/10.5281/zenodo.3364560

:Release: |version|
:Source code: https://github.com/icb-dcm/pyabc

.. image:: logo/logo.png
   :alt: pyABC logo
   :align: center
   :scale: 30 %

pyABC is a framework for distributed, likelihood-free inference.
That means, if you have a model and some data and want to know the posterior
distribution over the model parameters, i.e. you want to know with which
probability which parameters explain the observed data, then pyABC might be
for you.

All you need is some way to numerically draw samples from the model, given
the model parameters. pyABC "inverts" the model for you and tells you
which parameters were well matching and which ones not. You do **not** need
to analytically calculate the likelihood function.

pyABC runs efficiently on multi-core machines and distributed cluster setups.
It is easy to use and flexibly extensible.

If you use it in your work, you can cite the paper:

    Emmanuel Klinger, Dennis Rickert, Jan Hasenauer; pyABC: distributed, likelihood-free inference;
    Bioinformatics 2018; https://doi.org/10.1093/bioinformatics/bty361


.. toctree::
   :maxdepth: 2
   :caption: User's guide

   what

.. toctree::
   :maxdepth: 2
   
   installation

.. toctree::
   :maxdepth: 2
   
   examples

.. toctree::
   :maxdepth: 2

   sampler

.. toctree::
   :maxdepth: 2
   
   sge

.. toctree::
   :maxdepth: 2
   
   export_db

.. toctree::
   :maxdepth: 2
   
   web_visualization


.. toctree::
   :maxdepth: 2
   :caption: About
   
   releasenotes

.. toctree::
   :maxdepth: 2
   
   about


.. toctree::
   :maxdepth: 2
   :caption: Developer's guide
   
   contribute


.. toctree::
   :maxdepth: 2
   
   deploy


.. toctree::
   :maxdepth: 2
   :caption: API reference
   
   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
