# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:37:43 2014

@author: emmanuel
"""

from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "pyabc", "version.py")) as f:
    version = f.read().split("\n")[0].split("=")[-1].strip(' ').strip('"')

setup(install_requires=['numpy', 'scipy', 'pandas', 'cloudpickle',
                        "flask_bootstrap", "flask", "bokeh", "redis",
                        "dill",
                        'gitpython', 'seaborn', 'scikit-learn',
                        'matplotlib', 'sqlalchemy', 'click',
                        'feather-format', 'bkcharts',
                        'distributed', 'pygments', 'IPython'],
      extra_requires={"R": ["rpy2"],
                      "git": ["gitpython"]},
      packages=find_packages(exclude=["examples*", "devideas*",
                                      "test*", "test"]),
      author='Emmanuel Klinger, Yannik Schaelte, Elba Raimundez',
      author_email='yannik.schaelte@helmholtz-muenchen.de',
      name="pyabc",
      version=version,
      license="BSD",
      platforms="all",
      url="http://pyabc.readthedocs.io/en/latest/",
      include_package_data=True,
      description='Distributed, likelihood-free ABC-SMC inference',
      classifiers=[
        'Programming Language :: Python :: 3.6'
      ],
      keywords='likelihood-free inference abc approximate bayesian '
               'computation sge distributed',
      zip_safe=False,  # not zip safe b/c of Flask templates
      entry_points={
        'console_scripts': [
              'abc-server = pyabc.visserver.server:run_app',
              'abc-redis-worker = '
              'pyabc.sampler.redis_eps.cli:work',
              'abc-redis-manager = '
              'pyabc.sampler.redis_eps.cli:manage',
              'abc-export = '
              'pyabc.storage.export:main',
        ]
    },
      )
