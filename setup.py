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
                        "flask", "flask_bootstrap", "bokeh", "redis",
                        "dill",
                        'gitpython', 'seaborn', 'scikit-learn',
                        'matplotlib', 'sqlalchemy',
                        'distributed'],
      packages=find_packages(exclude=["examples*", "devideas*",
                                      "test*", "test"]),
      author='Emmanuel Klinger, Dennis Rickert',
      author_email='emmanuel.klinger@brain.mpg.de, '
                   'dennis.rickert@helmholtz-muenchen.de',
      name="pyabc",
      version=version,
      license="GPLv3",
      platforms="all",
      url="http://pyabc.readthedocs.io/en/latest/",
      include_package_data=True,
      description='Parallel ABC for Python',
      classifiers=[
        'Programming Language :: Python :: 3.6'
      ],
      keywords='inference abc approximate bayesian '
               'computation parallel distributed',
      zip_safe=False,  # not zip safe b/c of Flask templates
      entry_points={
        'console_scripts': [
              'abc-server = pyabc.visserver.server:run_app',
        ]
    },
      )
