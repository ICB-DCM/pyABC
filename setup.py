# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:37:43 2014

@author: emmanuel
"""

from setuptools import setup, find_packages


setup(install_requires=['numpy', 'scipy', 'pandas', 'cloudpickle',
                'gitpython', 'seaborn', 'scikit-learn',
                'matplotlib', 'sqlalchemy'],
     packages=find_packages(exclude=["IPythonNotebooks*", "figures*", "misc*", "pydokuwiki*", "scripts*"]),
     author='Emmanuel Klinger, Dennis Rickert',
     author_email='emmanuel.klinger@brain.mpg.de, dennis.rickert@helmholtz-muenchen.de',
     name="ICBayes",
     version="0.1.0",
     license="GPLv3",
     platforms="all",
     url="http:/icbayes.de",
     include_package_data=True,
     description='Massively parallel ABC for Python')
