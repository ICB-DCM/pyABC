# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:37:43 2014

@author: emmanuel
"""

from setuptools import setup


setup(install_requires=['numpy', 'scipy', 'pandas', 'cloudpickle',
                "flask", "flask_bootstrap", "bokeh", "redis",
                "dill",
                'gitpython', 'seaborn', 'scikit-learn',
                'matplotlib', 'sqlalchemy',
                'distributed'],
    packages=["pyabc", "visserver"],
    author='Emmanuel Klinger, Dennis Rickert',
    author_email='emmanuel.klinger@brain.mpg.de, dennis.rickert@helmholtz-muenchen.de',
    name="pyabc",
    version="0.2.0",
    license="GPLv3",
    platforms="all",
    url="https://github.com/neuralyzer/pyabc",
    include_package_data=True,
    description='Massively parallel ABC for Python',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
   keywords='inference abc approximate bayesian computation parallel distributed',
    zip_safe=False,  # not zip safe b/c of Flask templates
    entry_points={
        'console_scripts': [
              'abc-server = visserver.server:run_app',
        ]
},
      )
