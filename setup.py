from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__), "pyabc", "version.py")) as f:
    version = f.read().split("\n")[0].split("=")[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    install_requires=["numpy>=1.15.2", "scipy>=1.1.0",
                      "pandas>=0.23.4", "cloudpickle>=0.7.0",
                      "flask_bootstrap>=3.3.7.1.dev1", "flask>=1.0.2",
                      "bokeh>=0.13.0", "redis>=2.10.6",
                      "dill>=0.2.8.2", "gitpython>=2.1.11",
                      "scikit-learn>=0.21.2", "matplotlib>=3.0.0",
                      "sqlalchemy>=1.3.0", "click>=7.0",
                      "feather-format>=0.4.0", "bkcharts>=0.2",
                      "distributed>=1.23.3", "pygments>=2.2.0",
                      "IPython>=7.0.1", "pyarrow>=0.14.1"],
    extras_require={"R": ["rpy2>=3.2.0", "cffi>=1.13.1"],
                    "amici-petab": ["petab>=0.1.1", "amici>=0.10.18"]},
    python_requires='>=3.6',
    packages=find_packages(exclude=["examples*", "devideas*",
                                    "test*", "test"]),
    author='Emmanuel Klinger, Yannik Schälte, Elba Raimundez',
    author_email='yannik.schaelte@gmail.com',
    name="pyabc",
    version=version,
    platforms="all",
    url="https://github.com/icb-dcm/pyabc",
    include_package_data=True,
    description='Distributed, likelihood-free ABC-SMC inference',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
      'Programming Language :: Python :: 3.6',
      'License :: OSI Approved :: BSD License',
      'Operating System :: OS Independent',
    ],
    license='BSD-3-Clause',
    keywords='likelihood-free inference, abc, '
             'approximate bayesian computation, sge, distributed',
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
