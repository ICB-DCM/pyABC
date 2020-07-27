from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__), "pyabc", "version.py")) as f:
    version = f.read().split("\n")[0].split("=")[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    install_requires=["numpy>=1.19.1", "scipy>=1.5.2",
                      "pandas>=1.0.5", "cloudpickle>=1.5.0",
                      "flask_bootstrap>=3.3.7.1", "flask>=1.1.2",
                      "bokeh>=2.1.1", "redis>=2.10.6",
                      "dill>=0.3.2", "gitpython>=3.1.7",
                      "scikit-learn>=0.23.1", "matplotlib>=3.3.0",
                      "sqlalchemy>=1.3.18", "click>=7.1.2",
                      "feather-format>=0.4.1",
                      "distributed>=2.21.0", "pygments>=2.6.1",
                      "IPython>=7.16.1", "pyarrow>=1.0.0"],
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
