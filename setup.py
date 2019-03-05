from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__), "pyabc", "version.py")) as f:
    version = f.read().split("\n")[0].split("=")[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(install_requires=["numpy", "scipy", "pandas", "cloudpickle",
                        "flask_bootstrap", "flask", "bokeh", "redis",
                        "dill",
                        "gitpython", "scikit-learn",
                        "matplotlib", "sqlalchemy", "click",
                        "feather-format", "bkcharts",
                        "distributed", "pygments", "IPython"],
      extra_requires={"R": ["rpy2"],
                      "git": ["gitpython"]},
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
