from setuptools import setup
import os

# extract version
with open(os.path.join(os.path.dirname(__file__), "pyabc", "version.py")) as f:
    version = f.read().split("\n")[0].split("=")[-1].strip(' ').strip('"')

# all other information comes from setup.cfg

setup(version=version)
