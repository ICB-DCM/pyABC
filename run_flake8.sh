#!/bin/sh

python3 -m flake8 pyabc test --per-file-ignores='*/__init__.py:F401'
