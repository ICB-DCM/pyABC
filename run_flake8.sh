#!/bin/sh

python3 -m flake8 pyabc test \
    --extend-ignore='S403,S301' \
    --per-file-ignores='*/__init__.py:F401 test/*:T001,S101 */cli.py:T001'
