#!/bin/sh

PER_FILE_IGNORES='
*/__init__.py:F401
test/*.py:T001,S101
test_performance/*.py:T001
*/cli.py:T001'

python3 -m flake8 pyabc test test_performance \
  --extend-ignore='S403,S301,C408' \
  --per-file-ignores="$PER_FILE_IGNORES"
