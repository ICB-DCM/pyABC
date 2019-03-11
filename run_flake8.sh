#!/bin/sh

EXCLUDE=$(cat flake8_exclude.txt)
python -m flake8 --exclude=$EXCLUDE $2
