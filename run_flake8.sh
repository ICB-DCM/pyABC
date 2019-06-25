#!/bin/sh

EXCLUDE=$(cat flake8_exclude.txt)
python3 -m flake8 --exclude=$EXCLUDE $2
