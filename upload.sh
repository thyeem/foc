#!/bin/sh

python -m build
python -m twine upload dist/*
