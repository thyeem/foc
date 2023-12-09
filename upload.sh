#!/bin/sh

python setup.py sdist bdist_wheel

# python -m twine upload dist/*
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
