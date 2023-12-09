#!/bin/sh

pytest -v && python -m doctest -v foc/*.py
