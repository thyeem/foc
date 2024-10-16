import re

import setuptools

setuptools.setup(
    name="foc",
    version=re.compile(r"__version__\s*=\s*['\"](.*)['\"]").findall(
        open("foc/__init__.py", "r").read()
    )[0],
    description="A collection of python functions for somebody's sanity",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thyeem/foc",
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    license="MIT",
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="functional functools functional-python",
    packages=setuptools.find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)
