import setuptools

from foc import capture, reader

setuptools.setup(
    name="foc",
    version=capture(
        r"__version__\s*=\s*['\"](.*)['\"]", reader("foc/__init__.py").read()
    ),
    description="A collection of python functions for somebody's sanity",
    long_description=reader("README.md").read(),
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
