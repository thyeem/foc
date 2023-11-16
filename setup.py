import setuptools

from foc import capture, reader

setuptools.setup(
    name="foc",
    version=capture(
        r"__version__\s*=\s*['\"](.*)['\"]", reader("foc/__init__.py").read()
    ),
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    description="A collection of python functions for somebody's sanity",
    long_description=reader("README.md").read(),
    long_description_content_type="text/markdown",
    license_files=("LICENSE",),
    install_requires=[],
    url="https://github.com/thyeem/foc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
