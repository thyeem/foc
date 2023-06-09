import setuptools

with open("README.md", "r") as f:
    long_decription = f.read()

setuptools.setup(
    name="foc",
    version="0.1.3",
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    description="A collection of python functions for somebody's sanity",
    long_decription=long_decription,
    long_decription_content_type="text/markdown",
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
