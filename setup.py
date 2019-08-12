from setuptools import find_packages, setup

install_requires = [
    "dill",
    "multiprocess",
    "numpy",
    "matplotlib"
]

dev_requires = [
    "pytest",
    "pytest-cov",
    "flake8",
    "codecov"
]

doc_requires = [
    "sphinx"
    "sphinx-gallery"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="psopt",
    packages=find_packages(),
    version="v0.1.0",
    author="Artur de Luca",
    author_email="arturbackdeluca@gmail.com",
    description="A particle swarm optimizer for general use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artur-deluca/psopt",
    include_package_data=True,
    install_requires=install_requires,

    extras_require={
        "dev": install_requires + dev_requires,
        "doc": install_requires + doc_requires,
        "all": install_requires + dev_requires + doc_requires
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"

    ]
)
