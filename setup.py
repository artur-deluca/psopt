from setuptools import find_packages, setup

install_requires = [
    'dill==0.3.0',
    'multiprocess==0.70.8',
    'numpy==1.16.4'
]

tests_require = [
    'pytest==5.0.1',
]

setup(
    name='psopt',
    packages=find_packages(),
    version='v0.1.0a0',
    author='Artur de Luca',
    install_requires=install_requires,
    extras_require={
        'tests': tests_require,
        'all': install_requires + tests_require
    }
)