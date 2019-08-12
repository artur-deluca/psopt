.. _install:

============
Installation
============

Python version support
----------------------

Officially Python 3.6 and above

Installing PSOpt
-----------------

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

PSOpt can be installed via pip from
`PyPI <https://pypi.org/project/psopt>`__.

::

    pip install psopt


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

Clone the repository via:

::

    git clone https://github.com/artur-deluca/psopt/ --depth=1


Activate your virtual environment and run:

::

    python setup.py install
    # or alternatively
    pip install -e .


If you wish to install the development dependencies, run:
:: 

    python setup.py build
    # or alternatively
    pip install -e.[all]


See the :ref:`contributing guide <contributing>` for further information on contributing

Running the test suite
----------------------

To run the tests written for psopt, make sure you have `pytest` installed in your venv. 
Additionally, if you wish to run coverage analysis as well, make sure to have ``pytest-cov`` installed as well.
::

    # to simply execute the tests run:
    pytest
    # to run coverage as well run:
    pytest --cov=psopt
    # or alternatively:
    make test

.. _install.dependencies:

Dependencies
------------

================================================================ ==========================
Package                                                          Minimum supported version
================================================================ ==========================
`NumPy <http://www.numpy.org>`__                                 1.13.3
`dill <https://github.com/uqfoundation/dill>`__                  0.3.0
`multiprocess <https://github.com/uqfoundation/multiprocess>`__  0.70.8
================================================================ ==========================

.. note:: As of today, ``psopt`` uses ``dill`` and ``multiprocess`` as a workaround to deal with pickling issues on the standard ``multiprocessing`` Python library.