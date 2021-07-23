Intallation
===============================

To install from pip, for package name issues the package name was chosen as PyCustomFocus:

.. code-block:: python

   pip install PyCustomFocus


Dependences
------------

PyFocus uses various packages. Here we show an example of the needed installation:

.. code-block:: python
   conda install numpy
   conda install scipy
   pip install config tqdm matplotlib PyQt5 qtpy pyqtgraph qdarkstyle os configparser time

Importing
----------

The modules used in the examples are imported with:

.. code-block:: python

   from PyFocus import sim, plot
   import numpy as np