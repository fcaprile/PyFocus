Installation
===============================

At https://github.com/Stefani-Lab/PyFocus we provide a windows executable version of PyFocus's GUI, called "PyFocus.exe", that can be downloaded and executed without any other installation or Python enviroment.

For instalation of the high-level functions and the GUI class for use in custom scripts, first we install a new enviroment in wich we use Python 3.6. This version is needed for using the GUI, but is not required for the high-level functions we provide. From an anaconda prompt with admin privileges:

.. code-block:: python

   conda create --name pyfocus python=3.6
   conda activate pyfocus


To install from pip, for package name issues the package name was chosen as PyCustomFocus:

.. code-block:: python

   pip install PyCustomFocus


Dependencies
============

PyFocus uses various packages. Here we show an example of the needed installation, some packages are preferably installed using conda:

.. code-block:: python

   conda install numpy
   conda install scipy
   conda install qdarkstyle
   pip install PyCustomFocus
   pip install config 
   pip install tqdm 
   pip install matplotlib 
   pip install PyQt5 
   pip install qtpy 
   pip install pyqtgraph 
   pip install configparser  

After this, we can use a python script.

Importing
============

The modules used in the examples are imported with:

.. code-block:: python

   from PyFocus import sim, plot
   import numpy as np