Installation
===============================

At https://github.com/Stefani-Lab/PyFocus we provide a Windows executable version of PyFocus's GUI, called "PyFocus.exe", that can be downloaded and executed without any installations or Python enviroment.

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
   pip install config==0.5.1 
   pip install tqdm==4.62.3 
   pip install matplotlib==3.3.4
   pip install PyQt5==5.15.4
   pip install qtpy==1.11.2 
   pip install pyqtgraph==0.11.1 
   pip install configparser==5.0.2  

Package versions are specified for consistency. If a package does not have the specified version available (represented by the ==version part of pip install), you can try removing this restrain (by deleting the ==version part of the command).  

After this, we can use a python script.

Importing
============

The modules used in the examples are imported with:

.. code-block:: python

   from PyFocus import sim, plot
   import numpy as np