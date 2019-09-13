.. _installation:

Installation
============

You can install EQTK using pip. ::

	$ pip install --upgrade eqtk


Dependencies
------------

EQTK has the following dependencies.

- Python 3.6 or newer
- Numpy_
- Pandas_
- Numba_

EQTK will work without Numba, but will be much, much slower. It is very strongly recommended that you have Numba installed.

If you with you work through the case studies on your machine, you will need Jupyter notebook or JupyterLab installed, along with HoloViews and Bokeh. The easiest way to ensure you have this is to follow the `HoloViz installation instructions`_.

.. _NumPy: http://www.numpy.org/
.. _Pandas: http://pandas.pydata.org
.. _Numba: http://numba.pydata.org
.. _HoloViz installation instructions: https://holoviz.org/installation.html