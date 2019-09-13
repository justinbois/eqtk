.. _eqtk_to_df:

Convert output to Pandas DataFrame
==================================

Low level interface functions return equilibrium concentrations as Numpy arrays. The high level interface functions (``eqtk.solve()``, ``eqtk.fixed_value_solve()``, and ``eqtk.volumetric_titration()``) can also return Numpy arrays if the input is also Numpy arrays and the ``names`` keyword argument is unspecified. After performing a calculation that returns Numpy arrays, the user may with to convert the output to a Pandas Series of DataFrame for convenience in plotting or for further processing. The ``eqtk.to_df()`` function provides this functionality.

Before demonstrating how to use the function, it is important to note that the preferred way to generate rich output is to generate it directly from the high level interfaces by supplying arguments that enable return of DataFrames. The ``eqtk.to_df()`` function is only meant as a post processing step if the user either neglected to provide descriptive input to the high level interface function or chose to use the low level interface for speed purposes.

As a demonstration, we will use the example chemical reaction system

.. math::
    \begin{array}{lcl}
    \mathrm{A} \rightleftharpoons \mathrm{C} & & K = 0.5\\
    \mathrm{AB} \rightleftharpoons \mathrm{A} + \mathrm{B} & & K = 0.02 \text{ mM}\\
    \mathrm{BB} \rightleftharpoons 2\mathrm{B}& & K = 0.1 \text{ mM}\\
    \mathrm{BC} \rightleftharpoons \mathrm{B} + \mathrm{C}& & K = 0.01 \text{ mM}.
    \end{array}

The annotated stoichiometric matrix is

.. math::
  \mathsf{N} =
  \begin{pmatrix}
    \mathrm{A} & \mathrm{B} & \mathrm{C} & \mathrm{AB} & \mathrm{BB} & \mathrm{BC} \\ \hline
    -1 & 0 & 1 & 0 & 0 & 0 \\
    1 & 1 & 0 & -1 & 0 & 0 \\
    0 & 2 & 0 & 0 & -1 & 0 \\
    0 & 1 & 1 & 0 & 0 & -1
  \end{pmatrix},

with equilibrium constants

.. math::
    \mathbf{K} = \left(\begin{array}{l}
    0.5\\ 
    0.02\text{ mM}\\
    0.1\text{ mM}\\
    0.01\text{ mM}
    \end{array}
    \right).

We will first solve the coupled equilibrium problem using ``eqtk.solve()`` with all input as Numpy arrays and without the ``names`` keyword argument.

.. code-block:: python

    N = np.array([[-1,  0,  1,  0,  0,  0],
                  [ 1,  1,  0, -1,  0,  0],
                  [ 0,  2,  0,  0, -1,  0],
                  [ 0,  1,  1,  0,  0, -1]])

    c0 = np.array([1, 1, 0, 0, 0, 0])

    K = np.array([0.05, 0.02, 0.1, 0.01])

    c = eqtk.solve(c0=c0, N=N, K=K, units='mM')

Here, we have calculated the equilibrium concentrations for a single set of initial conditions, so calling ``eqtk.to_df()`` will return a Pandas Series.

.. code-block:: python

	names = ['A', 'B', 'C', 'AB', 'BB', 'BC']
	c_series = eqtk.to_df(c=c, c0=c0, names=names, units='mM')

The result is ::

	A__0 (mM)     1.000000
	B__0 (mM)     1.000000
	C__0 (mM)     0.000000
	AB__0 (mM)    0.000000
	BB__0 (mM)    0.000000
	BC__0 (mM)    0.000000
	A (mM)        0.188228
	B (mM)        0.077504
	C (mM)        0.009411
	AB (mM)       0.729418
	BB (mM)       0.060068
	BC (mM)       0.072942
	dtype: float64

If, however, we consider a set of concentrations, we get a DataFrame when converting using ``eqtk.to_df()``.

.. code-block:: python

    c0 = np.array([[1.0, 1.0, 0, 0, 0, 0],
    			   [0.5, 0.5, 0, 0, 0, 0],
    			   [0.1, 0.1, 0, 0, 0, 0]])

    c = eqtk.solve(c0=c0, N=N, K=K, units='mM')

    c_df = eqtk.to_df(c=c, c0=c0, names=names, units='mM')

The result is has columns ``['A__0 (mM)', 'B__0 (mM)', 'C__0 (mM)', 'AB__0 (mM)', 'BB__0 (mM)', 'BC__0 (mM)', 'A (mM)', 'B (mM)', 'C (mM)', 'AB (mM)', 'BB (mM)', 'BC (mM)']`` and has three rows, one for each set of concentrations. Executing ``print(c_df[c_df.columns[~c_df.columns.str.contains('__0')]])`` gives ::

	     A (mM)    B (mM)    C (mM)   AB (mM)   BB (mM)   BC (mM)
	0  0.188228  0.077504  0.009411  0.729418  0.060068  0.072942
	1  0.118379  0.057704  0.005919  0.341547  0.033297  0.034155
	2  0.039494  0.026946  0.001975  0.053211  0.007261  0.005321



