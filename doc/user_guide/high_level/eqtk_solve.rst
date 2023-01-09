.. _eqtk_solve:

Generic equilibrium solver
==========================

EQTK's main high level interface to solving the coupled equilibrium problem is ``eqtk.solve()``. It requires as input initial concentrations :math:`\mathbf{c}^0` and either a stoichiometric matrix :math:`\mathsf{N}` or a conservation matrix :math:`\mathsf{A}`. Depending on the data types of these inputs, other inputs, such as equilibrium constants and free energies, may be required. 

In what follows, we will assume that Numpy, Pandas, and EQTK have been imported respectively as ``np``, ``pd``, and ``eqtk``.



Example problem
---------------

As we demonstrate the usage of ``eqtk.solve()``, it is useful to have an example in mind. We will use the example from the :ref:`core concepts <Core concepts>` part of the user guide (which you should read if you have not already). The chemical reactions and associated equilibrium constants for that example are

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

An elemental conservation matrix is

.. math::
  \mathsf{A} =
  \begin{pmatrix}
    \mathrm{A} & \mathrm{B} & \mathrm{C} & \mathrm{AB} & \mathrm{BB} & \mathrm{BC} \\ \hline
    1 & 0 & 1 & 1 & 0 & 1 \\
    0 & 1 & 0 & 1 & 2 & 1
  \end{pmatrix}.

The free energies, in units of the thermal energy :math:`kT` are related to the equilibrium constants. We will compute them from the equilibrium constants later when we need them.



Stoichiometric matrix N given
-----------------------------

If an :math:`r\times n` stoichiometric matrix :math:`\mathsf{N}` is given, you may either specify :math:`r` equilibrium constants or :math:`n` free energies, in addition to :math:`\mathbf{c}^0`, which is always required.


N given as a Numpy array
^^^^^^^^^^^^^^^^^^^^^^^^

The stoichiometric matrix `\mathsf{N}` may be given as a Numpy array. In this case, the ordering of the columns maps to the respective species. In our definition of :math:`\mathsf{N}`, we implicitly chose the index-species mapping shown below.

+---------+-----------+
| index   | species   |
+=========+===========+
| 0       | A         |
+---------+-----------+
| 1       | B         |
+---------+-----------+
| 2       | C         |
+---------+-----------+
| 3       | AB        |
+---------+-----------+
| 4       | BB        |
+---------+-----------+
| 5       | BC        |
+---------+-----------+

So, we can build a Numpy array for :math:`\mathsf{N}`.

.. code-block:: python

    N = np.array([[-1,  0,  1,  0,  0,  0],
                  [ 1,  1,  0, -1,  0,  0],
                  [ 0,  2,  0,  0, -1,  0],
                  [ 0,  1,  1,  0,  0, -1]])

We need to preserve the ordering in the specification of the initial concentrations :math:`\mathbf{c}^0`. To specify a initial concentration of :math:`[A]_0 = [B]_0 = 1` mM with all other concentrations zero, we build a Numpy array

.. code-block:: python

    c0 = np.array([1, 1, 0, 0, 0, 0])


Equilibrium constants as Numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We must also specify the equilibrium constants as a Numpy array if the stoichiometric matrix is also specified as a Numpy array.

.. warning::
    The units of the inputted ``c0`` and ``K`` must be consistent, meaning that they both must use the same units for concentration. In this case, the concentration units are millimolar.

.. code-block:: python

    K = np.array([0.05, 0.02, 0.1, 0.01])

Entry ``K[i]`` corresponds to the chemical reaction given by the *i*th row of the stoichiometric matrix ``N``.

We can now solve for the equilibrium concentrations

.. code-block:: python

    eqtk.solve(c0=c0, N=N, K=K, units='mM')

The output is a Numpy array containing the equilibrium concentrations preserving the order of the inputs. ::

    array([0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184])

Free energies as Numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, we can specify free energies of each species instead of the equilibrium constants for the chemical reactions. In practice, you would enter these directly, but to keep the calculations consistent, we will calculate the free energies, using one of EQTK's private functions to compute the density of water to make the conversion. The resulting free energies are dimensionless (in units of the thermal energy :math:`kT`).

.. code-block:: python

    water_density = eqtk.water_density(293.15, 'mM')

    G_A = 0
    G_B = 0
    G_C = -np.log(K[0])
    G_AB = np.log(K[1] / water_density)
    G_BB = np.log(K[2] / water_density)
    G_BC = np.log(K[3] / water_density) + G_C

    G = np.array([G_A, G_B, G_C, G_AB, G_BB, G_BC])

With ``N`` as a Numpy array, ``G`` contains the free energies where ``G[j]`` is the free energy of the compound given by column ``j`` in ``N``.

Now, solving for the equilibrium concentrations,

.. code-block:: python

    eqtk.solve(c0=c0, N=N, G=G, units='mM')

The result is the same. ::

    array([0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184])


Initial concentrations as a 2D Numpy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We may wish to compute the equilibrium concentrations for multiple initial concentrations. This is accomplished by passing in ``c0`` as a two-dimensional Numpy array. Each row corresponds to a different equilibrium calculation, with the columns corresponding to the chemical species. Here is an example using three different concentrations of B.

.. code-block:: python

    c0 = np.array([[1,   0, 0, 0, 0, 0],
                   [1, 0.5, 0, 0, 0, 0],
                   [1,   1, 0, 0, 0, 0]])

    eqtk.solve(c0=c0, N=N, K=K, units='mM')

The output is ::

    array([[0.95238103, 0.        , 0.04761905, 0.        , 0.        , 0.        ],
           [0.49849994, 0.01738215, 0.024925  , 0.43325006, 0.00302139, 0.04332501],
           [0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184]])


Naming the chemical species
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If desired, you may specify names for the respective chemical species using the ``names`` keyword argument. This allows for richer output; the result is either a Pandas Series (for one-dimensional ``c0``) or DataFrame (for two-dimensional ``c0``). Here, we will again use the two-dimensional ``c0`` from the previous calculation.

.. code-block:: python

    names = ['A', 'B', 'C', 'AB', 'BB', 'BC']
    c = eqtk.solve(c0=c0, N=N, K=K, units='mM', names=names)

The result is a Pandas DataFrame with descriptive column names, ::

    ['[A]__0 (mM)', '[B]__0 (mM)', '[C]__0 (mM)', '[AB]__0 (mM)',
     '[BB]__0 (mM)', '[BC]__0 (mM)', '[A] (mM)', '[B] (mM)', '[C] (mM)',
     '[AB] (mM)', '[BB] (mM)', '[BC] (mM)']

The columns with ``__0`` indicate the initial conditions used in the calculation, and the remaining columns indicate the equilibrium concentrations. We can extract just the columns that contain the equilibrium concentrations by selecting those that do not contain the string ``__0``.

.. code-block:: python

    c[c.columns[~c.columns.str.contains('__0')]]
       [A] (mM)  [B] (mM)  [C] (mM)  [AB] (mM)  [BB] (mM)  [BC] (mM)
    0  0.952381  0.000000  0.047619   0.000000   0.000000   0.000000
    1  0.498500  0.017382  0.024925   0.433250   0.003021   0.043325
    2  0.188228  0.077504  0.009411   0.729418   0.060068   0.072942

.. note::
    
    The units are given in parentheses next to the brackets (denoting concentration) around the species name. If the ``units`` keyword argument is ``None``, the phrase ``mole fraction`` appears in the parentheses.


N given as a Pandas DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The descriptive output when the names of the chemical species are given is useful for keeping the output organized. Such organization is also useful when specifying the *input* for ``eqtk.solve()``. The function accepts the stroichometric matrix given as a Pandas DataFrame as well. The names of the columns are then assumed to be the names of the chemical species (just the names, not including the brackets and units included in output).

.. code-block:: python

    N = np.array([[-1,  0,  1,  0,  0,  0],
                  [ 1,  1,  0, -1,  0,  0],
                  [ 0,  2,  0,  0, -1,  0],
                  [ 0,  1,  1,  0,  0, -1]])

    names = ['A', 'B', 'C', 'AB', 'BB', 'BC']

    N_df = pd.DataFrame(data=N, columns=names)



Specification of equilibrium constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this data frame, we use Pandas's default row indexing, but a user may wish to name each reaction for reference. Because of this, EQTK does not assume an ordering of the data frame, so the equilibrium constants *must* be included in the data frame containing the stoichiometric matrix. They are included in a column entitled ``'equilibrium constant'``. This column must be present, so we will add it.

.. code-block:: python

    N_df['equilibrium constant'] = [0.05, 0.02, 0.1, 0.01]

The inputted ``N_df`` is ::

       A  B  C  AB  BB  BC  equilibrium constant
    0 -1  0  1   0   0   0                  0.05
    1  1  1  0  -1   0   0                  0.02
    2  0  2  0   0  -1   0                  0.10
    3  0  1  1   0   0  -1                  0.01


EQTK also does not assume an ordering to the columns. Therefore, the initial concentrations ``c0`` *must* be supplied as a Pandas Series or DataFrame.

.. code-block:: python

    # For a single calculation, a Series
    c0 = pd.Series(data=[1, 1, 0, 0, 0, 0], index=names)

    # For multiple calculations, a DataFrame
    c0 = pd.DataFrame(data=[[1,   0, 0, 0, 0, 0],
                            [1, 0.5, 0, 0, 0, 0],
                            [1,   1, 0, 0, 0, 0]],
                      columns=names)

.. note:: 

    The names of the indices for ``c0`` as a Series and the columns for ``c0`` as a DataFrame are the names of the chemical species, *not*, e.g., ``'[A]__0 (mM)'``. While such input may be convenient, as it allows for specification of units and matching outputs, this is not allowed. The user should explicitly supply the ``units`` keyword argument and ensure that *all* units of concentrations and equilibrium constants are consistent with those concentration units. If the user could specify units within the ``c0`` Series or DataFrame, the equilibrium constants units could be ambiguous, which is why the concentration units may only be specified with the ``units`` keyword argument.

When we call ``eqtk.solve()``, we do not include the argument ``K`` because the equilibrium constants are already included in the inputted data frame. Executing

.. code-block:: python

    c = eqtk.solve(c0=c0, N=N_df, units='mM')
    c[c.columns[~c.columns.str.contains('__0')]]

gives ::

       [A] (mM)  [B] (mM)  [C] (mM)  [AB] (mM)  [BB] (mM)  [BC] (mM)
    0  0.952381  0.000000  0.047619   0.000000   0.000000   0.000000
    1  0.498500  0.017382  0.024925   0.433250   0.003021   0.043325
    2  0.188228  0.077504  0.009411   0.729418   0.060068   0.072942

Free energies as a dictionary of Pandas Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If, however, we wish to input the free energies of the chemical species instead of the equilibrium constants, the ``'equilibrium constant'`` column should not be in the inputted data frame. Again, because no order is assumed in the inputted data frame, ``G`` must be inputted as a Pandas Series with indices given by the names of the chemical species, or as a dictionary with the keys given by the names of the chemical species.

.. code-block:: python

    # Name sure there is no 'equilibrium constant' column in the data frame
    N_df = N_df.drop(columns='equilibrium constant')

    G = pd.Series(data=[G_A, G_B, G_C, G_AB, G_BB, G_BC], index=names)

    c = eqtk.solve(c0=c0, N=N_df, G=G, units='mM')


Summary of I/O using stoichiometric matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table below summarizes the allowed input and output types for ``eqtk.solve()`` when specifying the problem in terms of the stoichiometric matrix :math:`\mathsf{N}`. (The table is wide, so may need to scroll to see the whole table.)

+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+-------------------------------------------------------------+
| ``N`` format                                        | ``K`` format                 | ``G`` format                 | ``c0`` format              | minimal call                 | output type                                                 |
+=====================================================+==============================+==============================+============================+==============================+=============================================================+
| :math:`r\times n` Numpy array                       | length :math:`r` Numpy array | ``None``                     | Numpy array                | ``eqtk.solve(c0, N=N, K=K)`` | Numpy array (Series or DataFrame if ``names`` specified)    |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+-------------------------------------------------------------+
| :math:`r\times n` Numpy array                       | ``None``                     | length :math:`n` Numpy array | Numpy array                | ``eqtk.solve(c0, N=N, G=G)`` | Numpy array    (Series or DataFrame if ``names`` specified) |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+-------------------------------------------------------------+
| DataFrame with ``'equilibrium constant'`` column    | column in ``N`` DataFrame    | ``None``                     | Series, DataFrame, or dict | ``eqtk.solve(c0, N=N)``      | Series or DataFrame                                         |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+-------------------------------------------------------------+
| DataFrame without ``'equilibrium constant'`` column | ``None``                     | Series or dict               | Series, DataFrame, or dict | ``eqtk.solve(c0, N=N, G=G)`` | Series or DataFrame                                         |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+-------------------------------------------------------------+


Conservation matrix A given
---------------------------

Instead of specifying a stoichiometric matrix :math:`\mathsf{N}`, we may specify a conservation matrix :math:`\mathsf{A}`. (`Recall <core_concepts.html#conservation-laws>`_ that :math:`\mathsf{N}` and :math:`\mathsf{A}` are related by :math:`\mathsf{A}^\mathsf{T}\cdot\mathsf{N} = 0`, and we need only specify :math:`\mathsf{N}` *or* :math:`\mathsf{A}`.) If we specify the conservation matrix :math:`\mathsf{A}`, however, we *must* specify the free energies :math:`\mathbf{G}`; the equilibrium constants are ill-defined absent a stoichiometric matrix. Each column of :math:`\mathsf{A}` corresponds to a chemical species. So, entry :math:`j` in :math:`\mathbf{G}` is the free energy of the chemical species corresponding to column :math:`j` of :math:`\mathsf{A}`.

Recall also that all entries of the conservation matrix :math:`\mathsf{A}` `must be nonnegative <core_concepts.html#specification-in-terms-of-conservation-matrices-and-free-energies>`_.

We will use an elemental conservation matrix as an example,

.. math::
  \mathsf{A} =
  \begin{pmatrix}
    \mathrm{A} & \mathrm{B} & \mathrm{C} & \mathrm{AB} & \mathrm{BB} & \mathrm{BC} \\ \hline
    1 & 0 & 1 & 1 & 0 & 1 \\
    0 & 1 & 0 & 1 & 2 & 1
  \end{pmatrix}.


A given as a Numpy array
^^^^^^^^^^^^^^^^^^^^^^^^

If we choose to specify the argument ``A`` for ``eqtk.solve()`` as a Numpy array, ``G`` and ``c0`` must also be specified as Numpy arrays.

.. code-block:: python

    A = np.array([[1, 0, 1, 1, 0, 1],
                  [0, 1, 0, 1, 2, 1]])

    c0 = np.array([1, 1, 0, 0, 0, 0])

    # Use the same G as before

    eqtk.solve(c0=c0, A=A, G=G, units='mM')

The result is as before. ::

    array([0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184])

A two-dimensional ``c0`` has similar behavior as we have seen when ``N`` is specified.

.. code-block:: python

    c0 = np.array([[1,   0, 0, 0, 0, 0],
                   [1, 0.5, 0, 0, 0, 0],
                   [1,   1, 0, 0, 0, 0]])
    eqtk.solve(c0=c0, A=A, G=G, units='mM')

The result is: ::

    array([[0.95238103, 0.        , 0.04761905, 0.        , 0.        , 0.        ],
           [0.49849994, 0.01738215, 0.024925  , 0.43325006, 0.00302139, 0.04332501],
           [0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184]])


If the ``names`` keyword argument is supplied, the ordering of the names must match the ordering of ``G`` and the ordering of the columns in ``N``. The result is then either a Pandas Series (for a single set of initial concentrations), or a Pandas DataFrame (for multiple initial concentrations). Executing

.. code-block:: python

    names = ['A', 'B', 'C', 'AB', 'BB', 'BC']
    c = eqtk.solve(c0=c0, A=A, G=G, units='mM', names=names)
    c[c.columns[~c.columns.str.contains('__0')]]

gives ::

       [A] (mM)  [B] (mM)  [C] (mM)  [AB] (mM)  [BB] (mM)  [BC] (mM)
    0  0.952381  0.000000  0.047619   0.000000   0.000000   0.000000
    1  0.498500  0.017382  0.024925   0.433250   0.003021   0.043325
    2  0.188228  0.077504  0.009411   0.729418   0.060068   0.072942


A given as a Pandas DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can instead specify ``A`` as a Pandas DataFrame, where each column name is the chemical species name. In this case, ``G`` must be given either as a Pandas Series with indices corresponding to the column names of ``A`` or a dictionary with keys corresponding to those of ``A``.  ``c0`` must also be supplied as a dictionary with keys given by the names of the chemical species, a Pandas Series with indices given by the species names, or a Pandas DataFrame with column names given by the species names.

.. code-block:: python

    A = pd.DataFrame(data=np.array([[1, 0, 1, 1, 0, 1],
                                    [0, 1, 0, 1, 2, 1]]),
                     columns=names)

    # Use same G's we calculated before and have been using
    G = pd.Series(data=G, index=names)

    c0 = pd.DataFrame(data=np.array([[1,   0, 0, 0, 0, 0],
                                     [1, 0.5, 0, 0, 0, 0],
                                     [1,   1, 0, 0, 0, 0]]),
                      columns=names)

    c = eqtk.solve(c0=c0, A=A, G=G, units='mM')
    c[c.columns[~c.columns.str.contains('__0')]]

The result is ::

       [A] (mM)  [B] (mM)  [C] (mM)  [AB] (mM)  [BB] (mM)  [BC] (mM)
    0  0.952381  0.000000  0.047619   0.000000   0.000000   0.000000
    1  0.498500  0.017382  0.024925   0.433250   0.003021   0.043325
    2  0.188228  0.077504  0.009411   0.729418   0.060068   0.072942

.. note::

    Unlike in the case with supplying the stoichiometric matrix ``N`` as a DataFrame, in which the equilibrium constants were given in the ``N`` DataFrame, no other information is included in the ``A`` DataFrame. Rather, ``G`` must be given separately.


Summary of I/O using conservation matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table below summarizes the allowed input and output types for ``eqtk.solve()`` when specifying the problem in terms of the conservation matrix :math:`\mathsf{A}`. (The table is wide, so may need to scroll to see the whole table.)

+---------------------------------+------------------------------+----------------------------+------------------------------+----------------------------------------------------------+
| ``A`` format                    | ``G`` format                 | ``c0`` format              | minimal call                 | output type                                              |
+=================================+==============================+============================+==============================+==========================================================+
| :math:`n-r\times n` Numpy array | length :math:`n` Numpy array | Numpy array                | ``eqtk.solve(c0, A=A, G=G)`` | Numpy array (Series or DataFrame if ``names`` specified) |
+---------------------------------+------------------------------+----------------------------+------------------------------+----------------------------------------------------------+
| DataFrame                       | Series or dict               | Series, DataFrame, or dict | ``eqtk.solve(c0, A=A, G=G)`` | Series or DataFrame                                      |
+---------------------------------+------------------------------+----------------------------+------------------------------+----------------------------------------------------------+

.. ``A`` given

.. - ``A`` as Numpy array with ``n`` columns, ``G`` as length ``n`` Numpy array, ``c0`` as Numpy array
.. - ``A`` as DataFrame, ``G`` as Series, ``c0`` as Series or DataFrame



.. The NK formalism
.. ----------------

.. We have a choice of specifying either the stoichiometric matrix :math:`\mathsf{N}` and the equilibrium constants :math:`\mathbf{K}`, or the conservation matrix :math:`\mathsf{A}` and the free energies of the chemical species. First, we will demonstrate how the problem can be formulated with the former, which we will call the NK formalism.

