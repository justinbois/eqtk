.. _high_level:

High level interface
====================

EQTK's main high level interface to solving the coupled equilibrium problem is ``eqtk.solve()``. It requires as input initial concentrations :math:`\mathbf{c}^0` and either a stoichiometric matrix :math:`\mathsf{N}` or a conservation matrix :math:`\mathsf{A}`. Depending on the data types of these inputs, other inputs, such as equilibrium constants and free energies, may be required. 

In what follows, we will assume that Numpy, Pandas, and EQTK have been imported respectively as ``np``, ``pd``, and ``eqtk``.



Example problem
---------------

As we demonstrate the usage of ``eqtk.solve()``, it is useful to have an example in mind. We will use the example from the :ref:`core concepts <Core concepts>` part of the user guide. The chemical reactions and associated equilibrium constants for that example are

.. math::
    \begin{array}{lcl}
    \mathrm{A} \rightleftharpoons \mathrm{C} & & K = 0.5\\
    \mathrm{A} + \mathrm{B} \rightleftharpoons \mathrm{AB}& & K = 0.02 \text{ mM}\\
    2\mathrm{B} \rightleftharpoons \mathrm{BB}& & K = 0.1 \text{ mM}\\
    \mathrm{B} + \mathrm{C} \rightleftharpoons \mathrm{BC}& & K = 0.01 \text{ mM}.
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

Note that entry ``K[i]`` corresponds to the chemical reaction given by the ``i``th row of the stoichiometric matrix ``N``.

We can now solve for the equilibrium concentrations

.. code-block:: python

    eqtk.solve(c0=c0, N=N, K=K, units='mM')

The output is a Numpy array containing the equilibrium concentrations preserving the order of the inputs. ::

    array([0.1882283 , 0.07750359, 0.00941142, 0.72941844, 0.06006806, 0.07294184])

Free energies as Numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, we can specify free energies of each species instead of the equilibrium constants for the chemical reactions. In practice, you would enter these directly, but to keep the calculations consistent, we will calculate the free energies, using one of EQTK's private functions to compute the density of water to make the conversion. The resulting free energies are dimensionless (in units of the thermal energy :math:`kT`).

.. code-block:: python

    water_density = eqtk.parsers._water_density(293.15, 'mM')

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

The result is a Pandas DataFrame with columns ::

    ['A__0', 'B__0', 'C__0', 'AB__0', 'BB__0', 'BC__0', 'A', 'B', 'C', 'AB', 'BB', 'BC']

The columns appended with ``__0`` indicate the initial conditions used in the calculation, and the remaining columns indicate the equilibrium concentrations. Executing ``c[names]`` gives ::

              A         B         C        AB        BB        BC
    0  0.952381  0.000000  0.047619  0.000000  0.000000  0.000000
    1  0.498500  0.017382  0.024925  0.433250  0.003021  0.043325
    2  0.188228  0.077504  0.009411  0.729418  0.060068  0.072942



Summary of I/O using stoichiometric matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table below summarized the allowed input and output types for ``eqtk.solve()`` when specifying the problem in terms of the stoichiometric matrix :math:`\mathsf{N}`.

+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+---------------------+
| ``N`` format                                        | ``K`` format                 | ``G`` format                 | ``c0`` format              | minimal call                 | output type         |
+=====================================================+==============================+==============================+============================+==============================+=====================+
| :math:`r\times n` Numpy array                       | length :math:`r` Numpy array | ``None``                     | Numpy array                | ``eqtk.solve(c0, N=N, K=K)`` | Numpy array         |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+---------------------+
| :math:`r\times n` Numpy array                       | ``None``                     | length :math:`n` Numpy array | Numpy array                | ``eqtk.solve(c0, N=N, G=G)`` | Numpy array         |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+---------------------+
| DataFrame with ``'equilibrium constant'`` column    | Series or dict               | None                         | Series or dict             | ``eqtk.solve(c0, N=N, K=K)`` | Series or DataFrame |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+---------------------+
| DataFrame without ``'equilibrium constant'`` column | ``None``                     | Series or dict               | Series, DataFrame, or dict | ``eqtk.solve(c0, N=N, G=G)`` | Series or DataFrame |
+-----------------------------------------------------+------------------------------+------------------------------+----------------------------+------------------------------+---------------------+


.. ``A`` given

.. - ``A`` as Numpy array with ``n`` columns, ``G`` as length ``n`` Numpy array, ``c0`` as Numpy array
.. - ``A`` as DataFrame, ``G`` as Series, ``c0`` as Series or DataFrame



.. The NK formalism
.. ----------------

.. We have a choice of specifying either the stoichiometric matrix :math:`\mathsf{N}` and the equilibrium constants :math:`\mathbf{K}`, or the conservation matrix :math:`\mathsf{A}` and the free energies of the chemical species. First, we will demonstrate how the problem can be formulated with the former, which we will call the NK formalism.

