.. _low_level_interface:

Low-level interfaces
=========================


EQTK offers low-level interfaces for solving coupled equilibria. Almost all users should use the high-level interfaces, which almost always means using ``eqtk.solve()``. The high-level functions allow different formats for input and output and handles all necessary parsing before executing the lower-level functions. The low-level functions are just-in-time compiled (JITted), and therefore are strict in the data types they expect as input and offer as output. The low-level function are useful for users who wish to incorporate equilibrium solving in a larger, possibly also JITted , pipeline.

Importantly, all low-level functions require inputs that are contiguous Numpy arrays in C order with a float data type (see the `Numpy docs <https://docs.scipy.org/doc/numpy/reference/internals.html>`_). The first argument for each function is ``x0``, the *dimensionless* initial concentration of all species. It is a 2D array, where each row corresponds to a set of initial concentrations for which a calculation is to be done. For a single set of initial concentrations, it still must be a 2D array, in that case with only one row.

The details about the remaining two arguments for the low-level functions are given in their individual descriptions, below. It is important to note that all concentrations, equilibrium constants, and free energies must be dimensionless. The return value of all low-level functions is the *natural logarithm* of the dimensionless equilibrium concentrations, returned as two-dimensional Numpy arrays the same shape as ``x0``.

Example problem
---------------

To demonstrate the use of the low-level interface, we will use the same example problem we used in the documentation of the :ref:`high-level interface <eqtk_solve>`. The chemical reactions and equilibrium constants are

.. math::
    \begin{array}{lcl}
    \mathrm{A} \rightleftharpoons \mathrm{C} & & K = 0.5\\
    \mathrm{AB} \rightleftharpoons \mathrm{A} + \mathrm{B} & & K = 0.02 \text{ mM}\\
    \mathrm{BB} \rightleftharpoons 2\mathrm{B}& & K = 0.1 \text{ mM}\\
    \mathrm{BC} \rightleftharpoons \mathrm{B} + \mathrm{C}& & K = 0.01 \text{ mM}.
    \end{array}

The stoichiometric matrix is

.. math::
  \mathsf{N} =
  \begin{pmatrix}
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
    1 & 0 & 1 & 1 & 0 & 1 \\
    0 & 1 & 0 & 1 & 2 & 1
  \end{pmatrix}.

To have them available, we will create them a Numpy arrays. The arrays are *not* the format the low-level interface functions require, but we can use `eqtk.parse_input()` to convert them to the appropriate format.

.. code-block:: python

    N = np.array([[-1,  0,  1,  0,  0,  0],
                  [ 1,  1,  0, -1,  0,  0],
                  [ 0,  2,  0,  0, -1,  0],
                  [ 0,  1,  1,  0,  0, -1]])

    c0 = np.array([1, 1, 0, 0, 0, 0])

    K = np.array([0.5, 0.2, 0.1, 0.01])

    x0, N, logK, _, _, _, _, _ = eqtk.parse_input(c0=c0, N=N, K=K, units="mM")

The results are:

``x0`` ::

    array([[1.80476367e-05, 1.80476367e-05, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00]])

``N`` ::

    array([[-1.,  0.,  1.,  0.,  0.,  0.],
           [ 1.,  1.,  0., -1.,  0.,  0.],
           [ 0.,  2.,  0.,  0., -1.,  0.],
           [ 0.,  1.,  1.,  0.,  0., -1.]])

``logK`` ::

    array([ -0.69314718, -12.53193373, -13.22508091, -15.527666  ])


Alternatively, we could create them by hand, ensuring the correct format.

.. code-block:: python

    N = np.array([[-1,  0,  1,  0,  0,  0],
                  [ 1,  1,  0, -1,  0,  0],
                  [ 0,  2,  0,  0, -1,  0],
                  [ 0,  1,  1,  0,  0, -1]], dtype=float)

    solvent_density = eqtk.water_density(T=293.15, units="mM")

    x0 = np.array([1, 1, 0, 0, 0, 0], dtype=float) / solvent_density
    x0 = x0.reshape((1, len(x0)))

    K = np.array([0.5, 0.2, 0.1, 0.01])
    K[1:] /= solvent_density
    logK = np.log(K)

Finally, we will need values of ``A`` and ``G`` for some of the calculations. To keep the results consistent, we will calculate them now.

.. code-block:: python

    A = np.array([[1, 0, 1, 1, 0, 1],
                 [0, 1, 0, 1, 2, 1]], dtype=float)

    G_A = 0
    G_B = 0
    G_C = -logK[0]
    G_AB = logK[1]
    G_BB = logK[2]
    G_BC = logK[3] + G_C

    G = np.array([G_A, G_B, G_C, G_AB, G_BB, G_BC])


Solve with N and K specified
----------------------------

For a problem where the stoichiometric matrix :math:`\mathsf{N}` and the equilibrium constants :math:`\mathbf{K}` are specified, use eqtk.solveNK()`. The first argument is ``x0``, described above. The second argument is the stoichiometric matrix ``N``, as a 2D Numpy array of *floats*. The third argument is a 1D Numpy array of the *natural logarithm* of the equilibrium constants with a float data type.

.. code-block:: python

    eqtk.solveNK(x0, N, logK)


The result is the natural logarithm of the dimensionless concentrations, ::

    array([[-12.76103925, -13.36384703, -13.45418643, -13.59295256,
            -13.50261316, -11.29036747]])

It is important to note that restrictions on ``N`` and ``logK`` hold. All entries must be finite, and ``N`` must have full row rank. In the low-level interface, **these are not checked.**


Solve with N and G specified
----------------------------

For a problem where the stoichiometric matrix :math:`\mathsf{N}` and the free energies of the chemical species, :math:`\mathbf{G}`, are specified, use eqtk.solveNG()`. The first argument is ``x0``, described above. The second argument is the stoichiometric matrix ``N``, as a 2D Numpy array of *floats*. The third argument is a 1D Numpy array free energies with a float data type.

.. code-block:: python

    eqtk.solveNG(x0, N, G)

The result is the natural logarithm of the dimensionless concentrations, ::

    array([[-12.76103925, -13.36384703, -13.45418643, -13.59295256,
            -13.50261316, -11.29036747]])

Again, ``N`` and ``G`` are subject to restrictions. All entries must be finite, and ``N`` must have full row rank. In the low-level interface, **these are not checked.**


Solve with A and G specified
----------------------------

For a problem where the conservation matrix :math:`\mathsf{A}` and the free energies of the chemical species, :math:`\mathbf{G}`, are specified, use eqtk.solveAG()`. The first argument is ``x0``, described above. The second argument is the conservation matrix ``A``, as a 2D Numpy array of *floats*. The third argument is a 1D Numpy array free energies with a float data type.

.. code-block:: python

    eqtk.solveAG(x0, A, G)

The result is the natural logarithm of the dimensionless concentrations, ::

    array([[-12.76103925, -13.36384703, -13.45418643, -13.59295256,
            -13.50261316, -11.29036747]])

``A`` and ``G`` are subject to restrictions. All entries must be nonnegative and finite, and ``A`` must have full row rank. In the low-level interface, **these are not checked.**
