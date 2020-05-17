.. _eqtk_parse_input:

Parse input to high-level interface
===================================

The high level interface functions (``eqtk.solve()``, ``eqtk.fixed_value_solve()``, and ``eqtk.volumetric_titration()``) take arguments as a variety of data types, including ``None``, but the low-level function require strict typing. All three high-level functions take as arguments

- ``c0``, initial concentrations
- ``N``, stoichiometric matrix
- ``K``, equilibrium constants
- ``logK``, natural logarithm of dimensionless equilibrium constants
- ``A``, conservation matrix
- ``G``, free energies of chemical species
- ``names``, names of chemical species
- ``units``, units of concentration
- ``G_units``, units of energy
- ``solvent_density``, number density of solvent
- ``T``, temperature in units of Kelvin

The low-level functions take the natural logarithm of dimensionless ``K`` as an argument.
Additionally, ``c0`` and ``G`` must be dimensionless. The variables must have the following data types.

- ``x0``: Numpy array, dtype float, shape (n_points, n_compounds)
- ``N``: Numpy array, dtype float, shape (n_reactions, n_compounds)
- ``logK``: Numpy array, dtype float, shape (n_reactions,)
- ``A``: Numpy array, dtype float, shape (n_conserv_laws, n_compounds)
- ``G``: Numpy array, dtype float, shape (n_compounds,)
- ``names``: list of strings, len n_compounds
- ``solvent_density``: float

Once everything is converted to dimensionless units and the solvent density is computed, ``units``, ``G_units``, and ``T`` are no longer needed. 

The function ``eqtk.parse_input()`` converts input to these data types for use in lower level function. For most applications, users use the high-level functions and do not need to directly call ``eqtk.parse_input()``, but this function can be useful if you intend to use the low-level interfaces, for example for repeated calculations, and want to appropriately prepare their input. 

The ``eqtk.parse_input()`` function has one additional return value. For formatting the output in the high-level interface, it is also useful to know if there is a single calculation to be done, or if multiple sets of initial concentrations are given. ``eqtk.parse_input()`` also returns a boolean that is True if only one calculation is to be computed.

We show as an example the dissociation of oxalic acid in presence of hydroxide.

.. code-block:: python

    rxns = """
           <=> OH⁻ + H⁺    ; 1e-14
    C₂O₄H₂ <=> C₂O₄H⁻ + H⁺ ; 0.0537
    C₂O₄H⁻ <=> C₂O₄²⁻ + H⁺ ; 5.37e-5
    """

    c0 = {"C₂O₄H₂": 0.1, "OH⁻": 0, "H⁺": 0, "C₂O₄H⁻": 0, "C₂O₄²⁻": 0}

    # We could first convert rxn's to N using eqtk.parse_rxns(rxns), 
    # but don't have to.

    x0, N, logK, A, G, names, solvent_density, single_point = eqtk.parse_input(
        c0=c0,
        N=rxns,
        K=None,
        logK=None,
        A=None,
        G=None,
        names=None,
        units='M',
        solvent_density=None,
        T=293.15,
        G_units=None,
    )

    print('x0:\n', x0, '\n')
    print('N:\n', N, '\n')
    print('logK:\n', logK, '\n')
    print('A:\n', A, '\n')
    print('G:\n', G, '\n')
    print('names:\n', names, '\n')
    print('solvent_density:\n', solvent_density, '\n')
    print('single_point:\n', single_point)

The result is ::

    x0:
     [[0.00180476 0.         0.         0.         0.        ]] 

    N:
     [[ 0.  1.  1.  0.  0.]
     [-1.  0.  1.  1.  0.]
     [ 0.  0.  1. -1.  1.]] 

    logK:
     [-40.26567237  -6.93908281 -13.84683809] 

    A:
     None 

    G:
     None 

    names:
     ['C₂O₄H₂', 'OH⁻', 'H⁺', 'C₂O₄H⁻', 'C₂O₄²⁻'] 

    solvent_density:
     55.408916789969595 

    single_point:
     True
