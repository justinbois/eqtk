.. _conventions:

Conventions
===========

Here, we define the conventions for units and solvent properties for ``eqtk.solve()``, EQTK's high level interface for solving couple equilibria. We describe the conventions for the low-level interfaces at the bottom of this page.

Concentration units
^^^^^^^^^^^^^^^^^^^

All inputs and outputs describing concentrations (initial concentrations, equilibrium constants, and equilibrium concentrations) are assumed to be in the same units. These units are then specified with the ``units`` keyword argument. Allowed units are:

- ``None``  (default, specifies that concentrations are dimensionless)
- ``"mole fraction"``
- ``"M"``
- ``"molar"``
- ``"mM"``
- ``"millimolar"``
- ``"µM"``
- ``"uM"`` (synonym for micromolar)
- ``"micromolar"``
- ``"nM"``
- ``"nanomolar"``
- ``"pM"``
- ``"picomolar"``


Energy units
^^^^^^^^^^^^

The free energies :math:`\mathbf{G}` may also carry units, and these must be specified using the ``G_units`` keyword argument of ``eqtk.solve()``. The allowed energy units are:

- ``None``  (default)
- "kcal/mol"
- "J"
- "J/mol"
- "kJ/mol"
- "pN-nm"

If ``G_units`` is ``None``, the inputted free energies are assumed to be in units of the thermal energy, :math:`kT`, where :math:`k` is the Boltzmann constant. If ``G_units`` are specified, the user should be sure to properly set the value of the ``T`` keyword argument, which gives the temperature of the solution in Kelvin. The default is ``T=293.15``.



It takes as required input the initial concentrations :math:`\mathbf{c}^0`. The user is also required to specify the stoichiometric matrix :math:`\mathsf{N}` and the equilibrium constants :math:`\mathbf{K}`, or the conservation matrix :math:`\mathsf{A}` and the free energies of all chemical species :math:`\mathbf{G}`.

Assumptions about the solvent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, EQTK assumes the solvent is water at atmospheric pressure and a temperature of 293.15 K. The temperature is adjusted using the ``T`` keyword argument, which allows specification of the temperature in Kelvin.

If the solvent is not water at atmospheric pressure, the user must specify the **number density** (*not* mass density) of the solvent in the same concentration units as specified by the ``units`` keyword argument. For example, at atmospheric pressure and room temperature, :math:`\rho_\mathrm{H_2O} \approx 55` moles per liter. 

Note that if the concentrations are given as mole fractions (``units=None``), then the solvent density is not necessary.


Conventions for low-level interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-level interfaces to the equilibrium solver, ``eqtk.solveNK()``, ``eqtk.solveNG()``, and ``eqtk.solveAG()``, use only dimensionless concentration and energy units. This means that the concentrations are all mole fractions and equilibrium constants are dimenesionless. All free energies are in units of the thermal energy :math:`kT`, where `k` is the Boltzmann constant and :math:`T` is the temperature.
