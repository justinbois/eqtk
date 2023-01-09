.. _eqtk_water_density:

Compute the density of water
============================

When converting between mole fraction and concentrations, it is necessary to multiply by the number density of the solvent. Water is often used as a solvent, and the ``eqtk.water_density()`` function offers calculation of the density of pure water at atmospheric pressure for various temperatures using the `IAPWS-95 standards <https://doi.org/10.1063/1.1461829>`_.

The use specifies the temperature in degrees Kelvin and the units for the density.

>>> eqtk.water_density(T=293.15, units='M')
55.408916789969595
