High-level interfaces
=====================

EQTK offers three high-level interfaces for solving couple equilibria.

.. 1. ``eqtk.solve()`` is EQTK's main high level interface; typically all calculations use this function.
.. 2. ``eqtk.fixed_value_solve()`` allows for solving equilibria where the concentration(s) of one or more species is fixed.
.. 3. ``eqtk.volumetric_titration()`` allows for computing titration curves as a function of volume of added titrant.


.. toctree::
   :maxdepth: 1

   eqtk.solve() <eqtk_solve> 
   eqtk.fixed_value_solve() <eqtk_fixed_value_solve> 