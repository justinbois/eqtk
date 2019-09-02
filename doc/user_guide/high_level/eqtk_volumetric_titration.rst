.. _eqtk_volumetric_titration:

Volumetric titration
====================

In a `volumetric titration <https://en.wikipedia.org/wiki/Titration>`_, a titrant is added to an existing solution. As each drop of titrant solution is added, a readout, such as pH or fluorescence, of the concentration of the contents of the solution is measured. EQTK offers the ``eqtk.volumetric_titration()`` function to compute titration curves. Its API, as we will demonstrate, is the same as ``eqtk.solve()``, except two additional inputs specifying the concentration of the chemical species in the titrant and the volume of titrant added are required.

In what follows, we will assume that Numpy, Pandas, and EQTK have been imported respectively as ``np``, ``pd``, and ``eqtk``.



Example problem
---------------

To demonstrate the use of ``eqtk.fixed_value_solve()``, we will consider titration of a solution of `oxalic acid <https://en.wikipedia.org/wiki/Oxalic_acid>`_ with a 1 M solution of sodium hydroxide. Oxalic acid is diprotic with the following deprotonation reactions and associated equilibrium constants.

C₂O₄H₂ ⇌ C₂O₄H⁻ + H⁺ ; *K* = 0.0537 M⁻¹

C₂O₄H⁻ ⇌ C₂O₄²⁻ + H⁺ ; *K* = 5.37 × 10⁻⁵ M⁻¹

The initial concentration of oxalic acid is 100 mM.


Stoichiometric matrix and equilibrium constants
-----------------------------------------------

We can conveniently specify the chemical reactions and species using EQTK's parser for string representations. Recall that solvent is **not** explicitly included in the chemical reactions.

.. code-block:: python

    rxns = """
           <=> OH⁻ + H⁺    ; 1e-14
    C₂O₄H₂ <=> C₂O₄H⁻ + H⁺ ; 0.0537
    C₂O₄H⁻ <=> C₂O₄²⁻ + H⁺ ; 5.37e-5
    """

    N = eqtk.parse_rxns(rxns)


Initial concentrations of solution and titrant
----------------------------------------------

The initial concentrations of all species, :math:`\mathbf{c}^0`, is specified in the same way as for ``eqtk.solve()``. For this calculation, we will begin with 0.1 M H₂D.

.. code-block:: python

    c0 = {"C₂O₄H₂": 0.1, "OH⁻": 0, "H⁺": 0, "C₂O₄H⁻": 0, "C₂O₄²⁻": 0}

While it is convenient to specify the concentrations of the species as a dictionary, we could also use a Pandas Series or DataFrame. (We could also specify ``N``, ``K``, and ``c0`` as Numpy arrays.) The specification of inputs is the same as for ``eqtk.solve()``.

We will then titrate in a 1 M NaOH solution. NaOH is effectively completely dissociated in water at one molar concentration, so the titrant is delivering OH⁻. Sodium is inert, so we do not need to include it in our list of chemical species.

.. code-block:: python

    c0_titrant = {"C₂O₄H₂": 0, "OH⁻": 1, "H⁺": 0, "C₂O₄H⁻": 0, "C₂O₄²⁻": 0}

Volume of titrant
-----------------

Finally, we need to specify the volume of titrant we wish to add. This is specified as a Numpy array, where each entry is the volume of the titrant that has been added to the solution *as a fraction of the initial solution volume*. This means that the volume of added titrant is dimensionless.

.. code-block:: python

    vol_titrant = np.linspace(0, 0.5, 400)


Solving for the titration curve
-------------------------------

We call ``eqtk.volumetric_titration()`` similarly to ``eqtk.solve()``, except with the additional arguments ``c0_titrant`` and ``vol_titrant``.

.. code-block:: python

    c = eqtk.volumetric_titration(
        c0=c0, c0_titrant=c0_titrant, vol_titrant=vol_titrant, N=N, units="M"
    )


Because we supplied our inputs as data frames, the output is also a data frame. The output is as for ``eqtk.solve()``, except there is an additional column ``'titrant volume / initial volume'``.

A plot of the titration curve with pH calculated as ``-np.log10(c['[H⁺] (M)'])`` is shown below.


.. bokeh-plot::
    :source-position: none

    import numpy as np
    import pandas as pd
    import eqtk
    import bokeh.plotting
    import bokeh.io
        
    rxns = """
           <=> OH⁻ + H⁺    ; 1e-14
    C₂O₄H₂ <=> C₂O₄H⁻ + H⁺ ; 0.0537
    C₂O₄H⁻ <=> C₂O₄²⁻ + H⁺ ; 5.37e-5
    """

    N = eqtk.parse_rxns(rxns)

    c0 = {"C₂O₄H₂": 0.1, "OH⁻": 0, "H⁺": 0, "C₂O₄H⁻": 0, "C₂O₄²⁻": 0}

    c0_titrant = {"C₂O₄H₂": 0, "OH⁻": 1, "H⁺": 0, "C₂O₄H⁻": 0, "C₂O₄²⁻": 0}

    vol_titrant = np.linspace(0, 0.5, 400)

    c = eqtk.volumetric_titration(
        c0=c0, c0_titrant=c0_titrant, vol_titrant=vol_titrant, N=N, units="M"
    )

    c["pH"] = -np.log10(c['[H⁺] (M)'])

    p = bokeh.plotting.figure(
        height=250,
        width=420,
        y_axis_label="pH",
        x_axis_label="titrant volume / initial volume",
        x_range=[0, 0.5],
    )

    p.line(c["titrant volume / initial volume"], c["pH"], color='#4c78a8', line_width=2)

    bokeh.io.show(p)
