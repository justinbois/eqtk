.. _eqtk_fixed_value_solve:

Equilibrium with some concentrations fixed
==========================================

Sometimes we may have a chemical system in which one or more species has their concentration held constant. For example, we could have a system where a chemical species is constantly fed in such that its concentration in the reaction vessel remains constant. As another example, we might have a weak acid in solution and we add a strong acid to control the pH, thereby keeping the concentration of hydrogen ions constant.

The ``eqtk.final_value_solve()`` function allows for solving the coupled equilibrium problem when one or more of of chemical species are present in fixed amounts. Its API, as we will demonstrate, is the same as ``eqtk.solve()``, except an additional input, ``fixed_c``, which specifies the fixed concentration is required. However, ``eqtk.final_value_solve()`` only accepts input of the stoichiometric matrix and equilibrium constants; input of free energies and a conservation matrix are not yet implemented.

In what follows, we will assume that Numpy, Pandas, and EQTK have been imported respectively as ``np``, ``pd``, and ``eqtk``.



Example problem
---------------

To demonstrate the use of ``eqtk.fixed_value_solve()``, we will consider the protonation state of `aspartic acid <https://en.wikipedia.org/wiki/Aspartic_acid>`_. The reactions describing deprotonation are

H₃D⁺ ⇌ H₂D + H⁺

H₂D ⇌ HD⁻ + H⁺

HD⁻ ⇌ D²⁻ + H⁺

The respective pKₐ's for the reaction are 1.99, 3.9, and 10.002. We additionally need to take into account the dissociation of water,

H₂O ⇌ OH⁻ + H⁺,

which has an equilibrium constant of 10⁻¹⁴ M².

Our goal is to compute the relative abundances of the four protonation states of aspartic acid as a function of pH. Thus, for each pH value, we want to fix [H⁺] = 10⁻ᵖᴴ and compute the equilibrium concentrations of the other species.


The chemical reactions
----------------------

We can conveniently specify the chemical reactions and species using EQTK's parser for string representations. Recall that solvent is **not** explicitly included in the chemical reactions.

.. code-block:: python

    rxns = """
         <=> OH⁻ + H⁺  ; 1e-14
    H₃D⁺ <=> H₂D + H⁺  ; {Ka1}
    H₂D  <=> HD⁻ + H⁺  ; {Ka2}
    HD⁻  <=> D²⁻ + H⁺  ; {Ka3}
    """.format(Ka1=10**(-1.99), Ka2=10**(-3.9), Ka3=10**(-10.002))

    N = eqtk.parse_rxns(rxns)


Initial and fixed concentrations
--------------------------------

The initial concentrations of all species, $\mathbf{c}^0$, is specified the same as for ``eqtk.solve()``. For this calculation, we will begin with 1 mM H₂D.

.. code-block:: python

    c0 = {
        "H⁺":   0,
        "OH⁻":  0,
        "H₃D⁺": 0,
        "H₂D":  0.001,
        "HD⁻":  0,
        "D²⁻":  0,
    }

While it is convenient to specify the concentrations of the species as a dictionary, we could also use a Pandas Series or DataFrame. (We could also specify ``N``, ``K``, and ``c0`` as Numpy arrays.) The specification of inputs is the same as for ``eqtk.solve()``.

We need to further specify the fixed concentrations. We will consider pH ranging from zero to 14, with 400 points in between. To set up the fixed concentrations, we create a DataFrame with six columns, one for each species, with 400 rows, one for each pH value we want to consider. Entries in the data frame that are either negative or `np.nan` denote concentrations that are **not** fixed. So, we initialize the data frame with negative ones. Following that, we fill the column for H⁺ with our desired fixed values.

.. code-block:: python

    fixed_c = pd.DataFrame(data=-np.ones((400, 6)), columns=c0.keys())

    pH = np.linspace(0, 14, 400)
    fixed_c["H⁺"] = 10**(-pH)

The resulting ``fixed_c`` DataFrame has the following first five rows. ::

             H⁺  OH⁻  H₃D⁺  H₂D  HD⁻  D²⁻
    0  0.100000 -1.0  -1.0 -1.0 -1.0 -1.0
    1  0.082386 -1.0  -1.0 -1.0 -1.0 -1.0
    2  0.066777 -1.0  -1.0 -1.0 -1.0 -1.0
    3  0.053177 -1.0  -1.0 -1.0 -1.0 -1.0
    4  0.041543 -1.0  -1.0 -1.0 -1.0 -1.0


Solving for the relative abundance
----------------------------------

We call ``eqtk.fixed_value_solve()`` similarly to ``eqtk.solve()``, except with the additional argument ``fixed_c``.

.. code-block:: python

    c = eqtk.fixed_value_solve(c0=c0, fixed_c=fixed_c, N=N, units='M')

Because we supplied our inputs as data frames, the output is also a data frame. The output is as for ``eqtk.solve()``, except there are additional columns with names like ``'[H⁺]__fixed (M)'``, denoting concentrations that were fixed in the calculation.

A plot of the relative abundances computed from ``c`` is shown below.


.. bokeh-plot::
    :source-position: none

    import numpy as np
    import pandas as pd
    import eqtk
    import bokeh.plotting
    import bokeh.io

    rxns = """
         <=> OH⁻ + H⁺  ; 1e-14
    H₃D⁺ <=> H₂D + H⁺  ; {Ka1}
    H₂D  <=> HD⁻ + H⁺  ; {Ka2}
    HD⁻  <=> D²⁻ + H⁺  ; {Ka3}
    """.format(Ka1=10**(-1.99), Ka2=10**(-3.9), Ka3=10**(-10.002))

    N = eqtk.parse_rxns(rxns)

    c0 = {
        "H⁺":   0,
        "OH⁻":  0,
        "H₃D⁺": 0,
        "H₂D":  0.001,
        "HD⁻":  0,
        "D²⁻":  0,
    }

    fixed_c = pd.DataFrame(data=-np.ones((400, 6)), columns=c0.keys())

    pH = np.linspace(0, 14, 400)
    fixed_c["H⁺"] = 10**(-pH)

    c = eqtk.fixed_value_solve(c0=c0, fixed_c=fixed_c, N=N, units='M')

    c['pH'] = -np.log10(c['[H⁺] (M)'])
    c['H₃D⁺'] = c['[H₃D⁺] (M)'] / c0["H₂D"]
    c['H₂D'] = c['[H₂D] (M)'] / c0["H₂D"]
    c['HD⁻'] = c['[HD⁻] (M)'] / c0["H₂D"]
    c['D²⁻'] = c['[D²⁻] (M)'] / c0["H₂D"]

    p = bokeh.plotting.figure(
        height=250,
        width=420,
        y_axis_label="relative abundance",
        x_axis_label="pH",
        x_range=[0, 14],
    )

    p.line(c['pH'], c['H₃D⁺'], color='#4c78a8', line_width=2, legend_label="H₃D⁺")
    p.line(c['pH'], c['H₂D'], color='#f58518', line_width=2, legend_label="H₂D")
    p.line(c['pH'], c['HD⁻'], color='#e45756', line_width=2, legend_label="HD⁻")
    p.line(c['pH'], c['D²⁻'], color='#72b7b2', line_width=2, legend_label="D²⁻")

    p.legend.location = 'center_right'

    bokeh.io.show(p)


Over-constraining a problem
---------------------------

In a given calculation, only some species may have fixed concentration, lest the coupled equilibrium problem be over-constrained. EQTK will check for this and raise an exception if the problem becomes over-constrained. For example, we cannot fix both the H⁺ and OH⁻ concentrations.

.. code-block:: python

    fixed_c = {
        "H⁺":   0.01,
        "OH⁻":  0.0001,
        "H₃D⁺": -1,
        "H₂D":  -1,
        "HD⁻":  -1,
        "D²⁻":  -1,
    }

    c = eqtk.fixed_value_solve(c0=c0, fixed_c=fixed_c, N=N, units='M')

This results in a ``ValueError`` saying that the stoichiometric matrix is rank deficient.