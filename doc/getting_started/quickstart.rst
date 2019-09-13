.. _quickstart:

Quick start
===========

The first step to getting started is to :ref:`install EQTK <installation>`. After that is done, you can start solving equilibrium problems.

Problem specification
---------------------

EQTK solves the following problem: 

    Given a set of chemical reactions and their equilibrium constants, as well as concentrations of all chemical species initially in a dilute solution, compute the equilibrium concentration of all chemical species.

As an example, consider a ligand A that may bind to either of two receptors, B and C, according to the following chemical reactions with associated equilibrium constants.

AB ⇌ A + B ; K = 0.015 mM

AC ⇌ A + C ; K = 0.003 mM

With the chemical reactions and equilibrium constants defined, we have to further specify what is placed in solution. Imagine we add A, B, and C into the solution such that the initial concentrations of the respective species are 1 mM, 0.5 mM, and 0.25 mM,
and we initially have no AB or AC complexes. The problem is now fully specified.


eqtk.solve()
------------

The ``eqtk.solve()`` function is the central function of EQTK. In the simplest form, it takes three arguments.

- Initial concentrations of all species, `c0`.
- A **stoichiometric matrix**, `N`.
- An array of equilibrium constants, `K`.

In the present example, ``c0 = [1, 0.5, 0.25, 0, 0]``, where we have ordered the species A, B, C, AB, AC.

Entry ``i,j`` of the stoichiometric matrix is the stoichiometric coefficient for species ``j`` in chemical reaction ``i``. The stoichiometric coefficients of reactants are negative and those of products are positive. This is perhaps more clear if we write the chemical reactions in an alternative way::

    A + B     - AB      = 0
    A     + C      - AC = 0

The stoichiometric matrix is then::

    N = [[1,  1,  0, -1,  0],
         [1,  0,  1,  0, -1]]

Finally, the equilibrium constants are ``K = [0.015, 0.003]``.

.. warning::
    The units of the inputted ``c0`` and ``K`` must be consistent, meaning that they both must use the same units for concentration. In this case, the concentration units are millimolar. The units are then specified with ``eqtk.solve()``'s `units` keyword argument.


Now we can solve the system.

.. note:: 

    Because the numerical routines of EQTK are `just in time compiled <http://en.wikipedia.org/wiki/Just-in-time_compilation>`_ (JITted), importing EQTK may take a few seconds, as will the first call you make to ``eqtk.solve()``. Subsequent calculations will be fast.

.. code-block:: python

    import eqtk

    c0 = [1, 0.5, 0.25, 0, 0]

    N = [[1,  1,  0, -1,  0],
         [1,  0,  1,  0, -1]]

    K = [0.015, 0.003]

    c = eqtk.solve(c0, N, K, units="mM")

The resulting ``c`` is given below, with the same units as specified with the ``units`` keyword argument (mM in this case). ::

    array([0.27824281, 0.02557607, 0.00266673, 0.47442393, 0.24733327])


Computing a titration curve
---------------------------

Alternatively, ``c0`` may be inputted as a two-dimensional array, where each row corresponds to a different initial set of concentrations. For example, if we wanted to compute a titration curve for the fraction of the receptors B and C that are bound as we increase the amount of ligand A present in the solution, we can do the following.

.. code-block:: python

    import numpy as np
    import eqtk

    # Set up initial concentrations for titration
    c0 = np.zeros((200, 5))
    c0[:, 0] = np.linspace(0, 2, 200)
    c0[:, 1] = 0.5
    c0[:, 2] = 0.25

    # Stoichiometry matrix
    N = [[1,  1,  0, -1,  0],
         [1,  0,  1,  0, -1]]

    # Equilibrium constants
    K = [0.015, 0.003]

    # Solve!
    c = eqtk.solve(c0, N, K, units="mM")

    # Compute fraction bound
    frac_B_bound = c[:, 3] / c0[:, 1]
    frac_C_bound = c[:, 4] / c0[:, 2]

Here is a plot of the result.

.. bokeh-plot::
    :source-position: none

    import numpy as np
    import eqtk
    import bokeh.plotting
    import bokeh.io

    # Set up initial concentrations for titration
    c0 = np.zeros((200, 5))
    c0[:, 0] = np.linspace(0, 2, 200)
    c0[:, 1] = 0.5
    c0[:, 2] = 0.25

    # Stoichiometry matrix
    N = [[1,  1,  0, -1,  0],
         [1,  0,  1,  0, -1]]

    # Equilibrium constants
    K = [0.015, 0.003]

    # Solve!
    c = eqtk.solve(c0, N, K, units="mM")

    # Compute fraction bound
    frac_B_bound = c[:, 3] / c0[:, 1]
    frac_C_bound = c[:, 4] / c0[:, 2]

    p = bokeh.plotting.figure(
        height=250,
        width=400,
        y_axis_label="fraction bound",
        x_axis_label="[A]₀ (mM)"
    )
    p.line(c0[:,0], frac_B_bound, line_width=2, legend="B")
    p.line(c0[:,0], frac_C_bound, line_width=2, color="orange", legend="C")
    p.legend.location = 'center_right'

    bokeh.io.show(p)



Rich input/output formats
-------------------------

Instead of using lists, tuples, and Numpy arrays for specifying inputs, and thereafter relying on integer-based indexing to retrieve results, the stoichiometry, equilibrium constants, and initial concentrations may be specified as `Pandas <http://pandas.pydata.org>`_ series and data frames. This allows for chemical species to be referenced by name. Conveniently, EQTK includes a parser that converts chemical reactions written a strings to data frames using syntax similar to `Cantera <http://cantera.org>`_. We can alternatively specify the problem as below, this time also considering dimerization of the ligand A, 

AA ⇌ 2A ; K = 0.02 mM.

.. code-block:: python

    import eqtk

    rxns = """
    AB <=> A + B ; 0.015
    AC <=> A + C ; 0.003
    AA <=> 2 A   ; 0.02
    """

    N = eqtk.parse_rxns(rxns)

The variable ``N`` is now a Pandas data frame: ::

        AB    A    B   AC    C   AA  equilibrium constant
    0 -1.0  1.0  1.0  0.0  0.0  0.0                 0.015
    1  0.0  1.0  0.0 -1.0  1.0  0.0                 0.003
    2  0.0  2.0  0.0  0.0  0.0 -1.0                 0.020

The data frame ``N`` now also includes the equilibrium constant for each reaction. This can be passed directly into ``eqtk.solve()``, and specification of ``K`` is no longer necessary, since ``N`` now contains the equilibrium constants.

Because the chemical species now have names, we should pass ``c0`` as a Pandas Series (for a single equilibrium calculation), as a DataFrame (for a titration-like calculations as we did in the last example), or as a dictionary.

.. code-block:: python

    c0 = {"A": 1.0, "B": 0.5, "C": 0.25, "AA": 0, "AB": 0, "AC": 0}

    c = eqtk.solve(c0, N, units="mM")

The resulting ``c`` is a Pandas series. ::

    A__0     1.000000
    B__0     0.500000
    C__0     0.250000
    AA__0    0.000000
    AB__0    0.000000
    AC__0    0.000000
    A        0.055910
    B        0.105768
    C        0.012731
    AA       0.156295
    AB       0.394232
    AC       0.237269
    dtype: float64

The result includes the initial concentrations of each species, with the species names appended with ``__0``.


Next steps
----------

The :ref:`user guide <User guide>` has more details about

- The class of problems EQTK can solve.
- All modes of specifying the problem.
- Lower level interfaces to the equilibrium solving algorithm.

Finally, the :ref:`case studies <Case studies>` section of this guide provides examples of using EQTK to study chemical systems.
