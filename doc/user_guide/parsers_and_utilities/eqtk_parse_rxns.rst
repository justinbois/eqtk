.. _eqtk_parse_rxns:

Parse reactions given as strings
================================

An equilibrium problem is specified using a stoichiometric matrix :math:`\mathsf{N}`, a set of equilibrium constants :math:`\mathbf{K}`, and a set of initial concentations of chemical species :math:`\mathbf{c}^0`. Directly specifying the stoichiometric matrix may be difficult even for systems of chemical reactions that contain less than 5 species. Indeed, many users first think of the chemical reactions as traditionally written, for example like

AB ⇌ A + B,

and then convert them to stoichiometric matrices.

The ``eqtk.parse_rxns()`` function enables conversion of reactions specified as strings to stoichiometric matrices stored as Pandas_ DataFrames.

The syntax for entering chemical reaction systems as strings is similar to that of Cantera_. Specifically,

- The chemical equality operator is defined by ``<=>`` or ``⇌`` and must be preceded and followed by whitespace.
- The chemical ``+`` operator must be preceded and followed by whitespace.
- Stoichiometric coefficients are followed by a space.
- Each chemical reaction appears on its own line.

Here are some examples.

+----------------------+------------------------------+
| Reaction             | Representation               |
+======================+==============================+
| AB ⇌ A + B           | ``"AB <=> A + B"``           |
+----------------------+------------------------------+
| C₂O₄H₂ ⇌ C₂O₄H⁻ + H⁺ | ``"C₂O₄H₂ <=> C₂O₄H⁻ + H⁺"`` |
+----------------------+------------------------------+
| 2A + 3B ⇌ A₂B₃       | ``"2 A + 3 B <=> A₂B₃"``     |
+----------------------+------------------------------+


As an example, consider a system where two ligands, A and B, can each bind either of two receptors, R and S. There is a third receptor, T, that binds two A's or two B's.

.. code-block:: python

    rxns = """
    AR <=> A + R
    AS <=> A + S
    BR <=> B + R
    BS <=> B + S
    ATA <=> 2 A + T
    BTB <=> 2 B + T
    """

We can then call

.. code-block:: python

    eqtk.parse_rxns(rxns)

to give a DataFrame with the stoichiometric matrix. The result is ::

        AR    A    R   AS    S   BR    B   BS  ATA    T  BTB
    0 -1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    1  0.0  1.0  0.0 -1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
    2  0.0  0.0  1.0  0.0  0.0 -1.0  1.0  0.0  0.0  0.0  0.0
    3  0.0  0.0  0.0  0.0  1.0  0.0  1.0 -1.0  0.0  0.0  0.0
    4  0.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0 -1.0  1.0  0.0
    5  0.0  0.0  0.0  0.0  0.0  0.0  2.0  0.0  0.0  1.0 -1.0


We can also specify equilibrium constants in our reactions. The equilibrium constants are separated from the reaction they are associated with by a semicolon.

.. code-block:: python

    rxns = """
    AR <=> A + R    ; 1.5
    AS <=> A + S    ; 3.4e-5
    BR <=> B + R    ; 0.03 
    BS <=> B + S    ; 0.0045
    ATA <=> 2 A + T ; 1.9e2
    BTB <=> 2 B + T ; 2.34
    """

    eqtk.parse_rxns(rxns)

The result is ::

        AR    A    R   AS    S   BR    B   BS  ATA    T  BTB  equilibrium constant
    0 -1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0              1.500000
    1  0.0  1.0  0.0 -1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0              0.000034
    2  0.0  0.0  1.0  0.0  0.0 -1.0  1.0  0.0  0.0  0.0  0.0              0.030000
    3  0.0  0.0  0.0  0.0  1.0  0.0  1.0 -1.0  0.0  0.0  0.0              0.004500
    4  0.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0 -1.0  1.0  0.0            190.000000
    5  0.0  0.0  0.0  0.0  0.0  0.0  2.0  0.0  0.0  1.0 -1.0              2.340000


Note that the equilibrium constants were added to a column ``"equilibrium constant"``. The units of the equilibrium constants are left unspecified in the ``rxns`` string and in the call to ``eqtk.parse_rxns()``, but must be consistent with each other and furthermore consistent with those given by ``c0`` in a call to ``eqtk.solve()`` or other high-level function.

Alternatively, you can have a column ``"log equilibrium constant"``, which contains the natural logarithm of the equilibrium constants (which must be dimensionless if the log equilibrium constant is specified).

.. _Pandas: http://pandas.pydata.org
.. _Cantera: http://cantera.org