.. _core_concepts:

Core concepts
=============

Here, we go over the core concepts behind the problem that EQTK solves and the approach. It is advisable to read this section before proceeding through the user guide and looking at test cases because this section introduces key terminology and gives the basis for the naming convention for the arguments of EQTK's functions. Importantly, you should understand the meanings of the terms

- Stoichiometric matrix,
- Equilibrium constants,
- Initial concentrations,
- Conservation matrix,
- Free energy of a chemical species.

Problem definition
------------------

EQTK calculates the steady state of a closed dilute solution containing reacting chemical species in cases where all chemical reactions are reversible (finite equilibrium constants). By "closed," we mean that no material may flow in to our out of the solution. Energy may in general enter or exit. In the case where the system is also closed to energy, the steady state is an equilibrium. We will use the terms equilibrium and steady state interchangeably going forward.

Importantly, **all chemical reactions must be reversible.** EQTK does *not* solve steady states for chemical reaction systems in which one or more reaction is irreversible. Furthermore, as we will discuss below, the chemical kinetics of all reactions must be governed by the `law of mass action`_, which implies that all assumptions underlying the law of mass action apply to systems that EQTK handles.


The stoichiometric matrix
-------------------------

To specify the problem, we define a set of :math:`n` chemical species and a set of :math:`r` reversible chemical reactions. Let :math:`S_j` be the symbol
for chemical species :math:`j`.  Then, chemical reaction :math:`i` is represented as

.. math::

  \sum_{j} \nu_{ij}\,S_j = 0,


where :math:`\nu_{ij}` is the stoichiometric coefficient of species :math:`j` in
reaction :math:`r`.  We can assemble the :math:`\nu_{ij}` into an ::math`r \times
n` **stoichiometric matrix** :math:`\mathsf{N}`.  Each row of :math:`\mathsf{N}` represents a single chemical reaction.

As an illustrative example, consider a chemical reaction system where species A, B, and C may bind each other to form dimers AB, BB, and BC, and A and C may be interconverted via the chemical reactions

.. math::
	&\mathrm{A} \rightleftharpoons \mathrm{C}\\
	&\mathrm{A} + \mathrm{B} \rightleftharpoons \mathrm{AB}\\
	&2\mathrm{B} \rightleftharpoons \mathrm{BB}\\
	&\mathrm{B} + \mathrm{C} \rightleftharpoons \mathrm{BC}.

We can represent these reactions with the stoichiometric coefficients as defined above.

.. math::
	\begin{array}{rrrrrrcr}
	&-\mathrm{A} &  & + \mathrm{C} & & &  & = & 0 \\	
	&-\mathrm{A} & -\mathrm{B} &  & + \mathrm{AB} &  &  & = & 0 \\
	& & -2\mathrm{B} & & & + \mathrm{BB} &  & = & 0 \\
	& & -\mathrm{B} & - \mathrm{C} &  &  & + \mathrm{BC} & = &0.
	\end{array}


Then, the stoichiometric matrix is

.. math::

	\mathsf{N} =
	\begin{pmatrix}
	-1 & 0 & 1 & 0 & 0 & 0 \\
	-1 & -1 & 0 & 1 & 0 & 0 \\
	0 & -2 & 0 & 0 & 1 & 0 \\
	0 & -1 & -1 & 0 & 0 & 1
	\end{pmatrix}.

Note that we have defined each row of :math:`\mathsf{N}` to describe both
the forward and reverse reaction. This is not to be confused with a like-named matrix
:math:`\mathsf{S}` from the systems biology literature (see, e.g., `Palsson's systems biology book`_) that can be constructed from :math:`\mathsf{N}` as

.. math::
  \mathsf{S} = \left(\mathsf{N}^\mathsf{T} \; | \; -\mathsf{N}^\mathsf{T}\right).


For a properly posed problem, we stipulate that :math:`N` must have full row rank. Put simply, this means that each species must be accounted for in at least one chemical reaction and that the set of chemical reactions is minimal, meaning that no given reaction can be written as a combination of any other two or more. In the example, we would not consider the chemical reaction

.. math::

	2 \mathrm{A} + \mathrm{B} \rightleftharpoons \mathrm{AB} + \mathrm{C}

because it is a linear combination of the first two reactions. Along with the requirement that the :math:`N` have full row rank is the requirement that :math:`n \ge r`.


Equilibrium constants
---------------------

For a set of chemical reactions, we define the net rates of the chemical reactions by the :math:`r`-vector :math:`\mathbf{v}`.  The dynamics of the concentrations of the chemical species, denoted by the :math:`n`-vector :math:`\mathbf{c}`, evolve according to

.. math::
  \frac{\mathrm{d}\mathbf{c}}{\mathrm{d}t} = \mathsf{N}^\mathsf{T} \cdot \mathbf{v},

For dilute solutions, the law of mass action gives

.. math::
  v_i = k_i^+ \prod_{\substack{j \\ \nu_{ij} < 0}} c_j^{|\nu_{ij}|}
  - k_i^-  \prod_{\substack{j \\ \nu_{ij} > 0}} c_j^{|\nu_{ij}|},

The first term is the rate of the forward reaction, and the second is the rate of the reverse reaction.  The parameters :math:`k_i^+` and :math:`k_i^-` are the forward and reverse rate constants, respectively, and are a function only of temperature and pressure.  At steady state, :math:`\dot{\mathbf{c}} = \mathbf{0}`, so :math:`\mathbf{v} = \mathbf{0}`.  Thus, at steady state :math:`\mathbf{c}` must satisfy

.. math::
  \prod_{j} c_j^{\nu_{ij}} = \frac{k_i^+}{k_i^-} \equiv K_i \;\forall j

:math:`K_i` is commonly called the **equilibrium constant**, though strictly
speaking may describe a steady state that is not necessarily at
equilibrium, as would be the case for a system that consumes or
produces energy through its reactions. We can write the equilibrium expression in a more compact form.

.. math::
  \ln \mathbf{K} = \mathsf{N} \cdot \ln \mathbf{c}.

Thus, to compute the steady state, the :math:`r`-vector :math:`\mathbf{K}` containing
the equilibrium constants must be specified.


Conservation law
----------------

If :math:`\mathsf{N}` is square (:math:`n = r`), then the equilibrium concentrations are immediately attained by solving the linear system

.. math::
  \ln \mathbf{K} = \mathsf{N} \cdot \ln \mathbf{c}.

This is almost never the case; in most applications there are more chemical species than there are reactions, and :math:`n > r`. The equilibrium expression is then underdetermined, and we need :math:`n - r` additional equations to solve for the concentrations.

Let us assume that we initially have concentrations :math:`\mathbf{c}^0$ of chemical species in our dilute solution. We refer to the :math:`n`-vector :math:`\mathbf{c}^0` as the **initial concentrations**. There exists a **conservation matrix** :math:`\mathsf{A}` such that

.. math::
	\mathsf{A} \cdot \mathbf{c} = \mathsf{A} \cdot \mathbf{c}^0.

The rows of the conservation matrix :math:`\mathsf{A}` span the null space of the stoichiometric matrix :math:`\mathsf{N}` such that

.. math::
	\mathsf{A}\cdot\mathsf{N}^\mathsf{T} = \mathsf{0}.

We can see where the conservation matrix gets its name by left-multiplying the kinetics differential equation by :math:`\mathsf{A}`.

.. math::
	\mathsf{A}\cdot\frac{\mathrm{d}\mathbf{c}}{\mathrm{d}t} = \frac{\mathrm{d}}{\mathrm{d}t}\,\mathsf{A}\cdot\mathbf{c} =  \mathsf{A}\cdot \mathsf{N}^\mathsf{T} \cdot \mathbf{v} = \mathbf{0}.

Therefore, the quantity :math:`\mathsf{A}\cdot \mathbf{c}` is conserved. Thus, we have a complete system of equations to specify equilibrium,

..  math::
	&\ln \mathbf{K} = \mathsf{N} \cdot \ln \mathbf{c}, \\
	&\mathsf{A} \cdot \mathbf{c} = \mathsf{A} \cdot \mathbf{c}^0.




.. _law of mass action: http://en.wikipedia.org/wiki/Law_of_mass_action
.. _Palsson's systems biology book: https://doi.org/10.1017/CBO9781139854610.012

