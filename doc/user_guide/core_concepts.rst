.. _core_concepts:

Core concepts
=============

Here, we go over the core concepts behind the problem that EQTK solves and the approach. It is advisable to read this section before proceeding through the user guide and looking at test cases because this section introduces key terminology and gives the basis for the naming convention for the arguments of EQTK's functions. Importantly, you should understand the meanings of the terms

- Stoichiometric matrix,
- Equilibrium constants,
- Number density of solvent,
- Initial concentrations,
- (Elemental) conservation matrix,
- Free energy of a chemical species.

Problem definition
------------------

EQTK calculates the steady state of a closed dilute solution containing reacting chemical species in cases where all chemical reactions are reversible (finite equilibrium constants). By "closed," we mean that no material may flow into or out of the solution. Energy may in general enter or exit. In the case where the system is also closed to energy, the steady state is an equilibrium. We will use the terms equilibrium and steady state interchangeably going forward.

Importantly, **all chemical reactions must be reversible.** EQTK does *not* solve steady states for chemical reaction systems in which one or more reaction is irreversible. Furthermore, as we will discuss below, the chemical kinetics of all reactions must be governed by the `law of mass action`_, which implies that all assumptions underlying the law of mass action apply to systems that EQTK handles. Note, though, that EQTK does *not* handle gas phase reactions.


The stoichiometric matrix
-------------------------

To specify the problem, we define a set of :math:`n` chemical species and a set of :math:`r` reversible chemical reactions. Let :math:`S_j` be the symbol
for chemical species :math:`j`.  Then, chemical reaction :math:`i` is represented as

.. math::

  \sum_{j} \nu_{ij}\,S_j = 0,


where :math:`\nu_{ij}` is the stoichiometric coefficient of species :math:`j` in
reaction :math:`r`.  We can assemble the :math:`\nu_{ij}` into an :math:`r \times
n` **stoichiometric matrix** :math:`\mathsf{N}`.  Each row of :math:`\mathsf{N}` represents a single chemical reaction.

As an illustrative example, consider a chemical reaction system where species A, B, and C may bind each other to form dimers AB, BB, and BC, and A and C may be interconverted via the chemical reactions

.. math::
	&\mathrm{A} \rightleftharpoons \mathrm{C}\\
	&\mathrm{AB} \rightleftharpoons \mathrm{A} + \mathrm{B}\\
	&\mathrm{BB} \rightleftharpoons 2\mathrm{B}\\
	&\mathrm{BC} \rightleftharpoons \mathrm{B} + \mathrm{C}.

We can represent these reactions with the stoichiometric coefficients as defined above.

.. math::
	\begin{array}{rrrrrrcr}
	&-\mathrm{A} &  & + \mathrm{C} & & &  & = & 0 \\	
	&\phantom{-}\mathrm{A} & + \mathrm{B} &  & - \mathrm{AB} &  &  & = & 0 \\
	& & \phantom{+}2\mathrm{B} & & & - \mathrm{BB} &  & = & 0 \\
	& & \phantom{+}\mathrm{B} & + \mathrm{C} &  &  & - \mathrm{BC} & = &0.
	\end{array}


Then, the stoichiometric matrix is

.. math::

	\mathsf{N} =
	\begin{pmatrix}
	-1 & 0 & 1 & 0 & 0 & 0 \\
	1 & 1 & 0 & -1 & 0 & 0 \\
	0 & 2 & 0 & 0 & -1 & 0 \\
	0 & 1 & 1 & 0 & 0 & -1
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
  \prod_{j} c_j^{\nu_{ij}} = \frac{k_i^+}{k_i^-} \equiv K_i \;\forall i

:math:`K_i` is commonly called the **equilibrium constant**, though strictly
speaking may describe a steady state that is not necessarily at
equilibrium, as would be the case for a system that consumes or
produces energy through its reactions. 

From the above equation, it is clear that the equilibrium constant is in general not dimensionless, but has units of concentration raise to some power. If instead we use dimensionless concentrations, or mole fractions :math:`\mathbf{x}`, defined by

.. math::

	\mathbf{x} = \mathbf{c} / \rho_\mathrm{solv},

where :math:`\rho_\mathrm{solv}` is the **number density of the solvent**. For example, at atmospheric pressure and room temperature, :math:`\rho_\mathrm{H_2O} \approx 55` moles per liter. 

We can write the equilibrium expression in a more compact form.

.. math::
  \ln \mathbf{K} = \mathsf{N} \cdot \ln \mathbf{c}.

(It appears as though we are taking logarithms of dimensional quantities here, but the units do appropriately cancel upon rearrangement of the equation.)


Conservation laws
-----------------

If :math:`\mathsf{N}` is square (:math:`n = r`), then the equilibrium concentrations are immediately attained by solving the linear system

.. math::
  \ln \mathbf{K} = \mathsf{N} \cdot \ln \mathbf{c}.

This is almost never the case; in most applications there are more chemical species than there are reactions, and :math:`n > r`. The equilibrium expression is then underdetermined, and we need :math:`n - r` additional equations to solve for the concentrations.

Let us assume that we initially have concentrations :math:`\mathbf{c}^0` of chemical species in our dilute solution. We refer to the :math:`n`-vector :math:`\mathbf{c}^0` as the **initial concentrations**. There exists a **conservation matrix** :math:`\mathsf{A}` such that

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

The set equilibrium concentrations satisfying the above system of equations is unique (proven in the paper accompanying this software).


Problem specification
---------------------

The necessary ingredients to fully specify an equilibrium calculation are now clear.

- The :math:`r \times n` stoichiometric matrix, :math:`\mathsf{N}` (full row rank).
- The :math:`r` equilibrium constants, :math:`\mathbf{K}` (positive and finite).
- The :math:`n` initial concentrations, :math:`\mathbf{c}^0` (nonnegative and finite).

It is not necessary to specify the conservation matrix :math:`\mathsf{A}`, as it can be calculated from the null space of the stoichiometric matric :math:`\mathsf{N}`.


Elemental conservation matrices
-------------------------------

Keeping in mind our example, 

.. math::
  \mathsf{N} =
  \begin{pmatrix}
    \mathrm{A} & \mathrm{B} & \mathrm{C} & \mathrm{AB} & \mathrm{BB} & \mathrm{BC} \\ \hline
    -1 & 0 & 1 & 0 & 0 & 0 \\
    1 & 1 & 0 & -1 & 0 & 0 \\
    0 & 2 & 0 & 0 & -1 & 0 \\
    0 & 1 & 1 & 0 & 0 & -1
  \end{pmatrix},


where we have annotated the columns of :math:`\mathsf{N}` to indicate the
respective chemical species.  We can compute the null space of
:math:`\mathsf{N}` to be

.. math::
  \mathsf{A} =
  \begin{pmatrix}
    \mathrm{A} & \mathrm{B} & \mathrm{C} & \mathrm{AB} & \mathrm{BB} & \mathrm{BC} \\ \hline
    1 & 0 & 1 & 1 & 0 & 1 \\
    0 & 1 & 0 & 1 & 2 & 1
  \end{pmatrix}.

The conservation law :math:`\mathsf{A} \cdot \mathbf{c} = \mathsf{A}
\cdot \mathbf{c}^0` is interpreted as a statement of conservation of mass for
irreducible species of type A and B.

.. math::
  &c_A + c_C + 2c_{AB} + c_{BC} = c_A^0 + c_C^0 + 2c_{AB}^0 + c_{BC}^0, \\
  &c_B + 2c_{BB} + c_{BC} = c_B^0 + 2c_{BB}^0 + c_{BC}^0.

We will use the term **element** to define an irreducible chemical species (not necessarily the elements that appear in the periodic table; just any chemical species that cannot be broken down). In our example system, A and B are elements, while AB, BB, and BC are not. We do not consider C to be an element, because it is a transformation of another element, A.

In the case above, the conservation matrix :math:`A` is an **elemental conservation matrix**. The entry in
column :math:`j` of row :math:`i` of an **elemental matrix** is the number of
elements of type :math:`i` that are in compound :math:`j`. In other words, each column represents the elemental composition of a compound. Having balanced chemical reactions (in which the number elements of every given type has equal representation on each side of the chemical reactions) is a prerequisite for a conservation matrix :math:`\mathsf{A}` being an
elemental matrix, but is not sufficient, as shown in the next example.
Note also that an elemental matrix need not have linearly independent
rows in general, also shown in the next example, :math:`\mathsf{A}` must.

Note that the elemental conservation matrix is one choice among many conservation matrices. The matrix

.. math::

  \begin{pmatrix}
    1 & -3.5 & 1 & -2.5 & -7 & -2.5 \\
    -2 & 1 & -2 & -1 & 2 & -1
  \end{pmatrix}

is also a conservation matrix, but it not elemental.


Non-elemental conservation matrices
-----------------------------------

For a given set of chemical reactions, conservation matrices need not be elemental. This happens when, unlike in our previous example, there is no reaction to break compounds down into their elements. As an illustrative example, consider the chemical reaction

.. math::
  \mathrm{AB} + \mathrm{CD} \rightleftharpoons \mathrm{AC} + \mathrm{BD}.

Here,

.. math::
  \mathsf{N} =   \begin{pmatrix}
    \mathrm{AB} & \mathrm{CD} & \mathrm{AC} & \mathrm{BD} \\ \hline
    -1 & -1 & 1 & 1
    \end{pmatrix}

A conservation matrix whose rows span the null space is

.. math::
  \mathsf{A} =   \begin{pmatrix}
    \mathrm{AB} & \mathrm{CD} & \mathrm{AC} & \mathrm{BD} \\ \hline
    1 & 0 & 1 & 0 \\
    1 & 0 & 0 & 1 \\
    0 & 1 & 1 & 0 \\
    \end{pmatrix}.

This matrix is not elemental because the second and fourth columns do not represent the elemental composition of a compound. The first row of the matrix represents conservation of particles of type A, the second of type B, and the third of type C. If we were to have a conservation law for particles of type D, that row would be :math:`(0, 1, 0, 1)`, and we would have an elemental matrix. But this row is a linear combination of the other three rows, namely row 2 minus row 1 plus row 3. Thus, the elemental matrix for this example system is comprised of linear combinations of the null space, but is
redundant with respect to conservation laws.


The free energies of chemical species
-------------------------------------

We can define an :math:`n`-vector :math:`\mathbf{G}` such that

.. math::

  K_i = \exp\left\{ -\sum_{j} \nu_{ij}\,G_j\right\} \;\forall i,

or equivalently

.. math::

  -\ln \mathbf{K} = \mathsf{N} \cdot \mathbf{G}.

In the case of a system at equilibrium, :math:`\mathbf{G}` has the meaning
of the set of **free energies** (in units of the thermal energy :math:`kT`) associated with each chemical species, as given by `detailed balance`_.  Since
by construction, we almost always have :math:`n > r`, :math:`\mathbf{G}` is underdetermined. To determine :math:`\mathbf{G}`, we must set a reference free energy.  To do so, we augment :math:`\mathsf{N}` with :math:`\mathsf{A}` to create an
:math:`n \times r` matrix :math:`\mathsf{N}'`.

.. math::
  \mathsf{N}' = \begin{pmatrix} \mathsf{A} \\ \mathsf{N}
  \end{pmatrix}.

We similarly define an :math:`n`-vector :math:`\mathbf{b}`,

.. math::
  \mathbf{b} = \begin{pmatrix}
    0 \\
    -\log \mathbf{K}
    \end{pmatrix}.

Thus, the free energies of all species may be obtained by solving

.. math::
  \mathsf{N}' \cdot \mathbf{G} = \mathbf{b}.

(This equation is solvable because :math:`\mathsf{N}` has full row rank and
the augmented rows comprise its null space, being orthogonal to all
rows in the rest of :math:`\mathsf{N}'`.)  Setting the first
:math:`n-r` entries of :math:`\mathbf{b}` to zero simply sets the
reference free energy.


Specification in terms of conservation matrices and free energies
-----------------------------------------------------------------

We have demonstrated how a conservation matrix and set of free energies may be computed from a stoichiometric matrix and a set of equilibrium constants. We may also go the other way; given :math:`\mathsf{A}` and :math:`\mathbf{G}`, we can compute :math:`\mathsf{N}` and :math:`\mathbf{K}`. We first compute :math:`\mathsf{N}` from the null space of :math:`\mathsf{A}`, and then compute the equilibrium constants using

.. math::

  K_i = \exp\left\{ -\sum_{j} \nu_{ij}\,G_j\right\} \;\forall i,

where :math:`\nu_{ij}` is entry :math:`i, j` in :math:`\mathsf{N}`. So, we may alternatively specify the equilbrium problem giving:

- The :math:`(n-r) \times n` conservation matrix, :math:`\mathsf{A}` (nonnegative and full row rank).
- The :math:`n` free energies, :math:`\mathbf{G}` (finite).
- The :math:`n` initial concentrations, :math:`\mathbf{c}^0` (nonnegative and finite).

We have stipulated that the constraint matrix is nonnegative. While not strictly a requirement to formulate the problem, the nonnegativity of user-supplied :math:`\mathsf{A}` is necessary to allow treatment of cases where some of the initial concentrations are zero; :math:`c_j^0 = 0`. (For details on this requirement, see algorithmic details.) In practice, users will almost always supply elemental conservation matrices.


.. _law of mass action: http://en.wikipedia.org/wiki/Law_of_mass_action
.. _Palsson's systems biology book: https://doi.org/10.1017/CBO9781139854610.012
.. _detailed balance: https://en.wikipedia.org/wiki/Detailed_balance

