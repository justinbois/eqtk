.. _eqtk_checks:

Check satisfaction of equilibrium conditions
============================================

If EQTK's solve algorithm converges, equilibrium and conservation laws are guaranteed to be satisfied (see :ref:`algorithmic_details`). Nonetheless, it is sometimes useful to check if equilibrium conditions are met. To understand the discussion below, it may help to refresh yourself of the :ref:`core concepts <core_concepts>` behind the couple equilibrium problem.

As a simple example, consider the chemical reactions

AB ⇌ A + B ; *K* = *K* ₁

AC ⇌ A + C ; *K* = *K* ₂

The equilibrium conditions are

.. math::

	&K_1 = \frac{c_\mathrm{A}\,c_\mathrm{B}}{c_\mathrm{AB}},\\
	&K_2 = \frac{c_\mathrm{A}\,c_\mathrm{C}}{c_\mathrm{AC}}.

Additionally, three conservation laws, corresponding to conservation of A, B, and C, must be satisfied.

.. math::

	&c_\mathrm{A}^0 = c_\mathrm{A} + c_\mathrm{AB} + c_\mathrm{AC},\\
	&c_\mathrm{B}^0 = c_\mathrm{B} + c_\mathrm{AB},\\
	&c_\mathrm{C}^0 = c_\mathrm{C} + c_\mathrm{AC}.

Together, these five equations define the equilibrium. We can write these conditions in terms of the stoichiometric matrix :math:`\mathsf{N}` and conservation matrix :math:`\mathsf{A}`. The stoichiometric matrix corresponding to the chemical reactions is

.. math::

	\mathsf{N} = \begin{pmatrix}
	1 & 1 & 0 & -1 & 0 \\	
	1 & 0 & 1 & 0 & -1 
	\end{pmatrix}.

The conservation matrix :math:`\mathsf{A}` has rows that span the null space of :math:`\mathsf{N}`.

.. math::

	\mathsf{A} = \begin{pmatrix}
	1 & 0 & 0 & 1 & 0 \\	
	0 & 1 & 0 & 1 & 1 \\
	0 & 0 & 1 & 0 & 1
	\end{pmatrix}.

If we define 

.. math::

	&\mathbf{c} = (c_\mathrm{A}, c_\mathrm{B}, c_\mathrm{C}, c_\mathrm{AB}, c_\mathrm{AC})^\mathsf{T},\\
	&\mathbf{c}^0 = (c_\mathrm{A}^0, c_\mathrm{B}^0, c_\mathrm{C}^0, c_\mathrm{AB}^0, c_\mathrm{AC}^0)^\mathsf{T},

the equilibrium conditions are

.. math::

	&K_1 = \prod_j c_j^{N_{1j}},\\
	&K_2 = \prod_j c_j^{N_{2j}},

or more generally

.. math::

	K_i = \prod_j c_j^{N_{ij}} \;\forall i.

The condition for conservation is

.. math::

	\mathsf{A}\cdot\mathbf{c} = \mathsf{A} \cdot \mathbf{c}^0.


Check of satisfaction of equilibrium and conservation conditions
----------------------------------------------------------------

The ``eqtk.eqcheck()`` function conveniently checks to make sure equilibrium and conservation conditions are met. To check equilibrium conditions, it verifies that

.. math::

	\frac{\prod_j c_j^{N_{ij}}}{K_i} \approx 1 \;\forall i,

where equality is checked to some tolerance.

Similarly to check conservation conditions, it verifies that

.. math::

	\frac{\mathsf{A}\cdot\mathbf{c}}{\mathsf{A} \cdot \mathbf{c}^0} \approx \mathbf{1}.

This calculation has a wrinkle: if :math:`\mathsf{A} \cdot \mathbf{c}^0 \approx 0`, the above ratio is not properly defined. In that case, it checks that :math:`\mathsf{A}\cdot\mathbf{c} \approx 0` to within some tolerance.

Let's use this function to verify that an equilibrium calculation was successful. To start with, we will use Numpy arrays as inputs. (In what follows, we assume EQTK, Numpy, and Pandas are all imported.)

.. code-block:: python

	# Set up problem
    c0 = [1, 0.5, 0.25, 0, 0]

    N = [[1,  1,  0, -1,  0],
         [1,  0,  1,  0, -1]]

    K = [0.015, 0.003]

    # Solve
    c = eqtk.solve(c0=c0, N=N, K=K, units="mM")

    # Verify calculation converged
    eqtk.eqcheck(c, c0=c0, N=N, K=K, units="mM")

The function return ``True``.

If we instead ``N`` stored as a data frame and ``c0`` as a series or data frame, it is not necessary to supply ``K``, as it is already in the ``N`` data frame, nor is it necessary to supple ``c0`` or ``units``, as they can be inferred from ``c``.

.. code-block:: python

	# Set up problem
    names = ["A", "B", "C", "AB", "AC"]
    c0 = pd.Series(data=[1, 0.5, 0.25, 0, 0], index=names)

    N = pd.DataFrame(data=[[1,  1,  0, -1,  0],
				           [1,  0,  1,  0, -1]],
				     columns=names)
    N['equilibrium_constant'] = [0.015, 0.003]

    # Solve
    c = eqtk.solve(c0=c0, N=N, units="mM")

    # Verify calculation converged
    eqtk.eqcheck(c, N=N)

This calculation again return ``True``.


Quantitative check in error in equilibrium and conservation conditions
----------------------------------------------------------------------

To get more detailed information, specifically the value of the ratio

.. math::

	\frac{\prod_j c_j^{N_{ij}}}{K_i}

and the ratio

.. math::

	\frac{\mathsf{A}\cdot\mathbf{c}}{\mathsf{A} \cdot \mathbf{c}^0},

or, if :math:`\mathsf{A} \cdot \mathbf{c}^0 \approx 0`, the value of

.. math::

	\mathsf{A} \cdot \mathbf{c},

you can use ``eqtk.eqcheck_quant()``. In addition to returning arrays with these quantities, it returns a third array indicating whether :math:`\mathsf{A} \cdot \mathbf{c}^0 \approx 0` for each conservation law.

Running

.. code-block:: python

	eqtk.eqcheck_quant(c, N=N)

returns ::

	(array([1., 1.]), array([1., 1., 1.]), array([False, False, False]))

