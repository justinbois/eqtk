"""
Readout functions for EQTK.

The functions all have the same functional form.  They look like this:

readout_fun(c, fixed_params, float_params)
    Parameters
    ----------
    c : array_like, shape(n_compounds,) or shape(n_titration_pts, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in fixed_params[1].  If
        c has shape (n_compounds,), then c[i] is the equilibrium concentration
        of compound j for the single set of concentrations considered.
    fixed_params : tuple
        Each entry in the tuple is a parameter that is necessary to compute
        the readout of the concentrations, but may not be manipulated by
        curve fitting algorithms.  For example, the specification of
        fixed_params for pH is
        fixed_params[0] : int
            The index in c corresponding to hydrogen/hydronium ions.
        fixed_params[1] : string
            The units of the entries in c.  Allowable values are 'molar', 'M',
            'millimolar', 'mM', 'micromolar', 'uM', 'nanomolar', 'nM',
            'picomolar', 'pM', None.  If None, concentrations are given as
            mole fractions.
        fixed_params[2] : float
            The temperature, in deg. C, of the solution.  Ignored if
            fixed_params[1] is not None.
    float_params : float or array_like
        Either a float or a NumPy ndarray of parameters that may
        be manipulated by curve fitting algorithms.  For example, a
        specification of p for absorbance under the Beer-Lambert Law is:
        float_params : array_like, shape (n_compounds,)
            float_params[i] is the absorption coeffient for compound i in units
            commensurate with those of c.  The total absorbance is
            A = np.dot(c, float_params)

    Returns
    -------
    ret_val : float or array_like, shape(n_titration_pts,)
        If c has shape (n_compounds,), returns a float.  If c has shape
        (n_compounds, n_titration_pts), returns an array of shape
        (n_titration_pts,).

    Raises
    ------
    ValueError
        If input is in any way invalid
"""

import numpy as np

from .eqtk import _water_density

# ##############################################################################
def pH(c, fixed_params, float_params):
    """
    pH of a solution.

    Parameters
    ----------
    c : array_like, shape(n_compounds,) or shape(n_titration_pts, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in fixed_params[1].  If
        c has shape (n_compounds,), then c[i] is the equilibrium concentration
        of compound j for the single set of concentrations considered.
    fixed_params : tuple, length of 2 or 3
        fixed_params[0] : int
            The index in c corresponding to hydrogen/hydronium ions.
        fixed_params[1] : string
            The units of the entries in c.  Allowable values are 'molar', 'M',
            'millimolar', 'mM', 'micromolar', 'uM', 'nanomolar', 'nM',
            'picomolar', 'pM', None.  If None, concentrations are given as
            mole fractions.
        fixed_params[2] : float
            The temperature, in deg. C, of the solution.  Ignored if
            fixed_params[1] is not None.
    float_params : None
        Ignored.

    Returns
    -------
    pH : float or array_like, shape(n_titration_pts,)
        The pH of the solution.  If c has shape (n_compounds,), returns a float.
        If c has shape (n_compounds, n_titration_pts), returns an array of
        shape (n_titration_pts,).

    Raises
    ------
    ValueError
        If input is in any way invalid

    Examples
    --------
    1) Compute the pH of a 1 M solution of weak acid HA with acid dissociation
       constant 1e-5 M.

    >>> import readout as ro
    >>> import eqtk as eq
    >>> N = np.array([[1,  1,  0,  0],
                      [1,  0, -1,  1]])
    >>> K = np.array([1.0e-14, 1.0e-5])
    >>> c_0 = np.array([1.0e-7, 1.0e-7, 1.0, 0.0])
    >>> c = eq.eqtk(N, K, c_0, units='M')
    >>> ro.pH(c, (0, 'M', 25.0), None)
        2.5006866803595336

    2) Compute the pH titration curve of a 1 M solution of weak acid HA
       with acid dissociation constant 1e-5 M, titrating in 1.0 M NaOH.

    >>> import readout as ro
    >>> import eqtk as eq
    >>> N = np.array([[1,  1,  0,  0],
                      [1,  0, -1,  1]])
    >>> K = np.array([1.0e-14, 1.0e-5])
    >>> c_0 = np.array([1.0e-7, 1.0e-7, 1.0, 0.0])
    >>> c_0_titrant = np.array([0.0, 1.0, 0.0, 0.0])
    >>> initial_volume = 0.1
    >>> vol_titrated = np.array([0.0, 0.1, 0.2]) # Only a few for display
    >>> c = eq.volumetric_titration(N, K, c_0, initial_volume, c_0_titrant,
                                    vol_titrated, units='M')
    >>> ro.pH(c, (0, 'M', 25.0), None)
        array([  2.50068668,   9.34947963,  13.52287875])
    """

    # Make sure we have the required input
    if len(fixed_params) < 2:
        raise ValueError("Invalid fixed_params.")
    if type(fixed_params[0]) != int:
        raise ValueError("Invalid fixed_params.")
    if type(fixed_params[1]) != str:
        raise ValueError("Invalid fixed_params.")

    # Convert to molar
    if fixed_params[1] is None:
        if len(fixed_params) == 2:
            raise ValueError("Must specify fixed_params[2] as temperature.")
        c *= _water_density(fixed_params[2], "M")
    elif fixed_params[1] == "millimolar" or fixed_params[1] == "mM":
        c *= 1000.0
    elif fixed_params[1] == "micromolar" or fixed_params[1] == "uM":
        c *= 1.0e6
    elif fixed_params[1] == "nanomolar" or fixed_params[1] == "nM":
        c *= 1.0e9
    elif fixed_params[1] == "picomolar" or fixed_params[1] == "pM":
        c *= 1.0e12
    elif fixed_params[1] == "molar" or fixed_params[1] == "M":
        pass
    else:
        raise ValueError("Invalid fixed_params[1] specification.")

    # 1D array; no titration, return scalar
    if len(c.shape) == 1:
        return -np.log10(c[fixed_params[0]])
    else:  # 2D array, return array
        return -np.log10(c[:, fixed_params[0]])


# ##############################################################################

# ##############################################################################
def absorbance_beer_lambert(c, fixed_params, float_params):
    """
    Absorbance of a solution as calculated by the Beer-Lambert law.

    Parameters
    ----------
    c : array_like, shape(n_compounds,) or shape(n_titration_pts, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in fixed_params[1].  If
        c has shape (n_compounds,), then c[i] is the equilibrium concentration
        of compound j for the single set of concentrations considered.
    fixed_params : empty tuple
        There are no fixed parameters for this readout function.
    float_params : array_like, shape (n_compounds,)
        float_params[i] is the absorption coeffient for compound i in units
        commensurate with those of c.  The total absorbance is
        A = np.dot(c, float_params).

    Returns
    -------
    A : float or array_like, shape(n_titration_pts,)
        The absorbance of the solution.  If c has shape (n_compounds,),
        returns a float. If c has shape (n_compounds, n_titration_pts),
        returns an array of shape (n_titration_pts,).

    Raises
    ------
    ValueError
        If input is in any way invalid
    """

    # Make sure we have the required input
    if len(c.shape) == 1:
        if len(float_params) != len(c):
            raise ValueError("float_params must have n_compounds entries.")
    elif len(float_params) != c.shape[1]:
        raise ValueError("float_params must have n_compounds entries.")

    return np.dot(c, float_params)


# ##############################################################################

# ##############################################################################
def concentration(c, fixed_params, float_params):
    """
    Returns a linear combination of the concentrations.

    Parameters
    ----------
    c : array_like, shape(n_compounds,) or shape(n_titration_pts, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in fixed_params[1].  If
        c has shape (n_compounds,), then c[i] is the equilibrium concentration
        of compound j for the single set of concentrations considered.
    fixed_params : empty tuple
        There are no fixed parameters for this readout function.
    float_params : array_like, shape (n_compounds,)
        float_params[i] is the weight for compound i in the linear combination
        of concentrations.  The final output is C = np.dot(c, float_params).

    Returns
    -------
    C : float or array_like, shape(n_titration_pts,)
        A linear combintation of concentrations.  If c has shape
        (n_compounds,), returns a float. If c has shape
        (n_compounds, n_titration_pts), returns an array of shape
        (n_titration_pts,).

    Raises
    ------
    ValueError
        If input is in any way invalid

    Notes
    -----
    .. This is mathematically identical to absorbance_beer_lambert.
    """

    # Make sure we have the required input
    if len(c.shape) == 1:
        if len(float_params) != len(c):
            raise ValueError("float_params must have n_compounds entries.")
    elif len(float_params) != c.shape[1]:
        raise ValueError("float_params must have n_compounds entries.")

    return np.dot(c, float_params)


# ##############################################################################


# ##############################################################################
def anisotropy(c, fixed_params, float_params):
    """
    Fluorescence anisotropy

    Parameters
    ----------
    c : array_like, shape(n_compounds,) or shape(n_titration_pts, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in fixed_params[1].  If
        c has shape (n_compounds,), then c[i] is the equilibrium concentration
        of compound j for the single set of concentrations considered.
    fixed_params : tuple, length 2
        fixed_params[0] : int
            Index in c corresponding to fluorescent species.
        fixed_params[1] : array_like, shape(n_compounds,)
            fixed_params[1][i] is the number of particles of the fluorescent
            species in compound i.  If A is the elemental matrix,
            fixed_params[1] is the row of A corresponding to the fluorescent
            species.
    float_params : array_like, shape (n_compounds,)
        float_params[i] is the anisotropy of compound i on a per-fluorescent
        particle basis.  The total (measured) anisotropy for titration point
        i is r[i] = np.dot(float_params,
                   fixed_params[1] * c[i,:] / np.dot(c[i,:], fixed_params[1])).

    Returns
    -------
    r : float or array_like, shape(n_titration_pts,)
        The anisotropy of the solution.  If c has shape (n_compounds,),
        returns a float. If c has shape (n_compounds, n_titration_pts),
        returns an array of shape (n_titration_pts,).

    Raises
    ------
    ValueError
        If input is in any way invalid
    """

    # Make sure we have the required input
    if len(fixed_params) != 2:
        raise ValueError("fixed_params must be tuple of length 2.")
    if len(c.shape) == 1:
        if len(float_params) != len(c):
            raise ValueError("float_params must have n_compounds entries.")
        if len(fixed_params[1]) != len(c):
            raise ValueError("fixed_params[1] must have n_compounds entries.")
    elif len(float_params) != c.shape[1]:
        raise ValueError("float_params must have n_compounds entries.")
    elif len(fixed_params[1]) != c.shape[1]:
        raise ValueError("fixed_params[1] must have n_compounds entries.")

    if len(c.shape) == 1:
        r = np.dot(float_params, fixed_params[1] * c / np.dot(c, fixed_params[1]))
    else:
        r = np.empty(c.shape[0])
        for i in xrange(c.shape[0]):
            r[i] = np.dot(
                float_params,
                fixed_params[1] * c[i, :] / np.dot(c[i, :], fixed_params[1]),
            )
    return r


# ##############################################################################
