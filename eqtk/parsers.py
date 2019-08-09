import collections
import warnings

import numpy as np
import pandas as pd

from . import constants


def parse_input(c0, N, K, A, G, names, units, solvent_density, T, G_units):
    """
    """
    _check_NK_AG(N, K, A, G)
    _check_units(units)
    _check_G_units(G_units, T)
    _check_T(T)
    _check_solvent_density(solvent_density, units)
    _check_names_type(names)

    c0, n_compounds, names, c0_from_df = _parse_c0(c0, names)
    _check_c0(c0)

    N, K = _parse_N_input(N, K, names, c0_from_df)
    _check_N(N, n_compounds)

    K = _parse_K(K)
    _check_K(N, K)

    A = _parse_A(A, names, c0_from_df)
    _check_A(A, n_compounds)

    G = _parse_G(G, names, c0_from_df)
    _check_G(A, G)

    if G is None:
        x0, K, solvent_density = _nondimensionalize_NK(
            c0, N, K, T, solvent_density, units
        )
    else:
        x0, G, solvent_density = _nondimensionalize_AG(
            c0, G, T, solvent_density, units, G_units
        )

    return x0, N, K, A, G, names, solvent_density


def _parse_output(x, c0, names, solvent_density, single_point):
    """
    """
    if solvent_density is None:
        c = x
    else:
        c = x * solvent_density

    if single_point:
        c = c.flatten()

    if names is None:
        return c

    # Names of columns for outputted data frames
    cols = [name + "__0" for name in names] + names

    if single_point:
        return pd.Series(data=np.concatenate((c0, c)), index=cols)
    else:
        return pd.DataFrame(data=np.concatenate((c0, c), axis=1), columns=cols)


def parse_rxns(rxns):
    rxn_list = rxns.splitlines()
    N_dict_list = []
    K_list = []
    for rxn in rxn_list:
        if rxn.strip() != "":
            N_dict, K = _parse_rxn(rxn)
            N_dict_list.append(N_dict)
            K_list.append(K)

    # Make sure K's are specified for all or none
    if not (all([K is None for K in K_list]) or all([K is not None for K in K_list])):
        raise ValueError(
            "Either all or none of the equilibrium constants must be specified."
        )

    if K_list[0] is None:
        K = None
    else:
        K = np.array(K_list, dtype=float)

    # Unique chemical species
    species = []
    for N_dict in N_dict_list:
        for compound in N_dict:
            if compound not in species:
                species.append(compound)

    # Build stoichiometric matrix
    N = np.zeros((len(N_dict_list), len(species)), dtype=float)
    for r, N_dict in enumerate(N_dict_list):
        for compound, coeff in N_dict.items():
            N[r, species.index(compound)] = coeff

    return N, K, species


def _nondimensionalize_NK(c0, N, K, T, solvent_density, units):
    # Compute solvent density in appropriate units
    solvent_density = _parse_solvent_density(solvent_density, T, units)

    # Convert K's and c0 to dimensionless
    K_nondim = K / solvent_density ** N.sum(axis=1)
    c0_nondim = c0 / solvent_density

    return c0_nondim, K_nondim, solvent_density


def _nondimensionalize_AG(c0, G, T, solvent_density, units, G_units):
    # Compute solvent density in appropriate units
    solvent_density = _parse_solvent_density(solvent_density, T, units)

    # Convert G's and c0 to dimensionless
    G_nondim = _dimensionless_free_energy(G, G_units, T)
    c0_nondim = c0 / solvent_density

    return c0_nondim, G_nondim, solvent_density


def _parse_solvent_density(solvent_density, T, units):
    if solvent_density is None:
        return _water_density(T, units)
    elif (units is None or units == "") and solvent_density != 1.0:
        raise ValueError(
            "If `solvent_density` is specified, `units` must also be specified."
        )

    return solvent_density


def _water_density(T, units):
    """
    Calculate the number density of water in specified units.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    units : string, default = 'M'
        The units in which the density is to be calculated.
        Valid values are: 'M', 'mM', 'uM', 'µM', 'nM', 'pM'.

    Returns
    -------
    water_density : float
        Particle density of water in `units`.

    References
    ----------
    Tanaka M., Girard, G., Davis, R., Peuto A.,
    Bignell, N.   Recommended table for the denisty
    of water..., Metrologia, 2001, 38, 301-309
    """
    # If dimensionless, take solvent density to be unity
    if units is None or units == "":
        return 1.0

    a1 = -3.983035
    a2 = 301.797
    a3 = 522528.9
    a4 = 69.34881
    a5 = 999.974950

    # Convert temperature to celsius
    T_C = T - constants.absolute_zero

    # Compute water density in units of molar
    dens = a5 * (1 - (T_C + a1) * (T_C + a1) * (T_C + a2) / a3 / (T_C + a4)) / 18.0152

    # Valid units
    allowed_units = (
        None,
        "M",
        "molar",
        "mM",
        "millimolar",
        "uM",
        "µM",
        "micromolar",
        "nM",
        "nanomolar",
        "pM",
        "picomolar",
    )

    # Convert to specified units
    if units in ["millimolar", "mM"]:
        dens *= 1000.0
    elif units in ["micromolar", "uM", "µM"]:
        dens *= 1000000.0
    elif units in ["nanomolar", "nM"]:
        dens *= 1000000000.0
    elif units in ["picomolar", "pM"]:
        dens *= 1000000000000.0
    elif units not in ["molar", "M"]:
        raise ValueError(
            f"Specified concentration units of {units} not in {allowed_units}."
        )

    return dens


def _dimensionless_free_energy(G, units, T=293.15):
    """
    Convert free energy to dimensionless units, where G is in given units.
    """
    if units is None or units == "kT":
        return G
    elif T is None:
        raise ValueError("If G is specified with units, must also supply T.")

    kT = _thermal_energy(T, units)
    return G / kT


def _thermal_energy(T, units):
    """
    Return value of thermal energy kT in specified units. T is
    assumed to be in Kelvin.
    """
    if T < 100.0:
        warnings.warn("WARNING: T may be in wrong units, must be in K.")

    allowed_units = ["kcal/mol", "J", "J/mol", "kJ/mol", "pN-nm"]

    if units == "kcal/mol":
        return constants.kB_kcal_per_mol * T
    elif units == "J":
        return constants.kB_J * T
    elif units == "J/mol":
        return constants.kB_J_per_mol * T
    elif units == "kJ/mol":
        return constants.kB_kJ_per_mol * T
    elif units == "pN-nm":
        return constants.kB_pN_nm * T
    else:
        raise ValueError(
            f"Specified thermal energy units of {units} not in {allowed_units}."
        )


def _check_NK_AG(N, K, A, G):
    if N is None:
        if A is None or G is None:
            raise ValueError("`A` and `G` must both be specified if `N` is None.")

        if type(A) == pd.core.frame.DataFrame:
            if type(G) not in [pd.core.frame.DataFrame, pd.core.series.Series]:
                raise ValueError(
                    "If `A` is inputted as a DataFrame, `G` must be inputted as a DataFrame or Series."
                )
        elif type(G) in [pd.core.frame.DataFrame, pd.core.series.Series]:
            raise ValueError(
                "If `G` is inputted as a DataFrame or Series, `A` must be inputted as a DataFrame."
            )

    elif type(N) == pd.core.frame.DataFrame:
        if "equilibrium constant" not in N and "equilibrium_constant" not in N:
            raise ValueError(
                "If `N` is inputted as a DataFrame, it must have an `'equilibrium constant'` column."
            )
        if K is not None:
            raise ValueError(
                "If `N` is inputted as a DataFrame, `K` must be `None`. The equilibrium constants are included as the `'equilibrium constant'` column of the inputted DataFrame."
            )
    elif K is None:
        raise ValueError("If `N` is not inputted as a DataFrame, `K` must be provided.")


def _check_units(units):
    allowed_units = (
        None,
        "M",
        "molar",
        "mM",
        "millimolar",
        "uM",
        "µM",
        "micromolar",
        "nM",
        "nanomolar",
        "pM",
        "picomolar",
    )
    if units not in allowed_units:
        raise ValueError(
            f"Specified concentration units of {units} not in {allowed_units}."
        )


def _check_G_units(G_units, T):
    allowed_units = (None, "kT", "kcal/mol", "J", "J/mol", "kJ/mol", "pN-nm")
    if G_units not in allowed_units:
        raise ValueError(
            f"Specified free energy units of {G_units} not in {allowed_units}."
        )

    if G_units not in (None, "kT") and T is None:
        raise ValueError("If G is specified with units, must also supply T.")


def _check_T(T):
    if T is not None:
        if T < 0:
            raise ValueError("`T` must be positive, in units of Kelvin.")
        if T < 100.0:
            warnings.warn("WARNING: T may be in wrong units, must be in Kelvin.")


def _check_solvent_density(solvent_density, units):
    if solvent_density is not None and units is None and solvent_density != 1.0:
        raise ValueError(
            "If `solvent_density` is specified, `units` must also be specified."
        )


def _check_names_type(names):
    if names is not None:
        if type(names) not in [list, tuple, np.ndarray]:
            raise ValueError("`names` must be a list, tuple, or Numpy array.")
        for name in names:
            if not isinstance(name, collections.Hashable):
                raise ValueError(
                    f"{name} is an invalid name because it is not hashable."
                )


def _parse_c0(c0, names):
    if type(c0) == list or type(c0) == tuple:
        c0 = np.array(c0, order="C", dtype=float)

    if type(c0) == np.ndarray:
        if len(c0.shape) == 1:
            n_compounds = c0.shape[0]
            c0 = np.expand_dims(c0, axis=0)
        elif len(np.shape(c0)) == 2:
            n_compounds = c0.shape[1]
        else:
            raise ValueError("c0 is the wrong shape.")
        c0_from_df = False

        if names is not None:
            if len(names) != n_compounds:
                raise ValueError(
                    "`len(names)` must equal the number of columns of `c0`."
                )
    else:
        if type(c0) == pd.core.frame.DataFrame:
            names = _check_names_df(names, list(c0.columns))
        elif type(c0) == pd.core.series.Series:
            names = _check_names_df(names, list(c0.index))
        else:
            raise ValueError(
                "`c0` must be a Pandas series or data frame or Numpy array."
            )
        c0 = c0[names].to_numpy(dtype=float, copy=True)
        n_compounds = len(names)
        c0_from_df = True
        if len(c0.shape) == 1:
            c0 = np.expand_dims(c0, axis=0)

    return np.ascontiguousarray(c0), n_compounds, names, c0_from_df


def _check_c0(c0):
    if np.isnan(c0).any():
        raise ValueError("No NaN values are allowed for c0.")
    if (c0 < 0.0).any():
        raise ValueError("All c0's must be nonnegative.")
    if np.isinf(c0).any():
        raise ValueError("All c0's must be finite!")


def _parse_N_input(N, K, names, c0_from_df):
    if N is not None:
        if type(N) == list or type(N) == tuple:
            N = np.array(N)

        if type(N) == np.ndarray:
            N = _N_from_array(N)
        elif type(N) == pd.core.frame.DataFrame:
            if not c0_from_df:
                raise ValueError(
                    "If `N` is specified as a DataFrame, so too must `c0`."
                )
            names_from_df = list(
                N.columns[
                    ~(
                        N.columns.str.contains("equilibrium constant")
                        | N.columns.str.contains("equilibrium_constant")
                    )
                ]
            )
            names = _check_names_df(names, names_from_df)
            N, K = _NK_from_df(N, names, c0_from_df)
            _check_name_close_to_eqconst()
        else:
            raise ValueError(
                "Invalid type for `N`; must be array_like or a Pandas DataFrame."
            )

    return N, K


def _check_N(N, n_compounds):
    if N is not None:
        if N.shape[1] != n_compounds:
            raise ValueError(
                "Dimension mismatch between `c0` and inputted chemical species via `N`."
            )
        if len(N) > 0 and np.linalg.matrix_rank(N) != N.shape[0]:
            raise ValueError(
                "Innputed `N` results in rank deficient stoichiometic matrix."
            )
        if np.isinf(N).any():
            raise ValueError("All entries in the stoichiometic matrix must be finite.")
        if np.isnan(N).any():
            raise ValueError("No NaN values are allowed in the stoichiometric matrix.")


def _parse_K(K):
    if K is not None:
        if type(K) == list or type(K) == tuple:
            K = np.array(K, order="C", dtype=float)

        if len(np.shape(K)) == 2 and (np.shape(K)[1] == 1 or np.shape(K)[0] == 1):
            K = np.asconitugousarray(K.flatten())
        elif len(np.shape(K)) != 1:
            raise ValueError("`K` is the wrong shape.")

    return K


def _check_K(N, K):
    if K is not None:
        if len(K) != N.shape[0]:
            raise ValueError("`K` must have `N.shape[0]` entries")
        if not (K > 0.0).all():
            raise ValueError("All `K`'s must be positive.")
        if np.isinf(K).any():
            raise ValueError("All `K`'s must be finite.")
        if np.isnan(K).any():
            raise ValueError("No NaN values are allowed for `K`.")


def _parse_A(A, names, c0_from_df):
    if A is not None:
        if type(A) == list or type(A) == tuple:
            A = np.array(A, order="C").astype(float)

        if type(A) == np.ndarray:
            if len(A.shape) == 1:
                A = A.reshape((1, -1), order="C").astype(float)
        elif type(A) == pd.core.frame.DataFrame:
            names = _check_names_df(names, list(A.columns))
            A = _A_from_df(A, names, c0_from_df)
        else:
            raise ValueError(
                "Invalid type for `A`; must be array_like or a Pandas DataFrame."
            )

        # Make sure empty array has the correct shape
        if np.sum(A.shape) == 1:
            A = A.reshape((0, 1))

        A = np.ascontiguousarray(A, dtype=float)

    return A


def _check_A(A, n_compounds):
    if A is not None:
        if A.shape[1] != n_compounds:
            raise ValueError(
                "Dimension mismatch between `c0` and the constraint matrix `A`."
            )
        if len(A) > 0 and np.linalg.matrix_rank(A) != A.shape[0]:
            raise ValueError("`A` must have full row rank.")
        if np.isinf(A).any():
            raise ValueError("All entries in `A` must be finite.")
        if np.isnan(A).any():
            raise ValueError("No NaN values are allowed in `A`.")
        if np.any(A < 0):
            raise ValueError("`A` must have all nonnegative entries.")


def _parse_G(G, names, c0_from_df):
    if G is not None:
        if type(G) == list or type(G) == tuple:
            G = np.array(G, order="C", dtype=float)

        if type(G) == np.ndarray:
            if len(np.shape(G)) == 2 and (np.shape(G)[1] == 1 or np.shape(G)[0] == 1):
                G = np.ascontiguousarray(G.flatten())
            elif len(np.shape(G)) != 1:
                raise ValueError("`G` is the wrong shape.")
            names_from_df = None
        elif type(G) == pd.core.frame.DataFrame:
            if len(G) != 1:
                raise ValueError(
                    "If `G` is inputted as a DataFrame, it must have exactly one row."
                )
            names = list(G.columns)
            names_from_df = _check_names_df(names, names_from_df)
            G = np.ascontiguousarray(G.to_numpy(dtype=float, copy=True).flatten())
        elif type(G) == pd.core.series.Series:
            names = list(G.index)
            G = np.ascontiguousarray(df.to_numpy(dtype=float, copy=True))

    return G


def _check_G(A, G):
    if G is not None:
        if len(G) != A.shape[1]:
            raise ValueError("`G` must have `A.shape[1]` entries")
        if np.isinf(G).any():
            raise ValueError("All `G`'s must be finite.")
        if np.isnan(G).any():
            raise ValueError("No NaN values are allowed for `G`.")


def _NK_from_df(df, names, c0_from_df):
    if not c0_from_df:
        raise ValueError("If `N` is given as a data frame, so too must `c0`.")

    N = df[names].to_numpy(dtype=float, copy=True)

    if "equilibrium constant" in df:
        K = df["equilibrium constant"].values.astype(float)
    elif "equilibrium_constant" in df:
        K = df["equilibrium_constant"].values.astype(float)
    else:
        K = None

    return np.ascontiguousarray(N), np.ascontiguousarray(K)


def _check_names_df(names, names_from_df):
    if names is None:
        names = names_from_df
    else:
        if len(names) != len(names_from_df):
            raise ValueError("Mismatch in provided names of chemical species.")

        for name in names:
            if name not in names_from_df:
                raise ValueError(
                    "Mismatch in provided names of chemical species. '{name}' is problematic."
                )

    return list(names)


def _N_from_array(N):
    if len(N.shape) == 1:
        N = N.reshape((1, -1), order="C").astype(float)

    return np.ascontiguousarray(N, dtype=float)


def _A_from_df(df, names, c0_from_df):
    if not c0_from_df:
        raise ValueError("If `A` is given as a data frame, so too must `c0`.")

    A = np.empty((len(df), len(names)), dtype=float, order="C")
    for i, name in enumerate(names):
        if name not in df:
            raise ValueError(f"Name mismatch between `c0` and `A`, {name}.")
        A[i, :] = df[name].values

    return np.ascontiguousarray(A)


def _G_from_df(df, names, c0_from_df):
    if not c0_from_df:
        raise ValueError("If `G` is given as a data frame, so too must `c0`.")

        if len(df) != 1:
            raise ValueError(
                "If `G` is inputted as a DataFrame, it must have exactly one row."
            )
        names_from_df = list(df.columns)
        names = _check_names_df(names, names_from_df)

        G = G.to_numpy(dtype=float, copy=True).flatten()

    G = np.empty((len(df), len(names)), dtype=float, order="C")
    for i, name in enumerate(names):
        if name not in df:
            raise ValueError(f"Name mismatch between `c0` and `A`, {name}.")
        A[i, :] = df[name].values

    return np.ascontiguousarray(A)


def _c0_from_df(df, names):
    return df[names].to_numpy(dtype=float, copy=True)


def _check_name_close_to_eqconst(names):
    for name in names:
        if (
            _levenshtein(name, "equilibrium constant") < 10
            or _levenshtein(name, "equilibrium_constant") < 10
        ):
            warnings.warn(
                f"Chemical species name '{name}' is close to 'equilibrium constant'. This may be a typo."
            )


def _levenshtein(s1, s2):
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _parse_rxn(rxn):
    N_dict = {}

    # Parse equilibrium constant
    if ";" in rxn:
        if rxn.count(";") > 1:
            raise ValueError("One one semicolon is allowed in reaction specification.")

        K_str = rxn[rxn.index(";") + 1 :].strip()
        if _is_positive_number(K_str):
            K = float(K_str)
        else:
            raise ValueError(
                "Equilibrium constant cannot be converted to positive float."
            )

        # Chopp equilibrium constant from the end of the string
        rxn = rxn[: rxn.index(";")]
    else:
        K = None

    # Ensure there is exactly one <=> operator and put spaces around it
    if rxn.count("<=>") != 1:
        raise ValueError("A reaction must have exactly one '<=>' operator.")

    op_index = rxn.find("<=>")
    lhs_str = rxn[:op_index]
    rhs_str = rxn[op_index + 3 :]

    lhs_elements = [s.strip() for s in lhs_str.split(" + ") if s.strip()]
    rhs_elements = [s.strip() for s in rhs_str.split(" + ") if s.strip()]

    for element in lhs_elements:
        _parse_element(N_dict, element, -1)
    for element in rhs_elements:
        _parse_element(N_dict, element, 1)

    return N_dict, K


def _is_positive_number(s):
    try:
        num = float(s)
        return True if num > 0 else False
    except ValueError:
        return False


def _parse_element(N_dict, element, sgn):
    term = element.split()
    if len(term) == 1:
        if term[0] not in N_dict:
            N_dict[term[0]] = sgn * 1.0
        else:
            N_dict[term[0]] += sgn * 1.0
    elif len(term) == 2:
        if _is_positive_number(term[0]):
            if term[1] not in N_dict:
                N_dict[term[1]] = sgn * float(term[0])
            else:
                N_dict[term[1]] += sgn * float(term[0])
        else:
            raise ValueError(f"Invalid term '{element}' in reaction.")
    else:
        raise ValueError(f"Invalid term '{element}' in reaction.")
