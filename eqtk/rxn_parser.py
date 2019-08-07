import warnings


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
