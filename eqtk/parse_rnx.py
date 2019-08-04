import warnings

def parse_rxns(rxns):
    rxn_list = rxns.splitlines()
    N_dict_list = []
    K_list = []
    for rxn in rxn_list:
        if rxn != '':
            N_dict, K = _parse_rxn(rxn)
            N_dict_list.append(N_dict)
            K_list.append(K)

    # Make sure K's are specified for all or none
    if not(all([K is None for K in K_list]) or all([K is not None for K in K_list])):
        raise ValueError('Either all or none of the equilibrium constants must be specified.')

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

    equal_operators = ("=", "<=>", "<=>")

    # Check to see if there is a semicolon with no spaces
    if ";" in rxn:
        if rxn.count(";") > 1:
            raise ValueError("One one semicolon is allowed in reaction specification.")
        if " ; " not in rxn:
            raise ValueError("`;` must be separated with whitespace.")

    elements = rxn.split()

    if ";" in elements:
        end_index = elements.index(";")
        if end_index != len(elements) - 2:
            raise ValueError(
                "`;` must be followed by a number giving equilibrium constant."
            )
        if _is_number(elements[-1]):
            K = float(elements[-1])
        else:
            raise ValueError("Equilibrium constant cannot be converted to float.")
    else:
        K = None
        end_index = len(elements)

    if sum([elements.count(eqop) for eqop in equal_operators]) != 1:
        raise ValueError("Each reaction must have exactly one `=` operator.")

    if "=" in elements:
        equal_operator = "="
    elif "<=>" in elements:
        equal_operator = "<=>"
    else:
        equal_operator = "<->"

    equal_operator_index = elements.index(equal_operator)

    # Left of equal operator, every other symbol must be a `+`
    if equal_operator_index % 2 != 1:
        raise ValueError(f"Improper left-hand side of chemical reaction: {rxn}.")

    if elements[:equal_operator_index].count("+") != equal_operator_index // 2:
        raise ValueError(f"Improper left-hand side of chemical reaction: {rxn}.")

    for i in range(1, equal_operator_index, 2):
        if elements[i] != "+":
            raise ValueError(f"Improper left-hand side of chemical reaction: {rxn}.")

    # Same must be true right of equal operator
    if (end_index - equal_operator_index) % 2 != 0:
        raise ValueError(f"Improper right-hand side of chemical reaction: {rxn}.")

    if (
        elements[equal_operator_index + 1 :].count("+")
        != (end_index - equal_operator_index - 1) // 2
    ):
        raise ValueError(f"Improper right-hand side of chemical reaction: {rxn}.")

    for i in range(equal_operator_index + 2, end_index, 2):
        if elements[i] != "+":
            raise ValueError(f"Improper right-hand side of chemical reaction: {rxn}.")

    # Obtain species and stoichiometric coefficients
    left_species = [_parse_stoich_coeff(s) for s in elements[:equal_operator_index:2]]
    right_species = [
        _parse_stoich_coeff(s)
        for s in elements[equal_operator_index + 1 : end_index : 2]
    ]

    for species in left_species:
        if species[1] not in N_dict:
            N_dict[species[1]] = -species[0]
        else:
            N_dict[species[1]] -= species[0]

    for species in right_species:
        if species[1] not in N_dict:
            N_dict[species[1]] = species[0]
        else:
            N_dict[species[1]] += species[0]

    return N_dict, K


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_stoich_coeff(s):
    """
    Stoichiometic coefficient * compound name.

    E.g.:
    2*A
    3*2-butanol

    Implied stoich coefficient of unity:
    2-butanol
    *A
    A*B*C
    2A

    """
    if "*" in s:
        if s[0] == "*":
            return 1.0, s
        elif s[-1] == "*":
            return 1.0, s

        stoich = s[: s.index("*")]

        if _is_number(stoich):
            return float(stoich), s[s.index("*") + 1 :]

        return 1.0, s

    if s[0].isdigit() and len(s) > 1 and not s[1].isdigit():
        warnings.warn(f"Interpreting '{s}'' as a chemical species, i.e., with no stoichiometic coefficient. You might have intended '2*{s[1:]}'.")

    return 1.0, s
