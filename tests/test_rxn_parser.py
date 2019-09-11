import pytest
import numpy as np

import eqtk


def test_invalid_equal_operators():
    rxn = "A + B<-> AB"
    with pytest.raises(ValueError) as excinfo:
        N_dict, K = eqtk.parsers._parse_rxn(rxn)
    excinfo.match("A reaction must have exactly one '<=>' or '⇌' operator.")

    rxn = "A + B= AB"
    with pytest.raises(ValueError) as excinfo:
        N_dict, K = eqtk.parsers._parse_rxn(rxn)
    excinfo.match("A reaction must have exactly one '<=>' or '⇌' operator.")

    rxn = "A + B -> AB"
    with pytest.raises(ValueError) as excinfo:
        N_dict, K = eqtk.parsers._parse_rxn(rxn)
    excinfo.match("A reaction must have exactly one '<=>' or '⇌' operator.")

    rxn = "A + B <==> AB"
    with pytest.raises(ValueError) as excinfo:
        N_dict, K = eqtk.parsers._parse_rxn(rxn)
    excinfo.match("A reaction must have exactly one '<=>' or '⇌' operator.")

    rxn = "A + B => AB"
    with pytest.raises(ValueError) as excinfo:
        N_dict, K = eqtk.parsers._parse_rxn(rxn)
    excinfo.match("A reaction must have exactly one '<=>' or '⇌' operator.")


def test_equals_in_chemical_formula():
    rxn = "C=C <=> C + C"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == {"C=C": -1.0, "C": 2.0}


def test_parse_rxn():
    rxn = "A + A <=> A2"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == {"A": -2.0, "A2": 1.0}

    rxn = "A + A <=> A2 ; 1.2"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K == 1.2
    assert N_dict == {"A": -2.0, "A2": 1.0}

    rxn = " <=> A + B + C"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == {"A": 1.0, "B": 1.0, "C": 1.0}

    rxn = "A + B + C <=> ; 10.0"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K == 10.0
    assert N_dict == {"A": -1.0, "B": -1.0, "C": -1.0}


def test_cantera_examples():
    N_dict_target = {"CH2": -2.0, "CH": 1.0, "CH3": 1.0}
    rxn = "2 CH2 <=> CH + CH3"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == N_dict_target

    rxn = "2 CH2<=>CH + CH3"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == N_dict_target

    # Here we differ from Cantera; this is not an error
    # We allow compound names to start with a number.
    rxn = "2CH2 <=> CH + CH3"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == {"2CH2": -1.0, "CH": 1.0, "CH3": 1.0}

    rxn = "CH2 + CH2 <=> CH + CH3"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == N_dict_target

    # Here we also differ from Canter; this is not an error
    # We allow plus symbols in compounds (signify charge)
    rxn = "2 CH2 <=> CH+CH3"
    N_dict, K = eqtk.parsers._parse_rxn(rxn)
    assert K is None
    assert N_dict == {"CH2": -2.0, "CH+CH3": 1.0}


def test_invalid_K():
    rxns = """
    2 A <=> AA   ; 1.0
    A + B <=> AB ; 0.2
    2 B <=> BB
    """
    with pytest.raises(ValueError) as excinfo:
        N, K, species = eqtk.parse_rxns(rxns)
    excinfo.match("Either all or none of the equilibrium constants must be specified.")

    rxn = """
    2 A <=> AA   ; 1.0
    A + B <=> AB
    2 B <=> BB
    """
    with pytest.raises(ValueError) as excinfo:
        N, K, species = eqtk.parse_rxns(rxns)
    excinfo.match("Either all or none of the equilibrium constants must be specified.")

    rxn = """
    2 A <=> AA   ; 1.0
    A + B <=> AB ;
    2 B <=> BB ;
    """
    with pytest.raises(ValueError) as excinfo:
        N, K, species = eqtk.parse_rxns(rxns)
    excinfo.match("Either all or none of the equilibrium constants must be specified.")


def test_is_positive_number():
    assert not eqtk.parsers._is_positive_number(" ")
    assert not eqtk.parsers._is_positive_number("")
    assert not eqtk.parsers._is_positive_number("nan")
    assert not eqtk.parsers._is_positive_number("np.inf")
    assert not eqtk.parsers._is_positive_number("1f")
    assert not eqtk.parsers._is_positive_number("blah")
    assert not eqtk.parsers._is_positive_number("11111.111.1")
    assert not eqtk.parsers._is_positive_number("129.f")
    assert not eqtk.parsers._is_positive_number("0")
    assert not eqtk.parsers._is_positive_number("0.0")
    assert not eqtk.parsers._is_positive_number("-1")
    assert not eqtk.parsers._is_positive_number("-2")
    assert not eqtk.parsers._is_positive_number("-6")
    assert not eqtk.parsers._is_positive_number("-1.6")
    assert not eqtk.parsers._is_positive_number("-1e4")
    assert not eqtk.parsers._is_positive_number("-7.8e-6")
    assert not eqtk.parsers._is_positive_number("-0.01")
    assert eqtk.parsers._is_positive_number("1")
    assert eqtk.parsers._is_positive_number("2")
    assert eqtk.parsers._is_positive_number("6")
    assert eqtk.parsers._is_positive_number("1.6")
    assert eqtk.parsers._is_positive_number("1e4")
    assert eqtk.parsers._is_positive_number("7.8e-6")
    assert eqtk.parsers._is_positive_number("0.01")


def test_parse_element():
    N_dict = {}
    element = "2 A"
    eqtk.parsers._parse_element(N_dict, element, -1)
    assert N_dict == {"A": -2.0}

    N_dict = {}
    element = "2A"
    eqtk.parsers._parse_element(N_dict, element, -1)
    assert N_dict == {"2A": -1.0}

    N_dict = {}
    element = "2 A2+"
    eqtk.parsers._parse_element(N_dict, element, -1)
    assert N_dict == {"A2+": -2.0}

    N_dict = {}
    element = "2 A"
    eqtk.parsers._parse_element(N_dict, element, 1)
    assert N_dict == {"A": 2.0}

    N_dict = {}
    element = "2A"
    eqtk.parsers._parse_element(N_dict, element, 1)
    assert N_dict == {"2A": 1.0}

    N_dict = {}
    element = "2 A2+"
    eqtk.parsers._parse_element(N_dict, element, 1)
    assert N_dict == {"A2+": 2.0}

    N_dict = {"A": 3.0}
    element = "2 A"
    eqtk.parsers._parse_element(N_dict, element, -1)
    assert N_dict == {"A": 1.0}

    N_dict = {"A": 3.0}
    element = "2 A"
    eqtk.parsers._parse_element(N_dict, element, 1)
    assert N_dict == {"A": 5.0}
