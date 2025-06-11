from slp_eval.compression_model import SLP
import pytest
from slp_eval.tests.automata_test import ends_with_b_dfa, ends_with_a_dfa
from slp_eval.automata import DFA

def test_super_compress():
    for n in range(10):
        slp = SLP(
            constants=["a"],
            instructions=[(i,i) for i in range(n)],
        ) 
        v = slp.evaluate()
        assert (all(x == "a" for x in v)) 
        assert len(v) == 2**n



@pytest.fixture
def slp_aba() -> SLP[str]:
    """ SLP for 'aba' """
    constants: list[str] = ['a', 'b']
    instructions: list[tuple[int, int]] = [(0, 1), (2, 0)]
    return SLP(constants=constants, instructions=instructions)

@pytest.fixture
def slp_abaab() -> SLP[str]:
    """SLP for 'abaab'."""    
    constants: list[str] = ['a', 'b']
    instructions: list[tuple[int, int]] = [(0, 1), (2, 0), (3,2)]
    return SLP(constants=constants, instructions=instructions)

@pytest.fixture
def slp_single_c() -> SLP[str]:
    """
    SLP for 'c'.
    constants = ['c']
    instructions = []  # no concatenation needed; final is constants[0]
    """
    constants: list[str] = ['c']
    instructions: list[tuple[int, int]] = []
    return SLP(constants=constants, instructions=instructions)


def test_slp_decompression(
    slp_aba,
    slp_abaab,
    slp_single_c
):
    """For each SLP fixture, check decompress() yields the expected string."""
    cases = [
        (slp_abaab, "abaab"),
        (slp_aba, "aba"),
        (slp_single_c, "c"),
    ]

    for slp, expected in cases:
        result = slp.evaluate()
        assert result == list(expected), f"Expected decompression '{expected}', got '{result}'"
    


@pytest.mark.parametrize(
    "slp_fixture,dfa_fixture,expected",
    [
        ("slp_aba", "ends_with_b_dfa", False),
        ("slp_aba", "ends_with_a_dfa", True),
        
        ("slp_abaab", "ends_with_b_dfa", True),
        ("slp_abaab", "ends_with_a_dfa", False)
        
    ],
    ids=[
        "aba_on_ends_with_a",
        "aba_on_ends_with_b",
        "abaab_on_ends_with_a",
        "abaab_on_ends_with_b"
    ]
)
def test_run_dfa_on_slp(request, slp_fixture: str, dfa_fixture: str, expected: bool):
    """
    Decompress the SLP fixture, convert to symbol list, and assert DFA accepts/rejects as expected.
    """
    slp = request.getfixturevalue(slp_fixture)
    dfa = request.getfixturevalue(dfa_fixture)
    assert slp.run_dfa(dfa) is expected