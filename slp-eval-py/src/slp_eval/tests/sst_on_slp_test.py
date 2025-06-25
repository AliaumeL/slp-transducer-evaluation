import pytest
from slp_eval.transducer_models import SST
from slp_eval.compression_model import SLP

#Fixtures

@pytest.fixture
def identity_sst() -> SST[str, str, str, str]:
    sst = SST(
        states={"q0"},
        registers={"R"},
        input_lang={'a', 'b'},
        output_lang={'a', 'b'},
        delta = {
            ("q0", 'a'): "q0",
            ("q0", 'b'): "q0",
        },
        reg_update = {
            ("q0", 'a'): {"R" : ["R",'a']},
            ("q0", 'b'): {"R" : ["R",'b']},
        },
        output_fn = {
            "q0" : ["R"],
        },
        init_state = "q0",
        init_regs = {
            "R" : [],
        } 
    )
    return sst

@pytest.fixture
def reverse_sst() -> SST[str, str, str, str]:
    sst = SST(
        states={"q0"},
        registers={"R"},
        input_lang={'a', 'b'},
        output_lang={'a', 'b'},
        delta = {
            ("q0", 'a'): "q0",
            ("q0", 'b'): "q0",
        },
        reg_update = {
            ("q0", 'a'): {"R" : ['a', "R"]},
            ("q0", 'b'): {"R" : ['b', "R"]},
        },
        output_fn = {
            "q0" : ["R"],
        },
        init_state = "q0",
        init_regs = {
            "R" : [],
        } 
    )
    return sst

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
def w() -> list[str]:
    return list("")

# Testcases

def test_identity_sst_on_aba(identity_sst, slp_aba):
    output = slp_aba.run_SST(identity_sst)
    assert output == list("aba")

def test_identity_sst_on_abaab(identity_sst, slp_abaab):
    output = slp_abaab.run_SST(identity_sst)
    assert output == list("abaab")

def test_reverse_sst_on_aba(reverse_sst, slp_aba):
    output = slp_aba.run_SST(reverse_sst)
    assert output == list("aba")

def test_reverse_sst_on_abaab(reverse_sst, slp_abaab):
    output = slp_abaab.run_SST(reverse_sst)
    assert output == list("baaba")

def test_string_to_slp(w):
    slp = SLP.string_to_slp_naive(w)
    output = slp.evaluate()
    print(slp.constants)
    print(slp.instructions)
    assert output == w

def test_identity_sst_output_on_aba(identity_sst, slp_aba):
    # Get output SLP
    output_slp = slp_aba.run_SST_output_slp(identity_sst, "")
    # Evaluate the resulting SLP
    output = output_slp.evaluate()
    output = ''.join(output) # gets rid of empty symbol 
    assert output == "aba"

def test_identity_sst_output_on_abaab(identity_sst, slp_abaab):
    # Get output SLP
    output_slp = slp_abaab.run_SST_output_slp(identity_sst, "")
    # Evaluate the resulting SLP
    output = output_slp.evaluate()
    output = ''.join(output) # gets rid of empty symbol 
    assert output == "abaab"

def test_reverse_sst_output_on_abaab(reverse_sst, slp_abaab):
    # Get output SLP
    output_slp = slp_abaab.run_SST_output_slp(reverse_sst, "")
    # Evaluate the resulting SLP
    output = output_slp.evaluate()
    output = ''.join(output) # gets rid of empty symbol 
    assert output == "baaba"