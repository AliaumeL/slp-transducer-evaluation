import pytest
from slp_eval.transducer_models import SST
from slp_eval.compression_model import SLP

@pytest.fixture
def identity_sst() -> SST[str, str, str, str]:
    sst = SST(
        states={"q0"},
        registers={"r"},
        delta = {
            ("q0", 'a'): "q0",
            ("q0", 'b'): "q0",
        },
        reg_update = {
            ("q0", 'a'): {"r" : ["r",'a']},
            ("q0", 'b'): {"r" : ["r",'b']},
        },
        output_fn = {
            "q0" : ["r"],
        },
        init_state = "q0",
        init_regs = {
            "r" : [],
        } 
    )
    return sst

@pytest.fixture
def reverse_sst() -> SST[str, str, str, str]:
    sst = SST(
        states={"q0"},
        registers={"r"},
        delta = {
            ("q0", 'a'): "q0",
            ("q0", 'b'): "q0",
        },
        reg_update = {
            ("q0", 'a'): {"r" : ['a', "r"]},
            ("q0", 'b'): {"r" : ['b', "r"]},
        },
        output_fn = {
            "q0" : ["r"],
        },
        init_state = "q0",
        init_regs = {
            "r" : [],
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