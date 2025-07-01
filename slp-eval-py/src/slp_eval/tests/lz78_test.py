import pytest
from typing import List
from slp_eval.lz78 import LZ78String  
from slp_eval.compression_model import SLP  


@pytest.fixture
def make_sequence():
    """
    Returns a factory for creating sequences (lists of single-character strings) from input_str.
    """
    return lambda s: list(s)


@pytest.mark.parametrize(
    "input_str",
    [
        "",               # empty string
        "a",              # single character
        "aaaaaa",         # repeated characters
        "abcabcabc",      # repeating pattern
        "abracadabra",    # typical string
        "abababababab",   # alternating pattern
        "mississippi",    # complex repetition
        "xyz"             # no repeats
    ],
    ids=[
        "empty",
        "single_char",
        "repeat_char",
        "repeat_pattern",
        "typical",
        "alternating",
        "complex",
        "no_repeats"
    ]
)
def test_roundtrip_from_list_to_list(make_sequence, input_str: str) -> None:
    data: List[str] = make_sequence(input_str)
    comp = LZ78String.from_list(data)
    # Check that content is a list of tuples
    codes = comp.to_codes()
    assert isinstance(codes, list)
    for item in codes:
        assert isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int)
    # Decompress and compare
    decompressed = comp.to_list()
    assert decompressed == data, f"Round-trip failed for input: {input_str}"


@pytest.mark.parametrize(
    "input_str",
    [
        "", "a", "aaaaaa", "abcabcabc", "abracadabra",
        "abababababab", "mississippi", "xyz"
    ],
    ids=[
        "empty", "single_char", "repeat_char", "repeat_pattern",
        "typical", "alternating", "complex", "no_repeats"
    ]
)

def test_codes_roundtrip(make_sequence, input_str: str) -> None:
    data = make_sequence(input_str)
    comp1 = LZ78String.from_list(data)
    codes = comp1.to_codes()
    # Recreate from codes
    comp2 = LZ78String.from_codes(codes)
    # Ensure codes match
    assert comp2.to_codes() == codes
    # Ensure decompression matches
    assert comp2.to_list() == data


@pytest.mark.parametrize(
    "input_list",
    [
        [],                     # empty list
        [1],                    # single element
        [1, 1, 1, 1],           # repeated ints
        [1, 2, 1, 2, 1, 3],     # small pattern
        [5, 6, 5, 6, 5, 7],     # another int pattern
    ],
    ids=[
        "empty_int",
        "single_int",
        "repeat_int",
        "pattern_int1",
        "pattern_int2",
    ]
)
def test_roundtrip_generic_int(input_list: List[int]) -> None:
    comp = LZ78String.from_list(input_list)
    assert comp.to_list() == input_list


@pytest.mark.parametrize(
    "input_str",
    [
        "", "a", "abab", "banana", "xyxyxy", "mississippi"
    ],
    ids=[
        "empty", "single", "alternating", "banana", "repeat_xy", "mississippi"
    ]
)
def test_to_slp_string(make_sequence, input_str: str) -> None:
    """
    Test to_slp if SLP.evaluate() reproduces the original.
    If you don't have SLP or evaluate method, you can skip/mark xfail.
    """
    data = make_sequence(input_str)
    comp = LZ78String.from_list(data)
    # Skip if no content
    slp = comp.to_slp()
    # Ensure slp is instance of SLP
    assert isinstance(slp, SLP)
    # Evaluate and compare
    evaluated = slp.evaluate()
    assert evaluated == data, f"SLP round-trip failed for input: {input_str}"


def test_invalid_from_codes() -> None:
    # Non-list input
    with pytest.raises(TypeError):
        LZ78String.from_codes("not a list")  # type: ignore
    # List with invalid items
    with pytest.raises(ValueError):
        LZ78String.from_codes([(1, 'a'), ("bad_prefix", 'b')])  # invalid prefix type # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LZ78String.from_codes([(1,), (0, 'c', 'extra')])  # wrong tuple length # type: ignore[arg-type]


def test_invalid_from_list() -> None:
    # Non-list input
    with pytest.raises(TypeError):
        LZ78String.from_list("not a list")  # type: ignore


def test_repr_and_str_methods(make_sequence):
    data = make_sequence("abc")
    comp = LZ78String.from_list(data)
    # repr should contain class name and content
    r = repr(comp)
    assert "LZ78String" in r and "content" not in r or "content=" in r
    # to_list gives back data
    assert comp.to_list() == data