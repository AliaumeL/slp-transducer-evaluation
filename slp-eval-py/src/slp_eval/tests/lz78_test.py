import pytest
from  slp_eval.lz78 import  LZ78


@pytest.fixture
def lz78() -> LZ78[str]:
    return LZ78[str]()


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
def test_correctness_of_lz78(lz78: LZ78[str], input_str: str) -> None:
    data = list(input_str)
    compressed = lz78.compress(data)
    decompressed = lz78.decompress(compressed)
    assert decompressed == data, f"Round-trip failed for input: {input_str}"
