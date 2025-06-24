import pytest
from slp_eval.automata import DFA


@pytest.fixture
def ends_with_b_dfa() -> DFA[str, str]:
      
    """DFA over {a, b} that accepts strings ending in 'b'."""

    dfa = DFA(
        states={"q0", "q1"},
        init="q0",
        final={"q1"},
        delta={
            ("q0", 'a'): "q0",
            ("q0", 'b'): "q1",
            ("q1", 'a'): "q0",
            ("q1", 'b'): "q1",
        }
    )
    return dfa

@pytest.fixture
def ends_with_a_dfa() -> DFA[str, str]:
      
    """DFA over {a, b} that accepts strings ending in 'a'."""

    dfa = DFA(
        states={"q0", "q1"},
        init="q0",
        final={"q1"},
        delta={
            ("q0", 'a'): "q1",
            ("q0", 'b'): "q0",
            ("q1", 'a'): "q1",
            ("q1", 'b'): "q0",
        }
    )
    return dfa


def test_ends_with_b_dfa(ends_with_b_dfa):
    assert not ends_with_b_dfa.is_accepting(list("a"))
    assert ends_with_b_dfa.is_accepting(list("ab"))
    assert ends_with_b_dfa.is_accepting(list("bbbb"))
    assert not ends_with_b_dfa.is_accepting(list("baba"))