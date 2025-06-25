from dataclasses import dataclass
from typing import TypeVar, Generic, Union

I = TypeVar('I') # Input Alphabet
O = TypeVar('O') # Output Alphabet
S = TypeVar('S') # States
R = TypeVar('R') # Registers


@dataclass
class Mealy(Generic[I, O, S]):
    """A (deterministic complete) mealy machine with
    input alphabet I, output alphabet O, and state
    space S.
    """
    states: set[S]
    delta: dict[tuple[S, I], S]
    lambda: dict[tuple[S, I], O]
    init: S


@dataclass
class Sequential(Generic[I, O, S]):
    """A sequential function, i.e., a
    deterministic transducer with word
    outputs
    """

    pass


@dataclass
class BiMachine(Generic[I, O, S]):
    """Bi-machine"""

    pass


@dataclass
class UFT(Generic[I, O, S]):
    """Unambiguous finite state transducer with
    word outputs
    """

    pass


@dataclass
class TDFT(Generic[I, O, S]):
    """deterministic two-way transducer with outputs"""

    pass

@dataclass
class SST(Generic[I, O, S, R]):
    """ Streaming String Transducer with copyless restriction
    Input alphabet I, output alphabet O, state space S, register list R"""
    states: set[S]
    registers: set[R]
    input_lang: set[I]
    output_lang: set[O]
    init_state: S 
    init_regs: dict[R, list[O]] # initial register valuation

    delta: dict[tuple[S, I], S]
    reg_update: dict[tuple[S, I], dict[R, list[Union[R, O]]]]  # Copyless: right-hand side uses registers at most once
    output_fn: dict[S, list[Union[R, O]]]
