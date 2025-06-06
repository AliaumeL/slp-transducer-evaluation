from dataclasses import dataclass


@dataclass
class Mealy[I,O,S]:
    """ A (deterministic complete) mealy machine with
        input alphabet I, output alphabet O, and state
        space S.
    """
    states: set[S]
    delta : dict[tuple[S,I], S]
    lambda: dict[tuple[S,I], O]
    init  : S

@dataclass
class Sequential[I,O,S]:
    """ A sequential function, i.e., a
        deterministic transducer with word
        outputs
    """
    pass

@dataclass
class BiMachine[I,O,S]:
    """ Bi-machine """
    pass

@dataclass
class UFT[I,O,S]:
    """ Unambiguous finite state transducer with
        word outputs
    """
    pass

@dataclass
class TDFT[I,O,S]:
    """ deterministic two-way transducer with outputs """
    pass

