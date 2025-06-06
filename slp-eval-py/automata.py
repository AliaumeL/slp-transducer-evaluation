from dataclasses import dataclass


@dataclass
class DFA[A,S]:
    """ A (deterministic complete) finite state automaton """
    states: set[S]
    delta : dict[tuple[S,A], S]
    final : set[S]
    init  : S

    def to_transition_monoid(self, w : list[A]) -> dict[S,S]:
        tr = { s: s for s in self.states }
        for a in w:
            tr = { p: self.delta[(q,a)] for (p,q) in tr.items() }
        return tr


    def is_accepting(self, w : list[A]) -> bool:
        s = self.init
        for a in w:
            s = self.delta[(s,a)]
        return (s in self.final)


@dataclass
class NFA[A,S]:
    """ A non-deterministic finite state automaton """
    states: set[S]
    delta : set[tuple[S,A,S]]
    final : set[S]
    init  : S


    def to_transition_monoid(self, w : list[A]) -> dict[S,S]:
        # pyrefly: ignore
        return "TODO implement" 

    def is_accepting(self, w : list[A]) -> bool:
        # pyrefly: ignore
        return "TODO implement" 

    def determinize(self) -> DFA[A, set[S]]:
        # pyrefly: ignore
        return "TODO implement" 


@dataclass
class TDFA[A,S]:
    """ A two-way deterministic finite state automaton """
    pass # TODO: implement


