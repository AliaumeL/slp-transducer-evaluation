
from dataclasses import dataclass

@dataclass
class Automaton[A,S]:
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

def main():
    print("Hello from slp-eval-py!")


if __name__ == "__main__":
    main()
