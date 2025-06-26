from typing import TypeVar, Generic
from dataclasses import dataclass
from slp_eval.automata import DFA

A = TypeVar('A')
S = TypeVar('S')

@dataclass
class SLP(Generic[A]):
    """Straight line program over alphabet A"""

    constants: list[A]
    instructions: list[tuple[int, int]]

    def __init__(self, constants: list[A], instructions: list[tuple[int, int]]):
        self.constants = constants
        self.instructions = instructions


    def evaluate(self) -> list[A]:
        if len(self.constants)==0:
            return []
        s: list[list[A]] = [[c] for c in self.constants]
        for i, j in self.instructions:
            r = s[i] + s[j]
            s.append(r)
        return s[-1]

    def run_dfa(self, dfa: DFA[A, S]) -> bool:
        #base case maps
        tr_maps: list[dict[S, S]] = []
        for c in self.constants:
            tr_maps.append({q: dfa.delta[(q, c)] for q in dfa.states})

        # composition for each instruction
        for i, j in self.instructions:
            tr_i = tr_maps[i]
            tr_j = tr_maps[j]
            tr_maps.append({q: tr_j[tr_i[q]] for q in dfa.states})

        
        final_map = tr_maps[-1]
        end_state = final_map[dfa.init]
        return end_state in dfa.final
