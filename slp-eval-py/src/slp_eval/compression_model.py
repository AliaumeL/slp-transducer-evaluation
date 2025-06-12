from typing import TypeVar, Generic, Union, cast
from dataclasses import dataclass
from slp_eval.automata import DFA
from slp_eval.transducer_models import SST

A = TypeVar('A')
S = TypeVar('S')
I = TypeVar('I')
O = TypeVar('O')
R = TypeVar('R')

# changed A to I everywhere to match inconsistency

@dataclass
class SLP(Generic[I]):
    """Straight line program over alphabet A"""
    constants: list[I]
    instructions: list[tuple[int, int]]

    def evaluate(self) -> list[I]:
        s: list[list[I]] = [[c] for c in self.constants]
        for i, j in self.instructions:
            r = s[i] + s[j]
            s.append(r)
        return s[-1]

    def run_dfa(self, dfa: DFA[I, S]) -> bool:
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

    def run_SST(self, sst: SST[I, O, S, R]) -> list[O]:
        """ In the transition map, a state is mapped to the next state along with a 
        list of all register updates after parsing an input constant or instruction"""
        
        tr_maps: list[dict[S, tuple[S, dict[R, list[Union[R, O]]]]]] = []

        # constant maps
        for c in self.constants:
            tr_maps.append({q : (sst.delta[(q, c)], sst.reg_update[(q, c)]) for q in sst.states})

        # composition instruction maps
        for i, j in self.instructions:
            tr_i = tr_maps[i]
            tr_j = tr_maps[j]
            composed_tr : dict[S, tuple[S, dict[R, list[Union[R, O]]]]] = {}

            for q in sst.states:
                q_i : S = tr_i[q][0]
                reg_map_i : dict[R, list[Union[R, O]]] = tr_i[q][1]
                q_j : S = tr_j[q_i][0]
                reg_map_j : dict[R, list[Union[R, O]]] = tr_j[q_i][1]

                composed_reg_update: dict[R, list[Union[R, O]]] = {}

                for r, update_expr in reg_map_j.items():
                    new_expr: list[Union[R, O]] = []
                    for token in update_expr:
                        if token in reg_map_i:
                            new_expr.extend(reg_map_i[token])  # substitute token # type: ignore
                        else:
                            new_expr.append(token)

                    composed_reg_update[r] = new_expr

                composed_tr[q] = (q_j, composed_reg_update)

            tr_maps.append(composed_tr)
        
        # output function
        final_map = tr_maps[-1]
        end_state, reg_map = final_map[sst.init_state]

        # replacing reg symbols with initial values
        end_reg_map: dict[R, list[O]] = {}
        for r, r_update in reg_map.items():
            new_update: list[O] = []
            for token in r_update:
                if token in sst.init_regs:
                    new_update.extend(sst.init_regs[token])  # substitute token #type: ignore 
                else:
                    new_update.append(token) #type: ignore 
            end_reg_map[r] = new_update

        # composing final output
        output_expr = sst.output_fn[end_state]
        output : list[O] = []
        for token in output_expr:
            if token in end_reg_map:
                output.extend(end_reg_map[token]) # type: ignore
            else:
                output.append(token) # type: ignore
        
        return output
    

