from typing import TypeVar, Generic, Union, cast
from dataclasses import dataclass
from slp_eval.automata import DFA
from slp_eval.transducer_models import SST
from collections import deque

# changed A to I everywhere to match inconsistency
# A = TypeVar('A') 
S = TypeVar('S')
I = TypeVar('I')
O = TypeVar('O')
R = TypeVar('R')


@dataclass
class SLP(Generic[I]):
    """Straight line program over alphabet I"""
    constants: list[I]
    instructions: list[tuple[int, int]]

    # 
    def evaluate(self) -> list[I]:
        EMPTY = cast(I, "")

        # empty SLP check
        if (len(self.constants) == 0):
            return []

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
    
    def evaluate_by_uncompress(self, sst):
        return sst.run_on_list(self.evaluate())
    
    from collections import deque

    def run_sst_dumb(self, sst : SST[I,O,S,R]) -> list[O]:
        registers  = { r: v for (r,v) in sst.init_regs.items() } # copy the value 
        state      = sst.init_state
        consts     = len(self.constants)
        instrns    = len(self.instructions)
        stack      = deque()

        if len(self.instructions) == 0:
            input = self.constants[-1]
            state = sst.deltastar([input], state, registers)
        else:
            stack.append(self.instructions[-1]) # assume that there is a last instruction
            while len(stack) > 0:
                op = stack.pop()
                if isinstance(op, tuple): # this is a binary operation
                    xi, xj = op
                    if xj < consts:
                        stack.append(xj)
                    elif xj < consts + instrns:
                        stack.append(self.instructions[xj - consts])
                    else:
                        raise ValueError("Wrong register operation in SLP: " + str(xj))
                    
                    if xi < consts:
                        stack.append(xi)
                    elif xi < consts + instrns:
                        stack.append(self.instructions[xi - consts])       
                    else:
                        raise ValueError("Wrong register operation in SLP: " + xi)
                elif op < consts: # if we are computing a constant
                    state = sst.deltastar(self.constants[op], state, registers)

        output_expr = sst.output_fn[state]
        output = []
        for sym in output_expr:
            if sym in sst.registers:
                output.extend(registers[cast(R, sym)])
            else:
                output.append(sym)

        return output

    def run_sst_on_slp(self, sst: SST[I, O, S, R]) -> list[O]:
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
                            new_expr.extend(reg_map_i[cast(R, token)])  # substitute token
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
                    new_update.extend(sst.init_regs[cast(R, token)])  # substitute token 
                else:
                    new_update.append(cast(O, token))
            end_reg_map[r] = new_update

        # composing final output
        output_expr = sst.output_fn[end_state]
        output : list[O] = []
        for token in output_expr:
            if token in end_reg_map:
                output.extend(end_reg_map[cast(R, token)])
            else:
                output.append(cast(O, token))
        
        return output
    
    def run_sst_output_compression(self, sst: SST[I, O, S, R], EMPTY:O) -> 'SLP[O]':
        # state x reg x input -> new reg expression compressed as SLP
        slp_regs: list[dict[tuple[S, R], SLP]] = []
         # state x input -> state
        tr_maps: list[dict[S, S]] = []

        # SLPs to store register initial valuations
        initial_slp_regs : dict[R, SLP] = {r : (self.list_to_slp(sst.init_regs[r])) for r in sst.registers}

        for c in self.constants:
            slp_regs.append({(q, r) : (self.list_to_slp(sst.reg_update[q, c][r])) for q in sst.states for r in sst.registers})
            tr_maps.append({q : sst.delta[(q, c)] for q in sst.states})

        # Standard map from reg constants and constants to their positions used in all register SLP compositions
        const_to_idx : dict[Union[R, O], int] = {r: i + 1 for i, r in enumerate(sst.registers)}
        const_to_idx[EMPTY] = 0
        num_regs = len(sst.registers)
        for i, c in enumerate(sst.output_lang):
            const_to_idx[c] = i + num_regs + 1

        for i, j in self.instructions:
            # naive composition of the 2 slps

            tr_i = tr_maps[i]
            tr_j = tr_maps[j]

            slpreg_i = slp_regs[i]
            slpreg_j = slp_regs[j]

            composed_slpreg : dict[tuple[S, R], SLP] = {}
            composed_trmap : dict[S, S] = {}

            for q in sst.states:
                q_i = tr_i[q]
                q_j = tr_j[q_i]
                composed_trmap[q] = q_j

                # We can go through the reg update and determine which all registers are needed for the composed reg_update
                for r in sst.registers:
                    slp_j = slpreg_j[q_i, r]
                    composed_slp : SLP[Union[R, O]] = SLP(constants=[], instructions=[])

                    # consider for empty case (r = []), then there wont be any instructions
                    if (len(slp_j.instructions) == 0 and len(slp_j.constants) == 0):
                        composed_slpreg[q, r] = composed_slp
                        continue


                    reg_to_instrn : dict[R, int] = {} # need instruction co-ordinate to map reg const in slp_j to the respective reg update instruction
                    regs_list : list[R] = []

                    for x, y in slp_j.instructions:
                        if (x < len(slp_j.constants) and slp_j.constants[x] in sst.registers):
                            regs_list.append(slp_j.constants[x])
                        if (y < len(slp_j.constants) and slp_j.constants[y] in sst.registers):
                            regs_list.append(slp_j.constants[y])

                    if len(regs_list) == 0: # no reg references, only constant SLP
                        composed_slpreg[q, r] = slp_j
                        continue

                    # set up constants of new SLP
                    composed_slp.constants.append(EMPTY)
                    composed_slp.constants.extend(sst.registers)
                    composed_slp.constants.extend(sst.output_lang)
                    composed_consts = len(composed_slp.constants)
                    instrn_counter = 0

                    for reg in regs_list:
                            slp_i = slpreg_i[(q, reg)]

                            clen : int = len(slp_i.constants)
                            if (len(slp_i.instructions) == 0 and clen == 0): # reg update is empty
                                reg_to_instrn[reg] = 0 # point to EMPTY
                                continue

                            if (len(slp_i.instructions) == 0): # reg update is only a constant
                                reg_to_instrn[reg] = const_to_idx[slp_i.constants[-1]]
                                continue

                            for x, y in slp_i.instructions:
                                x_new = const_to_idx[slp_i.constants[x]] if (x < clen) else x + composed_consts - clen + instrn_counter
                                y_new = const_to_idx[slp_i.constants[y]] if (y < clen) else y + composed_consts - clen + instrn_counter
                                composed_slp.instructions.append((x_new, y_new))
                                
                            instrn_counter += len(slp_i.instructions)
                            reg_to_instrn[reg] = composed_consts + instrn_counter - 1

                    # Now process slp_j's instructions
                    clen_j : int = len(slp_j.constants)
                    for x, y in slp_j.instructions:
                        # 3 cases : x refers to reg constant, normal constant or a previous intruction
                        x_new = (reg_to_instrn[slp_j.constants[x]]) if (x < clen_j and slp_j.constants[x] in regs_list) else (const_to_idx[slp_j.constants[x]]) if (x < clen_j) else (x + instrn_counter + composed_consts - clen_j)
                        y_new = (reg_to_instrn[slp_j.constants[y]]) if (y < clen_j and slp_j.constants[y] in regs_list) else (const_to_idx[slp_j.constants[y]]) if (y < clen_j) else (y + instrn_counter + composed_consts - clen_j)
                        composed_slp.instructions.append((x_new, y_new))   
                    
                    composed_slpreg[q, r] = composed_slp

            slp_regs.append(composed_slpreg)
            tr_maps.append(composed_trmap)


        # Final Output SLP Construction
        final_state = tr_maps[-1][sst.init_state]

        final_output : list[Union[R, O]] = sst.output_fn[final_state]
        reg_map : dict[tuple[S, R], SLP] = slp_regs[-1]  # Final register SLPs

        output_slp = SLP(constants=[], instructions=[])

        output_slp.constants.append(EMPTY)
        output_slp.constants.extend(sst.registers)
        output_slp.constants.extend(sst.output_lang)
        consts = len(output_slp.constants)
        instrn_counter = 0

        # process the initial valuation slps first
        reg_to_initval : dict[R, int] = {} # to store reg -> instrn number that stores the init_valuation

        for r, slp_init in initial_slp_regs.items():
            clen = len(slp_init.constants)

            if (len(slp_init.instructions) == 0 and clen == 0): # reg update is empty
                reg_to_initval[r] = 0
                continue

            if (len(slp_init.instructions) == 0):
                reg_to_initval[r] = const_to_idx[slp_init.constants[-1]]
                continue

            for x, y in slp_init.instructions:
                x_new = const_to_idx[slp_init.constants[x]] if x < clen else x + instrn_counter + consts - clen
                y_new = const_to_idx[slp_init.constants[x]] if y < clen else y + instrn_counter + consts - clen
                output_slp.instructions.append((x_new, y_new))

            instrn_counter += clen
            reg_to_initval[r] = consts + instrn_counter - 1

        indices_queue: list[int] = [] # queue of variable positions to create the final slp for the the output string
        reg_roots : dict[R, int] = {}
        found_regs : dict[R, bool] = {r: False for r in sst.registers}


        for token in final_output:
            if token in sst.registers:
                # multiple possible as output function need not be copyless, avoid duplicate imports
                if (found_regs[cast(R, token)]):
                    indices_queue.append(reg_roots[cast(R, token)])
                    continue

                found_regs[token] = True # type: ignore

                reg_slp = reg_map[(sst.init_state, cast(R, token))]
                clen = len(reg_slp.constants)

                if (len(reg_slp.instructions) == 0 and clen == 0): # reg update is empty
                    reg_roots[cast(R, token)] = 0
                    continue

                if (len(reg_slp.instructions) == 0):
                    reg_roots[cast(R, token)] = const_to_idx[reg_slp.constants[-1]] 
                    continue                

                for x, y in reg_slp.instructions:
                    x_new = reg_to_initval[reg_slp.constants[x]] if (x < clen and reg_slp.constants[x] in sst.registers) else const_to_idx[reg_slp.constants[x]] if (x < clen) else x + instrn_counter + consts - clen
                    y_new = reg_to_initval[reg_slp.constants[y]] if (y < clen and reg_slp.constants[y] in sst.registers) else const_to_idx[reg_slp.constants[y]] if (y < clen) else y + instrn_counter + consts - clen
                    output_slp.instructions.append((x_new, y_new))

                instrn_counter += len(reg_slp.instructions)
                reg_root = consts + instrn_counter - 1
                reg_roots[cast(R, token)] = reg_root
                indices_queue.append(reg_root)

            else:
                # It's a constant output symbol
                indices_queue.append(const_to_idx[token])

        if len(indices_queue) == 0:
            return SLP(constants=[], instructions=[])

        if len(indices_queue) >= 2:
            a = indices_queue.pop(0)
            b = indices_queue.pop(0)
            output_slp.instructions.append((a, b))
            current = len(output_slp.constants) + len(output_slp.instructions) - 1

            for idx in indices_queue:
                output_slp.instructions.append((current, idx))
                current = len(output_slp.constants) + len(output_slp.instructions) - 1

        else: # length is 1
            a = indices_queue.pop(0)
            output_slp.instructions.append((0, a))             

        return output_slp
    
    # Helper functions for SST output compression
    @staticmethod
    def list_to_slp(w : list[Union[R, O]]) -> 'SLP[Union[R, O]]':

        # empty string case - needed if intial register valuations are empty
        if (len(w) == 0):
            return SLP(constants=[], instructions=[])
        
        # Keep track of const indices to be used for instructions later
        constants: list[Union[R, O]] = []
        seen = set()

        for c in w:
            if c not in seen:
                seen.add(c)
                constants.append(c)

        const_index: dict[Union[R, O], int] = {c: i for i, c in enumerate(constants)}

        instructions: list[tuple[int, int]] = []
        for i in range(1, len(w)):
            if (i == 1):
                instructions.append((const_index[w[i - 1]], const_index[w[i]]))
            else:
                instructions.append((i + len(constants) - 2, const_index[w[i]]))

        return SLP(constants=constants, instructions=instructions)