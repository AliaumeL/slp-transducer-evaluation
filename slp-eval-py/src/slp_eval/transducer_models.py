from dataclasses import dataclass
from typing import TypeVar, Generic, Union, cast

I = TypeVar('I') # Input Alphabet # noqa
O = TypeVar('O') # Output Alphabet # noqa
S = TypeVar('S') # States # noqa
R = TypeVar('R') # Registers # noqa

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


    def deltastar(self, word: list[I], state: S, registers : dict[R, list[O]]) -> S:
        """Run the SST over the input `word`, updating `state` and `registers` in-place."""
        for symbol in word:
            key = (state, symbol)
            if key not in self.delta:
                raise ValueError(f"No transition defined for state {state} and symbol {symbol}")
        
            # Move to next state
            next_state = self.delta[key]

            # Update registers
            update = self.reg_update[key]
            new_registers : dict[R, list[O]] = {}

            for reg in self.registers:
                if reg not in update:
                    new_registers[reg] = registers[reg][:]
                    continue

                rhs = update[reg]
                new_val = []
                used_regs = set()
                for x in rhs:
                    if isinstance(x, str) and x in self.registers:
                        if x in used_regs:
                            raise ValueError(f"Copyless violation: register {x} used more than once")
                        used_regs.add(x)
                        new_val.extend(registers[cast(R, x)])
                    else:
                        new_val.append(x)
                new_registers[reg] = new_val

            # Mutate the original register dictionary
            for reg in self.registers:
                registers[reg] = new_registers[reg]

            # Move to next state
            state = next_state

        return state

    def run_on_list(self, w : list[I]) -> list[O]:
        state = self.init_state
        regs = {r : list(val) for r, val in self.init_regs.items()}

        for symbol in w:
            if (state, symbol) not in self.delta:
                return [] # Unkown transition (In case SST does not define full function: sigma -> gamma)

            next_state = self.delta[(state, symbol)]
            updates = self.reg_update[(state, symbol)]

            new_regs = {}

            for r in self.registers:
                if r in updates:
                    rhs = updates[r]
                    evaluated = []
                    used_registers = set()
                    for item in rhs:
                        if item in self.registers:
                            if item in used_registers:
                                raise ValueError("Copyless restriction violated: register used more than once")
                            evaluated += regs[cast(R, item)]
                            used_registers.add(item)
                        else:
                            evaluated.append(item)
                    new_regs[r] = evaluated
                else:
                    new_regs[r] = regs[r]  # unchanged

            regs = new_regs
            state = next_state

        # After processing input, compute the output using the output function
        output = []
        for item in self.output_fn.get(state, []):
            if item in self.registers:
                output += regs[cast(R, item)]
            else:
                output.append(item)

        return output
