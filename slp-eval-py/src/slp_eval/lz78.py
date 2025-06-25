from typing import List, Tuple, Generic, TypeVar, Dict
from dataclasses import dataclass
from .compression_model import SLP

A = TypeVar('A')

@dataclass
class LZ78String(Generic[A]):
    """A compressed sequence using LZ78 encoding."""
    content: List[Tuple[int, A]]

    @staticmethod
    def _compress(data: List[A]) -> List[Tuple[int, A]]:
        """Compress a sequence into LZ78 codes (index, item)."""
        dictionary: dict[Tuple[A, ...], int] = {}
        current: List[A] = []
        result: List[Tuple[int, A]] = []
        index = 1
        for symbol in data:
            current.append(symbol)
            prefix = tuple(current)
            if prefix not in dictionary:
                if len(current) == 1:
                    result.append((0, symbol))
                else:
                    prev_prefix = tuple(current[:-1])
                    result.append((dictionary[prev_prefix], symbol))
                dictionary[prefix] = index
                index += 1
                current.clear()
        if current:
            if len(current) == 1:
                result.append((0, current[0]))
            else:
                prev_prefix = tuple(current[:-1])
                if prev_prefix in dictionary:
                    result.append((dictionary[prev_prefix], current[-1]))
                else:
                    result.append((0, current[-1]))
        return result

    @staticmethod
    def _decompress(codes: List[Tuple[int, A]]) -> List[A]:
        """Decompress LZ78 codes into a sequence."""
        dictionary: dict[int, List[A]] = {0: []}
        result: List[A] = []
        index = 1
        for prefix_idx, symbol in codes:
            if prefix_idx not in dictionary:
                raise ValueError(f"Invalid prefix index {prefix_idx} in compressed data.")
            prefix_list = dictionary[prefix_idx]
            entry = prefix_list + [symbol]
            result.extend(entry)
            dictionary[index] = entry
            index += 1
        return result

    @classmethod
    def from_list(cls, data: List[A]) -> 'LZ78String[A]':
        """Create a compressed LZ78String from a raw sequence."""
        if not isinstance(data, list):
            raise TypeError("Input must be a list of items.")
        codes = cls._compress(data)
        return cls(codes)

    def to_list(self) -> List[A]:
        """Decompress and return the original sequence."""
        return self._decompress(self.content)

    @classmethod
    def from_codes(cls, codes: List[Tuple[int, A]]) -> 'LZ78String[A]':
        """Create LZ78String directly from compressed codes."""
        if not isinstance(codes, list):
            raise TypeError("Codes must be a list of (int, item) tuples.")
        for item in codes:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int)):
                raise ValueError("Each code must be a tuple (int, item).")
        return cls(codes)

    def to_codes(self) -> List[Tuple[int, A]]:
        """Return a copy of the compressed codes."""
        return list(self.content)

    def to_slp(self) -> 'SLP[A]':
        """Convert LZ78-compressed content into an SLP.

        constants: distinct symbols in order of first occurrence as suffixes.
        instructions: for each phrase (prefix_idx, symbol), record (prefix_idx', const_idx).
        Assumes an SLP class exists with constructor SLP(constants: List[A], instructions: List[Tuple[int, int]]).
        """
        compressed = self.content
        constants: List[A] = []
        constant_to_index: Dict[A, int] = {}
        instructions: List[Tuple[int, int]] = []
        phrase_index_map: Dict[int, int] = {}

        if not compressed:
            return SLP(constants=constants, instructions=instructions)

        # Process all phrases except the last to identify constants and map prefix-0 phrases
        for i, (prefix_idx, symbol) in enumerate(compressed[:-1]):
            if symbol not in constant_to_index:
                constant_to_index[symbol] = len(constants)
                constants.append(symbol)
            if prefix_idx == 0:
                phrase_index_map[i + 1] = constant_to_index[symbol]

        # Handle last symbol for constants
        last_prefix_idx, last_symbol = compressed[-1]
        if last_symbol not in constant_to_index:
            constant_to_index[last_symbol] = len(constants)
            constants.append(last_symbol)

        # Build instructions for phrases with non-zero prefixes
        for i, (prefix_idx, symbol) in enumerate(compressed[:-1]):
            if prefix_idx == 0:
                continue
            # phrase indices are 1-based
            phrase_index_map[i + 1] = len(constants) + len(instructions)
            instructions.append((phrase_index_map[prefix_idx], constant_to_index[symbol]))

        # Handle trivial case where only one phrase exists
        if len(phrase_index_map) < 2:
            constants = [compressed[0][1]]
            return SLP(constants=constants, instructions=instructions)

        # Add base instruction linking first two phrases
        instructions.append((phrase_index_map[1], phrase_index_map[2]))

        # Append remaining instructions for the subsequent phrases
        for i in range(2, len(compressed) - 1):
            instructions.append((len(constants) + len(instructions) - 1, phrase_index_map[i + 1]))

        # Add instructions for the last phrase if applicable
        if last_prefix_idx != 0:
            instructions.append((len(constants) + len(instructions) - 1, phrase_index_map[last_prefix_idx]))
        if last_symbol is not None:
            instructions.append((len(constants) + len(instructions) - 1, constant_to_index[last_symbol]))

        return SLP(constants=constants, instructions=instructions)

    def __repr__(self) -> str:
        return f"LZ78String(content={self.content})"

# Example usage:
if __name__ == '__main__':
    # Compress a list of integers
    raw_ints = [1, 2, 1, 2, 1, 3]
    comp_int = LZ78String.from_list(raw_ints)
    print("Compressed ints:", comp_int)
    print("Decompressed ints:", comp_int.to_list())

    # Compress a list of characters
    raw_chars = list("abracadabra")
    comp_chars = LZ78String.from_list(raw_chars)
    print("Compressed chars:", comp_chars)
    print("Decompressed chars:", ''.join(comp_chars.to_list()))

    # From codes
    codes = comp_chars.to_codes()
    comp_again = LZ78String.from_codes(codes)
    print("Round-trip OK?", ''.join(comp_again.to_list()) == "abracadabra")
