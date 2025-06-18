from dataclasses import dataclass
from typing import List, Tuple, Dict, TypeVar, Generic

A = TypeVar('A')

@dataclass
class LZ78(Generic[A]):
    """LZ78 Compressor and Decompressor."""

    def compress(self, data: List[A]) -> List[Tuple[int, A]]:
        dictionary: Dict[Tuple[A, ...], int] = {}
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
                    result.append((dictionary[tuple(current[:-1])], symbol))
                dictionary[prefix] = index
                index += 1
                current.clear()

        return result

    def decompress(self, compressed: List[Tuple[int, A]]) -> List[A]:
        dictionary: Dict[int, List[A]] = {0: []}
        result: List[A] = []
        index = 1

        for prefix_idx, symbol in compressed:
            prefix = dictionary[prefix_idx]
            entry = prefix + [symbol]
            result.extend(entry)
            dictionary[index] = entry
            index += 1

        return result
