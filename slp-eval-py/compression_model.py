from dataclasses import dataclass


@dataclass
class SLP[A]:
    """Straight line program over alphabet A"""

    constants: list[A]
    instructions: list[tuple[int, int]]

    def evaluate(self) -> list[A]:
        s: list[list[A]] = [[c] for c in self.constants]
        for i, j in self.instructions:
            r = s[i] + s[j]
            s.append(r)
        return s[-1]
