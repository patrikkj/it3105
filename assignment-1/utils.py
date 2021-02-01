from enum import Enum
from typing import NamedTuple


class SAP(NamedTuple):
    state: int
    action: str

    def __str__(self):
        return f"{self.state} \n {self.action}"


class Direction(Enum):
    UP = ([-1, 0], (1 << 4) + (1 << 5) + (1 << 1))
    RIGHT = ([0, 1], (1 << 8) + (1 << 9) + (1 << 1))
    DOWN = ([1, 0], (1 << 12) + (1 << 13) + (1 << 1))
    LEFT = ([0, -1], (1 << 16) + (1 << 17) + (1 << 1))

    UP_LEFT = ([-1, -1], (1 << 2) + (1 << 3) + (1 << 1))
    UP_RIGHT = ([-1, 1], (1 << 6) + (1 << 7) + (1 << 1))
    DOWN_RIGHT = ([1, 1], (1 << 10) + (1 << 11) + (1 << 1))
    DOWN_LEFT = ([1, -1], (1 << 14) + (1 << 15) + (1 << 1))

    @property
    def vector(self):
        return self.value[0]

    @property
    def kernel(self):
        return self.value[1]
