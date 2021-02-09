from enum import Enum
from typing import NamedTuple


class SAP(NamedTuple):
    state: int
    action: str

    def __str__(self):
        return f"{self.state} \n {self.action}"


class Direction(Enum):
    """
    Defines the direction vectors and the bit-strings that us used in checking that the direction is legal.
    
    Returns a tuple of the vector of the direction and the bitstring that will be used to check if a board (bitstring) has a legal move in that direction.
    If so, the AND operation with a board bitstring will return the direction vector.

    Example:
    
        1101110111
    AND 0000110010   <-- The bit string is used in this operation.
    =   0000110010  

    """
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
        """Returns the vector of the direction (indexed 0)."""
        return self.value[0]

    @property
    def kernel(self):
        """Returns the kernel bit-string of the direction (indexed 1)."""
        return self.value[1]
