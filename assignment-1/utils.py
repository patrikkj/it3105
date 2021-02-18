import json
from enum import Enum
from typing import NamedTuple
import numpy as np


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


# Construct 2D-kernel used for move generation
_ = 0
kernel = np.array([
    [ 3,  _,  5,  _,  7],
    [ _,  2,  4,  6,  _],
    [17, 16,  1,  8,  9],
    [ _, 14, 12, 10,  _],
    [15,  _, 13,  _,  11],
])
kernel = np.exp2(kernel)[::-1, ::-1] * (kernel != 0)


# Create edge mask - sum of entries on the kernels' circumference
edge_mask = kernel.copy()
edge_mask[1:-1, 1:-1] = 0                   # Clears non-edge values
edge_mask = edge_mask.sum().astype(int)     # Encode edge values in a bit pattern represented by an integer


def write_config(config, filepath):
    with open(filepath, "w+") as f:
        json.dump(config, f, indent=4)

def read_config(filepath):
    with open(filepath, "r") as f:
        return json.load(open(filepath))
