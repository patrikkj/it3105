
import numpy as np
from scipy import ndimage, signal


_KERNEL = np.array([
    [3,  0,  5,  0,  7],
    [0,  2,  4,  6,  0],
    [17, 16, 1,  8,  9],
    [0,  14, 12, 10, 0],
    [15, 0,  13, 0,  11],
])
kernel = np.exp2(_KERNEL)[::-1, ::-1] * (_KERNEL != 0)
edge_mask = np.exp2(np.array([3, 5, 7, 9, 11, 13, 15, 17])).sum().astype(int)

from enum import Enum
class Direction(Enum):
    UP = ([-1, 0], (1<<4) + (1<<5) + (1<<1))
    RIGHT = ([0, 1], (1<<8) + (1<<9) + (1<<1))
    DOWN = ([1, 0], (1<<12) + (1<<13) + (1<<1))
    LEFT = ([0, -1], (1<<16) + (1<<17) + (1<<1))

    UP_LEFT = ([-1, -1], (1<<2) + (1<<3) + (1<<1))
    UP_RIGHT = ([-1, 1], (1<<6) + (1<<7) + (1<<1))
    DOWN_RIGHT = ([1, 1], (1<<10) + (1<<11) + (1<<1))
    DOWN_LEFT = ([1, -1], (1<<14) + (1<<15) + (1<<1))
    
    @property
    def vector(self):
        return self.value[0]
        
    @property
    def kernel(self):
        return self.value[1]

D = Direction
TRIANGE_DIRECTIONS = [D.UP_LEFT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_RIGHT]
DIAMOND_DIRECTIONS = [D.UP_RIGHT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_LEFT]


class PegSolitaire:
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS_PER_PIN = -15

    def __init__(self, board_type='diamond', board_size=5, holes=1):
        self.board_type = board_type
        self.board_size = board_size
        self.holes = holes

        if board_type == 'diamond':
            self._board = np.ones((board_size, board_size), dtype=np.int64)
            self.directions = DIAMOND_DIRECTIONS
        else:
            self._board = np.tri(board_size, dtype=np.int64)
            self._mask = np.tri(board_size, k=-1, dtype=np.int64).T
            self.directions = TRIANGE_DIRECTIONS

        # Set holes in board
        for x, y in holes:
            self._board[x, y] = 0
        
        # Initialize environment state
        self.reset()

    def _generate_moves(self):
        conv = ndimage.convolve(self.board + self._mask, kernel, mode='constant', cval=1)
        conv = np.bitwise_xor(conv * self._board, edge_mask)
        for i, direction in enumerate(self.directions):
            indices = (np.bitwise_and(conv, direction.kernel) == direction.kernel ).nonzero()
            yield from ((x, y, i) for x, y in zip(*indices))

    def _set_cell(self, vector, value=0):
        self.board[vector[0], vector[1]] = value

    def step(self, action):
        """
        Apply action to this environment.
        Returns: 
            observation (object): an environment-specific object representing your observation of the environment.
            reward (float): amount of reward achieved by the previous action. 
        """
        # Perform action
        x, y, direction_index = action
        vector = np.array([x, y])
        direction = np.array(self.directions[direction_index].vector)

        # Set pegs
        self._set_cell(vector, value=0)
        self._set_cell(vector + direction, value=0)
        self._set_cell(vector + 2*direction, value=1)

        # Determine if terminal state
        self._actions = list(self._generate_moves())
        self._is_terminal = len(self._actions) == 0

        # Determine reward
        self._pegs_left = self.board.sum()
        if self._is_terminal and self._pegs_left == 1:
            reward = PegSolitaire.REWARD_WIN
        elif self._is_terminal:
            reward = PegSolitaire.REWARD_LOSS_PER_PIN * self._pegs_left
        else:
            reward = PegSolitaire.REWARD_ACTION
        
        self._step += 1
        return self.get_observation(), reward, self._is_terminal

    def get_current_step(self):
        """Returns the number of steps for the active episode."""
        return self._step

    def get_pegs_left(self):
        return self._pegs_left
        
    def get_observation(self):
        """Returns the specifications for the observation space."""
        #return "\n".join("".join(map(str, row)) for row in self.board.astype(int))
        return self.board.astype(bool).tobytes()

    def get_observation_spec(self):
        """Returns the specifications for the observation space."""
        return (2**self.board.size, )

    def get_action_spec(self):
        """Returns the specifications for the action space."""
        return (self.board.size, )

    def get_legal_actions(self):
        # Upon resetting the enviroment, no actions will be cached
        if not self._actions:
            self._actions = list(self._generate_moves())
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = self._board.copy()
        self._pegs_left = self.board.sum()
        self._actions = []
        self._is_terminal = False
        self._step = 0

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        return self._is_terminal
    
    def render(self):
        print(self.board)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        return False
"""
board = PegSolitaire(board_type="triangle")
total_moves = 0

import time
t0 = time.perf_counter()
for _ in range(1_000):
    actions = board.get_legal_actions()
    #print(actions)
t1 = time.perf_counter()
"""
#print(actions)
#print("done")
#print(t1-t0, "s")
#print(bin(UP))
#print(np.bitwise_and(conv, UP_LEFT) == UP_LEFT)


#print(kernel)
#print(bin(conv[2, 2]))

#conv_list2 = conv.tolist()
#for row in conv_list2:
#    print([f"{x:018b}" for x in row])