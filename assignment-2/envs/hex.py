from copy import deepcopy

import numpy as np

from .hex_grid import DisjointHexGrid, HexFlag
from .hex_renderer import board2string
from .state_manager import StateManager


class HexEnvironment(StateManager):
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS = -500

    def __init__(self, board_size=6):
        self.board_size = board_size

        # Initialize environment state (all initialization is done in self.reset())
        self.reset()

    def move(self, action, player):
        # Perform action
        x, y = action
        self.board[x, y] = player
        self._actions.discard(action)

        # Apply move to internal representation
        flags = self.disjoint_hexgrid.move(x, y, player)
        
        # Determine if terminal state
        self._is_terminal = HexFlag.is_win(flags)

        # Determine reward
        reward = self.calculate_reward()

        self._step += 1
        return self.get_observation(), reward, self._is_terminal

    def calculate_reward(self):
        """Determinse the reward for the most recent step."""
        if self._is_terminal and self._pegs_left == 1:
            reward = HexEnvironment.REWARD_WIN
        elif self._is_terminal:
            reward = HexEnvironment.REWARD_LOSS
        else:
            reward = HexEnvironment.REWARD_ACTION
        return reward
    
    def is_finished(self):
        return self._is_terminal

    def get_observation(self):
        """Returns the agents' perceivable state of the environment."""
        return self.board.astype(bool).tobytes()

    def get_legal_actions(self):
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.disjoint_hexgrid = DisjointHexGrid(self.board_size)
        self._actions = set(map(tuple, np.argwhere(self.board == 0)))
        self._is_terminal = False
        self._step = 0

    def copy(self):
        obj = object.__new__(self.__class__)
        obj.board_size = self.board_size
        obj.board = self.board.copy()
        obj.disjoint_hexgrid = deepcopy(self.disjoint_hexgrid)
        obj._actions = self._actions.copy()
        obj._is_terminal = self._is_terminal
        obj._step = self._step
        return obj

    def decode_state(self, state):
        # bytestring -> np.array([0, 1, 1, 0, 0, 1])
        return np.frombuffer(state, dtype=np.uint8)
    
    def __str__(self):
        return board2string(self.board)


env = HexEnvironment(6)
env.move((5, 2), player=2)
env.move((4, 2), player=1)
print(env)
