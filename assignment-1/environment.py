import numpy as np
from scipy import ndimage
from utils import Direction, kernel, edge_mask
from abc import abstractmethod


D = Direction
TRIANGE_DIRECTIONS = [D.UP_LEFT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_RIGHT]
DIAMOND_DIRECTIONS = [D.UP_RIGHT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_LEFT]


class Environment:
    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def is_terminal(self):
        ...

    @abstractmethod
    def get_observation(self):
        # NOTE: Observations must be hashable!
        ...

    def decode_state(self, state):
        """
        Overload this method if 'self.get_observation()' 
        returns a compressed state representation.
        """
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class PegEnvironment(Environment):
    # Board types
    TRIANGLE = "triangle"
    DIAMOND = "diamond"

    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS_PER_PIN = -15

    def __init__(self, board_size=5, holes=1):
        self.board_size = board_size
        self.holes = holes

        # Used for pruning invalid actions/positions
        self._mask = self._board.astype(bool)
        self._total_cells = self._mask.sum()

        # The broadcasted matrix is |directions| boards filled with the constant (bitpattern) for each direction
        self.dir_matrix = np.array([d.kernel for d in self.directions]).reshape((1, 1, -1))

        # Initialize environment state (all initialization is done in self.reset())
        self.reset()

    def _generate_moves(self):
        conv = ndimage.convolve(self.board + ~self._mask, kernel, mode="constant", cval=1)
        conv = np.bitwise_xor(conv, edge_mask)      # Flips bits corresponding to the kernels' circumference
        conv_3d = conv[..., np.newaxis]             # Broadcasted to 3d to vectorize calculations for all directions
        board_3d = self.board[..., np.newaxis]      # Element-wise multiplied to filter pegs
        return zip(*((np.bitwise_and(conv_3d, self.dir_matrix) == self.dir_matrix) * board_3d).nonzero())

    def _set_cell(self, vector, value=0):
        self.board[vector[0], vector[1]] = value

    def _assign_holes(self):
        for hole in self.holes:
            self._set_cell(hole, value=0)

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
        self._set_cell(vector + 2 * direction, value=1)

        # Cache previous move (can be handy for visualization)
        self._peg_start_position = vector
        self._peg_end_position = vector + 2 * direction
        self._peg_move_direction = direction

        # Determine if terminal state
        self._actions = tuple(self._generate_moves())
        self._is_terminal = len(self._actions) == 0

        # Determine reward
        self._pegs_left = self.board.sum()
        reward = self.calculate_reward()

        self._step += 1
        return self.get_observation(), reward, self._is_terminal

    def calculate_reward(self):
        if self._is_terminal and self._pegs_left == 1:
            reward = PegEnvironment.REWARD_WIN
        elif self._is_terminal:
            reward = PegEnvironment.REWARD_LOSS_PER_PIN * self._pegs_left**1.5
        else:
            reward = PegEnvironment.REWARD_ACTION
        return reward

    def get_pegs_left(self):
        return self._pegs_left

    def get_observation(self):
        """Returns the agents' perceivable state of the environment."""
        return self.board[self._mask].astype(bool).tobytes()

    def get_legal_actions(self):
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = self._board.copy()
        self._assign_holes()
        self._pegs_left = self.board.sum()
        self._actions = tuple(self._generate_moves())
        self._is_terminal = False
        self._step = 0

        if len(self._actions) == 0:
            raise EnvironmentError("There are no legal actions for the initial configuration!")

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        return self._is_terminal

    def decode_state(self, state):
        return np.frombuffer(state, dtype=np.uint8)

    @staticmethod
    def from_type(type_, *args, **kwargs):
        if type_== PegEnvironment.TRIANGLE:
            return PegEnvironmentTriangle(*args, **kwargs)
        elif type_== PegEnvironment.DIAMOND:
            return PegEnvironmentDiamond(*args, **kwargs)


class PegEnvironmentTriangle(PegEnvironment):
    def __init__(self, board_size=5, holes=None):
        self._board = np.tri(board_size, dtype=np.int64)
        self.directions = TRIANGE_DIRECTIONS
        super().__init__(board_size=board_size, holes=holes)


class PegEnvironmentDiamond(PegEnvironment):
    def __init__(self, board_size=5, holes=None):
        self._board = np.ones((board_size, board_size), dtype=np.int64)
        self.directions = DIAMOND_DIRECTIONS
        super().__init__(board_size=board_size, holes=holes)
