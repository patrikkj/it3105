import numpy as np
from scipy import ndimage
from utils import Direction


D = Direction
TRIANGE_DIRECTIONS = [D.UP_LEFT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_RIGHT]
DIAMOND_DIRECTIONS = [D.UP_RIGHT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_LEFT]

# Construct 2D-kernel used for move generation
_ = 0
_kernel = np.array([
    [ 3,  _,  5,  _,  7],
    [ _,  2,  4,  6,  _],
    [17, 16,  1,  8,  9],
    [ _, 14, 12, 10,  _],
    [15,  _, 13,  _,  11],
])
kernel = np.exp2(_kernel)[::-1, ::-1] * (_kernel != 0)

# Create edge mask - sum of entries on the kernels' circumference
edge_mask = kernel.copy()
edge_mask[1:-1, 1:-1] = 0                   # Clears non-edge values
edge_mask = edge_mask.sum().astype(int)     # Encode edge values in a bit pattern represented by an integer

class PegSolitaire:
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS_PER_PIN = -15

    def __init__(self, board_type="diamond", board_size=5, holes=1):
        self.board_type = board_type
        self.board_size = board_size
        self.holes = holes

        if board_type == "diamond":
            self._board = np.ones((board_size, board_size), dtype=np.int64)
            self._mask = np.zeros((board_size, board_size), dtype=np.int64)
            self.directions = DIAMOND_DIRECTIONS
        else:
            self._board = np.tri(board_size, dtype=np.int64)
            self._mask = np.tri(board_size, k=-1, dtype=np.int64).T
            self.directions = TRIANGE_DIRECTIONS

        # Create direction matrix, the first two dimensions are broadcasted to the boards' dimensions when applied
        # The broadcasted matrix is |directions| boards filled with the constant (bitpattern) corresponding to each direction
        self.dm = np.array([d.kernel for d in self.directions]).reshape((1, 1, -1))

        # Initialize environment state (all initialization is done in self.reset())
        self.reset()

    def _generate_moves(self):
        conv = ndimage.convolve(self.board + self._mask, kernel, mode="constant", cval=1)
        conv = np.bitwise_xor(conv, edge_mask)      # Flips bits corresponding to the kernels' circumference
        conv_3d = conv[..., np.newaxis]             # Broadcasted to 3d to vectorize calculations for all directions
        board_3d = self.board[..., np.newaxis]      # Element-wise multiplied to filter pegs
        return zip(*((np.bitwise_and(conv_3d, self.dm) == self.dm) * board_3d).nonzero())

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

        # Cache previous move
        self._peg_start_position = vector
        self._peg_end_position = vector + 2 * direction
        self._peg_move_direction = direction

        # Determine if terminal state
        n_before = len(self._actions)
        self._actions = tuple(self._generate_moves())
        n_after = len(self._actions)
        self._is_terminal = len(self._actions) == 0

        # Determine reward
        self._pegs_left = self.board.sum()
        if self._is_terminal and self._pegs_left == 1:
            reward = PegSolitaire.REWARD_WIN
        elif self._is_terminal:
            reward = PegSolitaire.REWARD_LOSS_PER_PIN * self._pegs_left**1.5
        else:
            reward = PegSolitaire.REWARD_ACTION * max(n_before - n_after, 0)

        self._step += 1
        return self.get_observation(), reward, self._is_terminal

    def get_pegs_left(self):
        return self._pegs_left

    def get_observation(self):
        """Returns the agents' perceivable state of the environment."""
        return tuple(self.board[self._board == 1])

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

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        return self._is_terminal

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        return False


def print_bin_matrix(matrix):
    for row in matrix:
        if row.ndim != 1:
            print_bin_matrix(row)
        else:
            print([f"{x:018b}" for x in row])
