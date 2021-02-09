import numpy as np
from scipy import ndimage
from utils import Direction


D = Direction #Fetched from utils.py
TRIANGE_DIRECTIONS = [D.UP_LEFT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_RIGHT] #List of legal move directions for triangle board. Hence also directions where neighbours is found.
DIAMOND_DIRECTIONS = [D.UP_RIGHT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_LEFT] #List of legal move directions for diamond board. Hence also directions where neighbours is found.


"""
Definition of (local) _kernel. 
Used for setting a unique bit in a bitstring for each neighbour, where the boolean value is determined by whether the neighbour has a peg or not.
"""
_kernel = np.array([
    
    [ 3,  0,  5,  0,  7],
    [ 0,  2,  4,  6,  0],
    [17, 16,  1,  8,  9],
    [ 0, 14, 12, 10,  0],
    [15,  0, 13,  0,  11],
])
kernel = np.exp2(_kernel)[::-1, ::-1] * (_kernel != 0) #Reverses the matrix in both directions. Raises 2 to the power of each element, multiplies with a boolean matrix to ensure zeros in cells that are not neighbours.
edge_mask = np.exp2(np.array([3, 5, 7, 9, 11, 13, 15, 17])).sum().astype(int) #Used for flipping the bit at each edge neighbour. Returns an integer representing the edges when represented as bit-string. Need .astype(int) since np.sum() returnes a float.






class PegSolitaire:
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS_PER_PIN = -15
    """
    Declaration of static variables representing the rewards.
    REWARD_WIN: When reaching win state
    REWARD_ACTION: General reward when executing a non-terminating move.
    REWARD_LOSS_PER_PIN: (Negative) reward per remaining peg in the final board at losing game.
    """

    def __init__(self, board_type="diamond", board_size=5, holes= None):
        """
        The constructor. Initializes the game board, its legal moves and its operators.

        args:
            board_type -- (string) either diamond or triangle
            board_size -- (int) the size of the sides in triangel or diamond
            holes      -- (list) list of hole coordinates in the initial board
        """
        self.board_type = board_type
        self.board_size = board_size
        self.holes = holes if holes is not None else [(2,2)]

        if board_type == "diamond":
            self._board = np.ones((board_size, board_size), dtype=np.int64)                 #Initializing _board of 1's
            self._mask = np.zeros((board_size, board_size), dtype=np.int64)                 #Initializing _mask of 0's
            self.directions = DIAMOND_DIRECTIONS
        else:
            self._board = np.tri(board_size, dtype=np.int64)                                #Generates a matrix with 1'n down left, including diagonal.
            self._mask = np.tri(board_size, k=-1, dtype=np.int64).T                         #Generates a matrix with 1's in the upper right corner (all illegal cells) that are not part of the board
            self.directions = TRIANGE_DIRECTIONS
        
        self.dm = np.array([d.kernel for d in self.directions]).reshape(1, 1, -1)           #Direction matrix of (1x1x6) (x,y,z). Generates an array consisting of the bit-strings representing the legal actions for the current board shape.
        if self.board_type == "triangle":
            self.dm = self.dm * np.tri(self.board_size, dtype=np.int64)[..., None]          #5x5x6 matrix. For each wall: Down left is zeros. Up right is 1's

        # Initialize environment state (all initialization is done in self.reset())
        self.reset()
        self.render()

    def _generate_moves(self):
        """
        Applies the convolution to the entire board.
        For triangular board the upper right is masked with ones to prevent generation of moves ending in this quadrant.
        The board is padded with ones with a radius of 2 to prevent pegs from jumping out of the board.
        This is also to ensure that the convolution is defined along the boards' edges.
        """
        conv = ndimage.convolve(self.board + self._mask, kernel, mode="constant", cval=1)   
        
        """
        Flips all bits in the resulting convolution which correspond to edges.
        This is done to enable bitwise logical AND-operations in the return-statement to check if the second neighbour is empty (i.e. has value 1 after being flipped)
        """
        conv = np.bitwise_xor(conv, edge_mask)
        
        """
        ANDs the convolved board with the bitpattern required to move in a specific direction.
        Shapes: 
        conv:             (n, n, 1)    
        dm:               (1, 1, |directions|)   
        conv & dm:        (n, n, |directions|)  (np fixes the reshaping of both conv and dm to this shape)

        Part explanations:
        (np.bitwise_and(conv[..., None], self.dm):      For each cell in the (n, n, |directions|), AND is used to check if move is legal. Legal if (AND-output == direction bit-string). This results in a cube consisting of 1's and 0's.
        nonzero():                                      Extracts the cells with possible moves (1's), and their legal directions: (x_1, x_2, ...), (y_1, y_2, ...), (d_1, d_2, ...).
        zip(*):                                         Reshapes the tuples to (x_1, y_1, d_1) etc.
        """
        return zip(*(np.bitwise_and(conv[..., None], self.dm) == self.dm).nonzero())
        
    def _set_cell(self, vector, value=0):
        """Setting cell value to 0 or 1 (=value). vector is formatted [x,y]"""
        self.board[vector[0], vector[1]] = value

    def _assign_holes(self):
        """"Setting hole cells to value 0. Holes are list of tuples (x,y)"""
        for hole in self.holes:
            self._set_cell(hole, value=0)

    def step(self, action):
        """
        Apply action to this environment.

        args: 
            action (tuple): Tuple of coordinates x and y, as well as indexed direction to move.


        Returns:
            observation (object): an environment-specific object representing your observation of the environment.
            reward (float): amount of reward achieved by the previous action.
        """
        # Collect cell and direction
        x, y, direction_index = action
        vector = np.array([x, y])
        direction = np.array(self.directions[direction_index].vector)

        # Execute step
        self._set_cell(vector, value=0)
        self._set_cell(vector + direction, value=0)
        self._set_cell(vector + 2 * direction, value=1)

        # Determine if terminal state
        self._actions = tuple(self._generate_moves())
        self._is_terminal = len(self._actions) == 0                         #setting the local variable if legal actions is zero

        # Determine reward
        self._pegs_left = self.board.sum()                                  #Numpy function: sum over elements in np.array
        if self._is_terminal and self._pegs_left == 1:                      #If win
            reward = PegSolitaire.REWARD_WIN
        elif self._is_terminal:                                             #If lose
            reward = PegSolitaire.REWARD_LOSS_PER_PIN * self._pegs_left
        else:                                                               #If not finished
            reward = PegSolitaire.REWARD_ACTION

        # Increse self.step by one.
        self._step += 1
        
        # returns the board (as bit-string!), the reward according to step executed, and if the step led to termination.
        return self.get_observation(), reward, self._is_terminal

    def get_current_step(self):
        """Returns the number of steps for the active episode."""
        return self._step

    def get_pegs_left(self):
        return self._pegs_left

    def get_observation(self):
        """
        Returns the specifications for the observation space.
        Returns only the board (not the full state), since this is the only relevant information for the agent.
        Since the critic assignes value to state in a dictionary, which does not accept mutable keys, the board needs to be represented in bytes, not as a (mutable) matrix.
        Note: uses astype(bool) so the 1's and 0's in the board are represented as booleans (1 bit), not integers (64 bits). Less memory usage!
        """
        return self.board.astype(bool).tobytes()

    def get_observation_spec(self):
        """Returns the specifications for the observation space."""
        return (2 ** self.board.size,)

    def get_action_spec(self):
        """Returns the specifications for the action space."""
        return (self.board.size,)

    def get_legal_actions(self):
        # Upon resetting the enviroment, no actions will be cached
        if not self._actions:
            self._actions = tuple(self._generate_moves())
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = self._board.copy()         #Copies the original board fors simplicity (do not need to generate a new one).
        self._assign_holes()                    #Actually setting hole cell to 0
        self._pegs_left = self.board.sum()      #Initializes amount of pegs on board (used to check terminality, win/lose, and hence reward calculation)
        self._actions = []                      #Reset actions per episode. Is overvritten for every step.
        self._is_terminal = False               #Resetting terminal-check.
        self._step = 0                          #Initializing step to 0

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        return self._is_terminal

    def render(self):
        print(self.board, "\n")

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
for _ in range(100_000):
    actions = board.get_legal_actions()
    #print(actions)
t1 = time.perf_counter()
"""
# print(actions)
# print("done")
# print(t1-t0, "s")
# print(bin(UP))
# print(np.bitwise_and(conv, UP_LEFT) == UP_LEFT)


# print(kernel)
# print(bin(conv[2, 2]))

# conv_list2 = conv.tolist()
# for row in conv_list2:
#    print([f"{x:018b}" for x in row])
"""
arr = np.array([direction.vector for direction in TRIANGE_DIRECTIONS])
board = np.tri(5, dtype=int)
print(arr)

print(board)


[[-1, -1],
 [-1,  0],
 [0, -1],
 [0,  1],
 [1,  0],
 [1,  1]]
 """