from copy import deepcopy
from enum import Flag, auto
from functools import cached_property, lru_cache
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, RegularPolygon

from .state_manager import EnvironmentSpec, StateManager


class HexFlag(Flag):
    """Flag stating whether a cell is connected to a given edge."""
    ISOLATED = 0

    # Wall connections
    P1_LOWER = auto()
    P1_UPPER = auto()
    P2_LOWER = auto()
    P2_UPPER = auto()

    # Victory flags (inferred)
    P1_WIN = P1_LOWER | P1_UPPER
    P2_WIN = P2_LOWER | P2_UPPER
    WIN = P1_WIN | P2_WIN

    @staticmethod
    def is_win(flag):
        return HexFlag.P1_WIN in flag or HexFlag.P2_WIN in flag


class HexSet:
    """Hex-specific implementation of disjoint sets."""
    def __init__(self, player, flags=None):
        self.parent = self
        self.rank = 0
        self.player = player
        self.flags = flags or HexFlag.ISOLATED


class HexGrid:
    DIRECTIONS = np.array([(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, -1)])

    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = np.full((board_size, board_size), fill_value=None)

    def _get_neighbours(self, x, y, player):
        # Indices in (x, y)'s neighbourhood
        indices = HexGrid.DIRECTIONS + (x, y)

        # Remove out-of-bounds indices
        indices = indices[np.all((indices >= 0) & (indices < self.board_size), axis=1)]

        # Return matching cells within neighborhood
        for _x, _y in indices:
            if (cell := self.grid[_x, _y]) and cell.player == player:
                yield cell
        
    def move(self, x, y, player):
        # Determine new cell flag
        if player == 1:
            flags = (HexFlag.P1_LOWER if x == 0 else 0) \
                  | (HexFlag.P1_UPPER if x == self.board_size - 1 else 0)
        else:
            flags = (HexFlag.P2_LOWER if y == 0 else 0) \
                  | (HexFlag.P2_UPPER if y == self.board_size - 1 else 0)
        
        # Create new disjoint set node
        cell = HexSet(player, flags=flags)

        # Find neighborhood of the newly generated cell
        neighbours = self._get_neighbours(x, y, player)

        # Merge neighbouring sets for the new cell
        self.union_many(neighbours)
        self.grid[x, y] = cell
        return self.find(cell).flags   # Return flag of representative
    

    # ----------- Disjoint set operations -----------
    def find(self, x):
        """
        Returns the representative for the disjoint set 'x'.
        Implements  the 'path compression' heuristic.
        """
        if x.parent != x:
            x.parent = self.find(x.parent)
        return x.parent

    def _link(self, x, y):
        """Links the disjoint sets, assumed to be disjoint prior to linking."""
        if x.rank < y.rank:
            x.parent = y
        else:
            y.parent = x
            if x.rank == y:
                x.rank += 1

    def union(self, x, y):
        """
        Merges 'x' and  'y' to the union of their disjoint sets.
        Ensures that 'x' and 'y' have the same representative.
        Implements  the 'union by rank' heuristic.
        """
        a, b = self.find(x), self.find(y)
        if a != b:
            self._link(a, b)

            # Combine flags when merging sets
            a.parent.flags = a.flags | b.flags

    def union_many(self, nodes):
        """Convenience method for merging multiple disjoint sets."""
        nodes = list(nodes)
        for x, y in zip(nodes, nodes[1:]):
            self.union(x, y)


class HexEnvironment(StateManager):
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS = -500

    def __init__(self, board_size=6):
        self.board_size = board_size
        self.spec = EnvironmentSpec(
            observations=self.board_size**2 + 1,    # Board + PID
            actions=self.board_size**2
        )
        # Initialize environment state (all initialization is done in self.reset())
        self.reset()

    def move(self, action, player):
        # Perform action
        x, y = action
        self.board[x, y] = player
        self._actions.discard(action)

        # Apply move to internal representation
        flags = self.hexgrid.move(x, y, player)
        
        # Determine if terminal state
        self._is_terminal = HexFlag.is_win(flags)

        # Determine reward
        reward = self.calculate_reward()

        self._step += 1
        return self.get_observation(), reward, self._is_terminal

    def calculate_reward(self):
        """Determines the reward for the most recent step."""
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
        self.hexgrid = HexGrid(self.board_size)
        self._actions = set(map(tuple, np.argwhere(self.board == 0)))
        self._is_terminal = False
        self._step = 0

    def copy(self):
        obj = object.__new__(self.__class__)
        obj.board_size = self.board_size
        obj.board = self.board.copy()
        obj.hexgrid = deepcopy(self.hexgrid)
        obj._actions = self._actions.copy()
        obj._is_terminal = self._is_terminal
        obj._step = self._step
        return obj

    def decode_state(self, state):
        # bytestring -> np.array([0, 1, 1, 0, 0, 1])
        return np.frombuffer(state, dtype=np.uint8)
    
    def __str__(self):
        return HexRenderer.board2string(self.board)


class HexRenderer:
    COLUMN_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    THETA = np.radians(30)


    @staticmethod
    @lru_cache(maxsize=1)
    def _transform_matrix():
        theta = HexRenderer.THETA
        cos, sin = np.cos(theta), np.sin(theta)

        # Construct affine transformations
        skew_matrix = np.array([[1, 0], [0, np.sqrt(3)/2]])     # Adjust 'y' coordinates for compact grid layout
        shear_matrix = np.array([[1, 0], [theta, 1]])
        rotation_matrix = np.array(((cos, -sin), (sin, cos)))
        mat = skew_matrix @ shear_matrix @ rotation_matrix
        return mat

    @staticmethod
    def plot(board):
        # Create figure
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Create ramdonly filled board to see colors
        board = np.random.randint(0, 3, board.shape)
        n = board.shape[0]
        low, high = 0, n - 1
        label_offset = 1
        tri_offset = 1.5
        colors = {
            0.: "#bbb",
            1.: "#F8A857",
            2.: "#57A8F8"
        }

        # Do some matte 3, TODO: cleanup
        coords = np.indices(board.shape).astype(float)
        coords = coords.reshape(2, -1).T
        new_coords = coords @ HexRenderer._transform_matrix()

        # Add hexagons
        for coord, player in zip(new_coords, board.flat):
            hexagon = RegularPolygon(coord, numVertices=6, radius=np.sqrt(1/3), 
                orientation=-HexRenderer.THETA, facecolor=colors[player], edgecolor='#606060', zorder=1)
            ax.add_patch(hexagon)
        
        # Add labels for cell references (A1, B3, ...)
        label_low, label_high = low - label_offset, high + label_offset
        top_coords = np.vstack([np.full(n, fill_value=label_low), np.arange(n)]).T
        bottom_coords = np.vstack([np.full(n, fill_value=label_high), np.arange(n)]).T
        alpha_coords = np.vstack([top_coords, bottom_coords]) @ HexRenderer._transform_matrix()
        alpha_labels = np.tile(np.array(list(ascii_uppercase[:n])), 2)

        left_coords = np.vstack([np.arange(n), np.full(n, fill_value=label_low)]).T
        right_coords = np.vstack([np.arange(n), np.full(n, fill_value=label_high)]).T
        numeric_coords = np.vstack([left_coords, right_coords]) @ HexRenderer._transform_matrix()
        numeric_labels = np.tile(np.array(list(map(str, range(1, n+1)))), 2)

        for label, coords in zip(alpha_labels, alpha_coords):
            ax.text(*coords, label)

        for label, coords in zip(numeric_labels, numeric_coords):
            ax.text(*coords, label)

        # Add triangles in the background
        tri_low, tri_high = low - tri_offset, high + tri_offset
        tri_coords = np.array([
            (tri_high, tri_high), (tri_low, tri_low), 
            (tri_low, tri_high), (tri_high, tri_low), 
            ((tri_low + tri_high)/2, (tri_low + tri_high)/2)
        ])
        tri_coords = tri_coords @ HexRenderer._transform_matrix()
        t, b, l, r, c = tri_coords
        red_triangles = ((t, l, c), (r, b, c))
        green_triangles = ((l, b, c), (r, t, c))
        [ax.add_patch(Polygon(tri, facecolor="#FAC48E", edgecolor='#606060', zorder=0)) for tri in red_triangles]
        [ax.add_patch(Polygon(tri, facecolor="#85BFF9", edgecolor='#606060', zorder=0)) for tri in green_triangles]
        
        # Display figure
        plt.autoscale(enable=True)
        plt.axis('off')
        plt.show()

    @staticmethod
    def board2string(board):
        """
        From https://stackoverflow.com/questions/65396231/print-hex-game-board-contents-properly
        We ❤️ StackOverflow, temporary solution.
        """
        out = ["\n"]
        rows = len(board)
        cols = len(board[0])
        indent = 0
        headings = " "*5+(" "*3).join(HexRenderer.COLUMN_NAMES[:cols])
        out.append(headings)
        out.append(" "*5+(" "*3).join("-"*cols))    # tops
        out.append(" "*4+"/ \\"+"_/ \\"*(cols-1))   # roof
        BLUE = '\x1b[0;0;43m \x1b[0m'
        RED = '\x1b[0;0;41m \x1b[0m'
        color_mapping = lambda i : (' ', BLUE, RED)[i]
        for r in range(rows):
            row_mid = " "*indent
            row_mid += " {} | ".format(r+1)
            row_mid += " | ".join(map(color_mapping,board[r]))
            row_mid += " | {} ".format(r+1)
            out.append(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r<rows-1:
                row_bottom += " \\"
            out.append(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        out.append(headings)
        return "\n".join(out)
    
    # def __str__(self):
    #    return print_board(self.board)
    #    # Lets make some fancy shear operations
    #    lines = " " + np.array2string(self.board, 
    #                            separator=' ', 
    #                            prefix='', 
    #                            suffix='').replace('[', '').replace(']', '')
    #    return "\n".join(f"{' '*i}\\{line}\\" for i, line in enumerate(lines.split("\n")))


env = HexEnvironment(6)
env.move((5, 2), player=2)
env.move((4, 2), player=1)
print(env)
