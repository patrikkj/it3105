from enum import Flag, auto
from functools import lru_cache
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, RegularPolygon

from .state_manager import EnvironmentSpec, StateManager
œ

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


class HexGrid:
    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = [None for _ in range(board_size**2)]
        self._neighbour_offsets = tuple(x * board_size + y for x, y in HexGrid.DIRECTIONS)
    
    def _create_cell(self, _id, player, flags=None):
        """Hex-specific implementation of disjoint sets."""
        return {
            "_id": _id,
            "parent": _id,
            "rank": 0,
            "player": player,
            "flags": flags
        }

    def _get_neighbours(self, _id, player):
        """Returns all within-bound neighbours using a flat board representation."""
        n, size = self.board_size, self.board_size**2

        north_id = _id - n
        if north_id >= 0 and (cell := self.grid[north_id]) and cell['player'] == player:
            yield north_id 
        
        west_id = _id - 1
        if _id % n != 0 and (cell := self.grid[west_id]) and cell['player'] == player:
            yield west_id
        
        east_id = _id + 1
        if east_id % n != 0 and (cell := self.grid[east_id]) and cell['player'] == player:
            yield east_id
        
        south_id = _id + n
        if south_id < size and (cell := self.grid[south_id]) and cell['player'] == player:
            yield south_id

        northeast_id = _id - n + 1
        if (northeast_id >= 0) and ((_id + 1) % n != 0) and (cell := self.grid[northeast_id]) and cell['player'] == player:
            yield northeast_id

        southwest_id = _id + n - 1
        if (southwest_id < size) and (_id % n != 0) and (cell := self.grid[southwest_id]) and cell['player'] == player:
            yield southwest_id

    def move(self, x, y, player):
        # Determine new cell flag
        if player == 1:
            flags = HexFlag.P1_LOWER if x == 0 else \
                    HexFlag.P1_UPPER if x == self.board_size - 1 else HexFlag.ISOLATED
        else:
            flags = HexFlag.P2_LOWER if y == 0 else \
                    HexFlag.P2_UPPER if y == self.board_size - 1 else HexFlag.ISOLATED
        
        # Create new disjoint set node
        _id = x * self.board_size + y
        self.grid[_id] = self._create_cell(_id=_id, player=player, flags=flags)

        # Find + merge neighborhood of the newly generated cell
        if neighbours := self._get_neighbours(_id, player):
            self.union_many(_id, neighbours)
            _id = self.find(_id)
        return self.grid[_id]['flags']                          # Return flag of representative

    def copy(self):
        obj = object.__new__(self.__class__)
        obj.board_size = self.board_size
        obj.grid = [{**d} if d else None for d in self.grid]
        obj._neighbour_offsets = self._neighbour_offsets
        return obj

    # ----------- Disjoint set operations -----------
    def find(self, _id):
        """
        Returns the representative for the disjoint set 'x'.
        Implements  the 'path compression' heuristic.
        Assumes the input is the _id  attribute (equal to index) of the relevant node.
        """
        x = self.grid[_id]
        if x['parent'] != _id:
            x['parent'] = self.find(x['parent'])
            self.grid[x['parent']]['flags'] = x['flags'] | self.grid[x['parent']]['flags']
        return x['parent']

    def _link(self, x, y):
        """Links the disjoint sets, assumed to be disjoint prior to linking."""
        if x['rank'] < y['rank']:
            x['parent'] = y['_id']
        else:
            y['parent'] = x['_id']
            if x['rank'] == y['rank']:
                x['rank'] += 1

    def union(self, x, y):
        """
        Merges 'x' and  'y' to the union of their disjoint sets.
        Ensures that 'x' and 'y' have the same representative.
        Implements  the 'union by rank' heuristic.
        Assumes the input is the _id  attribute (equal to index) of the relevant node.
        """
        a, b = self.find(x), self.find(y)
        if a != b:
            a, b = self.grid[a], self.grid[b]
            self._link(a, b)
            self.grid[a['parent']]['flags'] = a['flags'] | b['flags']      # Combine flags when merging sets

    # def union_many(self, nodes):
    #     """Convenience method for merging multiple disjoint sets."""
    #     nodes = list(nodes)
    #     for x, y in zip(nodes, nodes[1:]):
    #         self.union(x, y)

    def union_many(self, _id, neighbours):
        """Convenience method for merging multiple disjoint sets."""
        for _id_neigh in neighbours:
            self.union(_id, _id_neigh)


class HexEnvironment(StateManager):
    REWARD_WIN = 1
    REWARD_ACTION = 0
    REWARD_LOSS = -1
    STARTING_PLAYER = 1

    def __init__(self, board_size=6):
        super().__init__(spec=EnvironmentSpec(
            observations=board_size**2 + 1,    # Board + PID
            actions=board_size**2
        ))
        self.board_size = board_size
        self.reset()    # Initialize environment state (all initialization is done in self.reset())

    def _validate_move(self, action, player):
        assert player == self._current_player, f"Invalid move; it's not player {player}'s turn!"
        assert not self._is_terminal, f"Game is over; cannot make any more moves."
        assert action in self._actions, f"The selected action is not allowed."

    def move(self, action, player):
        self._validate_move(action, player)

        # Perform action
        x, y = action // self.board_size, action % self.board_size
        self.board[x, y] = player
        self._actions.discard(action)

        flags = self.hexgrid.move(x, y, player)     # Apply move to internal representation
        self._is_terminal = HexFlag.is_win(flags)   # Determine if terminal state
        reward = self.calculate_reward()            # Determine reward

        self._step += 1
        self._current_player = self._current_player % 2 + 1
        #HexRenderer.plot(self.board)
        return self.get_observation(), reward, self._is_terminal

    def calculate_reward(self):
        """Determines the reward for the most recent step."""
        if self._is_terminal and self._current_player == 2:
            reward = HexEnvironment.REWARD_WIN
        elif self._is_terminal:
            reward = HexEnvironment.REWARD_LOSS
        else:
            reward = HexEnvironment.REWARD_ACTION
        return reward
    
    def is_finished(self):
        return self._is_terminal

    def get_initial_observation(self):
        """Returns the agents' perceivable state of the initial environment."""
        init_board = np.zeros((self.board_size, self.board_size), dtype=int).flatten()
        return (HexEnvironment.STARTING_PLAYER, *init_board)

    def get_observation(self):
        """Returns the agents' perceivable state of the environment."""
        return (self._current_player, *self.board.flatten())

    def get_legal_actions(self):
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.hexgrid = HexGrid(self.board_size)
        self._current_player = 1
        self._actions = set(range(self.board.size))
        self._is_terminal = False
        self._step = 0
        return self

    def copy(self):
        obj = object.__new__(self.__class__)
        obj._spec = self._spec
        obj.board_size = self.board_size
        obj.board = self.board.copy()
        obj.hexgrid = self.hexgrid.copy()
        obj._current_player = self._current_player
        obj._actions = self._actions.copy()
        obj._is_terminal = self._is_terminal
        obj._step = self._step
        return obj
    
    @staticmethod
    def apply(state, action):
        """
        Returns the state defined by applying 'action' to 'state'.
        Convenience function, does no perform any move validation.
        """
        player, *board = state
        board[action] = player
        return (player % 2 + 1, *board)
    
    def __str__(self):
        return HexRenderer.board2string(self.board)


class HexRenderer:
    # Static variables
    THETA = np.radians(30)
    edge_color = "#606060"
    hex_colors = {
        0.: "#bbb",
        1.: "#F8A857",
        2.: "#57A8F8"
    }
    tri_colors = {
        1: "#FAC48E",
        2: "#85BFF9"
    }

    @staticmethod
    @lru_cache(maxsize=1)
    def _transform_matrix():
        """Applies an affine transformation which maps to the hex coordinate system."""
        # TODO: Rewrite transformations to using 'matplotlib.transforms.Affine2D'
        theta = HexRenderer.THETA
        cos, sin = np.cos(theta), np.sin(theta)

        # Construct affine transformations
        skew_matrix = np.array([[1, 0], [0, np.sqrt(3)/2]])     # Adjust 'y' coordinates for compact grid layout
        shear_matrix = np.array([[1, 0], [theta, 1]])
        rotation_matrix = np.array(((cos, -sin), (sin, cos)))
        return skew_matrix @ shear_matrix @ rotation_matrix

    @staticmethod
    def plot(board):
        # Create figure
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Create ramdonly filled board to see colors
        #board = np.random.randint(0, 3, board.shape)
        n = board.shape[0]
        low, high = 0, n - 1
        label_offset = 1
        tri_offset = 1.5

        # Add hexagons
        coords = np.indices(board.shape).astype(float)
        coords = coords.reshape(2, -1).T
        coords = coords @ HexRenderer._transform_matrix()

        for coord, player in zip(coords, board.flat):
            hexagon = RegularPolygon(coord, numVertices=6, radius=np.sqrt(1/3), orientation=HexRenderer.THETA, 
                facecolor=HexRenderer.hex_colors[player], edgecolor=HexRenderer.edge_color, zorder=1)
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

        for label, coords in zip([*alpha_labels, *numeric_labels], [*alpha_coords, *numeric_coords]):
            ax.text(*coords, label, fontsize=7, fontweight='bold', ha="center", va="center")

        # Add triangles in the background
        tri_low, tri_high = low - tri_offset, high + tri_offset
        tri_coords = np.array([
            (tri_high, tri_high), (tri_low, tri_low), 
            (tri_low, tri_high), (tri_high, tri_low), 
            ((tri_low + tri_high)/2, (tri_low + tri_high)/2)
        ])
        tri_coords = tri_coords @ HexRenderer._transform_matrix()
        t, b, l, r, c = tri_coords
        triangles = [((l, b, c), 1), ((r, t, c), 1), ((t, l, c), 2), ((r, b, c), 2)]

        for tri_coords, player in triangles:
            triangle = Polygon(tri_coords, facecolor=HexRenderer.tri_colors[player], edgecolor=HexRenderer.edge_color, zorder=0)
            ax.add_patch(triangle)
    
        # Display figure
        plt.autoscale(enable=True)
        plt.axis('off')
        plt.show(block=True)
        #plt.pause(0.1)
        #plt.close()

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
        headings = " "*5+(" "*3).join(ascii_uppercase[:cols])
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
