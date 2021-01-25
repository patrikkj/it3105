from abc import abstractmethod
from enum import Enum, auto

import matplotlib.pyplot as plt

import env

TRIANGE_DIRECTIONS = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
DIAMOND_DIRECTIONS = [(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1)]


class BoardType(Enum):
    DIAMOND = auto()
    TRIANGLE = auto()
    # DIAMOND_FULL kanskje?


# INITIAL BOARD 5x5


# Configs
# Triangles 4-8
# Diamonds 3-6


tri_indices = gen_triangular_indices(3)


def gen_triangular_indices(n):
    return [(r, c) for r in range(n) for c in range(r + 1)]


def gen_grid_indices(n):
    return [(r, c) for r in range(n) for c in range(n)]


class Cell:
    def __init__(self, row, column, neighbours=None):
        self.row = row
        self.column = column
        self.has_peg = False

        if neighbours is None:
            self.neighbours = []
        else:
            self.neighbours = neighbours

    def change_state(self):
        """ Changes state from empty to filled or the opposite way."""
        self.has_peg = not self.has_peg


class Board:
    def __init__(self, indices, directions):
        self.cells = {(row, column): Cell(row, column) for row, column in indices}
        self.directions = directions

    def _gen_neighbours_to_cell(self, cell):
        # row, column = cell.row, cell.column
        for direction in self.directions:
            # row_ = cell.row + direction[0]
            # column_ = cell.column + direction[1]
            row_, col_ = self.adjacent_cell(cell, direction)
            if (row_, col_) in self.cells:
                yield (row_, col_)

    def _populate_neighbours(self):
        for cell in self.cells.values():
            cell.neighbours = list(self.cells[self._gen_neighbours_to_cell(cell)])

    def fill_board(self, value):
        for cell in self.cells.values():
            cell.has_peg = value

    def set_holes(self, holes):
        for hole in holes:
            self.cells[hole] = False

    @staticmethod
    def from_type(board_type, board_size):
        if board_type == BoardType.TRIANGLE:
            indices = gen_triangular_indices(board_size)
            return Board(indices, TRIANGE_DIRECTIONS)
        elif board_type == BoardType.DIAMOND:
            indices = gen_grid_indices(board_size)
            return Board(indices, DIAMOND_DIRECTIONS)


class PegSolitaire(env.Environment):
    def __init__(self, board_type=BoardType.TRIANGLE, board_size=5, holes=1):
        self.board = Board.from_type(board_type, board_size)
        self.directions = self.board.directions

    def step(self, action):
        """
        Apply action to this environment.
        Returns:
            observation (object): an environment-specific object representing your observation of the environment.
            reward (float): amount of reward achieved by the previous action.
        """
        a = 1
        return (1, 2, 3), 10
        
    def get_current_step(self):
        """Returns the number of steps for the active episode."""
        pass

    def reset(self):
        """Resets the environment."""
        pass

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        pass

    def set_holes(self, holes):
        for cell in self.board.cells:
            coordinate = (cell.row, cell.column)
            if holes.contains(coordinate):
                cell.has_peg = False
            else:
                cell.has_peg = True

    def valid_actions(self):
        actions = []
        for cell in self.cells:
            cell_actions = self.cell_actions(cell)
            for action in cell_actions:
                actions.append((cell, action))
        return actions

    def cell_actions(self, cell):
        actions = []
        for direction in self.directions:
            row_, col_ = self.adjacent_cell(cell, direction)
            if (row_, col_) in self.cells:
                if self.cells[(row_, col_)].has_peg:
                    row_, col_ = self.adjacent_cell(self.cells[(row_, col_)], direction)
                    if (row_, col_) in self.cells:
                        if not self.cells[(row_, col_)].has_peg:
                            actions.append(direction)
        return actions

    def move(self, cell, direction):
        cell.has_peg = not cell.has_peg
        row_, col_ = self.adjacent_cell(cell, direction)
        self.cells[row_, col_].has_peg = not self.cells[row_, col_].has_peg
        row_, col_ = self.adjacent_cell(self.cells[(row_, col_)], direction)
        self.cells[row_, col_].has_peg = not self.cells[row_, col_].has_peg
        return self

    def adjacent_cell(self, cell, direction):
        row, col = cell.row, cell.col
        return row + direction[0], col + direction[1]

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        return False


def gen_diamond_indices(n):
    pass


def main():
    tri_indices = gen_triangular_indices(5)
    plt.scatter(*list(zip(*tri_indices)))
