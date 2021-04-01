from enum import Flag, auto
import numpy as np


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


class DisjointHexGrid:
    DIRECTIONS = np.array([(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, -1)])

    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = np.full((board_size, board_size), fill_value=None)

    def _get_neighbours(self, x, y, player):
        # Indices in (x, y)'s neighbourhood
        indices = DisjointHexGrid.DIRECTIONS + (x, y)

        # Remove out-of-bounds indices
        indices = indices[np.all((indices >= 0) & (indices < self.board_size), axis=1)]
        print(f"Neighborhood: {indices.tolist()}")

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
    

    # ----------- Disjoint set operations ------------------
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
