
from .flag import HexFlag


class HexGrid:
    def __init__(self, board_size):
        self.board_size = board_size
        self.grid = [None for _ in range(board_size**2)]
    
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
            flags = HexFlag.P1_UPPER if y == 0 else \
                    HexFlag.P1_LOWER if y == self.board_size - 1 else HexFlag.ISOLATED
        else:
            flags = HexFlag.P2_LOWER if x == 0 else \
                    HexFlag.P2_UPPER if x == self.board_size - 1 else HexFlag.ISOLATED
        
        # Create new disjoint set node
        _id = x * self.board_size + y
        self.grid[_id] = self._create_cell(_id=_id, player=player, flags=flags)

        # Find + merge neighborhood of the newly generated cell
        if neighbours := self._get_neighbours(_id, player):
            self.union_many(_id, neighbours)
            _id = self.find(_id)
        return self.grid[_id]['flags']  # Return flag of representative

    def copy(self):
        obj = object.__new__(self.__class__)
        obj.board_size = self.board_size
        obj.grid = [{**d} if d else None for d in self.grid]
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

    def union_many(self, _id, neighbours):
        """Convenience method for merging multiple disjoint sets."""
        for _id_neigh in neighbours:
            self.union(_id, _id_neigh)
