import numpy as np
from base import EnvironmentSpec, StateManager

from .flag import HexFlag
from .grid import HexGrid
from .renderer import HexRenderer


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
        assert not self._is_terminal, f"Game is over; cannot make any more moves."
        assert player == self._current_player, f"Invalid move; it's not player {player}'s turn!"
        assert action in self._actions, f"The selected action {action} is not allowed."

    def move(self, action, player):
        self._validate_move(action, player)

        # Perform action
        x, y = action // self.board_size, action % self.board_size
        self.board[x, y] = player
        self._actions.discard(action)

        flags = self.hexgrid.move(x, y, player)         # Apply move to internal representation
        self._is_terminal = HexFlag.is_win(flags)       # Determine if terminal state
            
        if self._is_terminal:
            self._winning_player = self._current_player
        self._current_player = self._current_player % 2 + 1
        self._step += 1
        return self.get_observation(), self.calculate_reward(), self._is_terminal

    def calculate_reward(self):
        """Determines the reward for the most recent step."""
        if self._is_terminal and self._winning_player == 1:
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
        self._winning_player = -1
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
        obj._winning_player = self._winning_player
        obj._actions = self._actions.copy()
        obj._is_terminal = self._is_terminal
        obj._step = self._step
        return obj
    
    def render(self, block=True, pause=0.1, close=True, callable_=None):
        title = f"Player {self._winning_player} won!" if self._is_terminal else None
        HexRenderer.render(self.board, block=block, pause=pause, close=close, title=title, callable_=callable_)
    
    def decode_state(self, state):
        """
        Overload this method if 'self.get_observation()' 
        returns a compressed state representation.
        Also works on arrays of states.
        """
        pids, boards = np.split(state, (1,), axis=-1)
        return np.atleast_2d(np.hstack((pids, boards==1, boards==2)))


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
