class HexEnvironment(Environment):
    REWARD_WIN = 500
    REWARD_ACTION = 0
    REWARD_LOSS = -500

    def __init__(self, board_size=6):
        self.board_size = board_size

        # Create a list of all valid indices, handy for iterating over all valid cells
        self._valid_indices = tuple(zip(*self._mask.nonzero()))

        # Initialize environment state (all initialization is done in self.reset())
        self.reset()

    def _generate_moves(self, player):
        return np.argwhere(self.board == 0)

    def move(self, action):
        """
        Apply action to this environment.

        Returns:
            observation (object): an environment-specific object representing your observation of the environment.
            reward (float): amount of reward achieved by the previous action.
            is_terminal (bool): whether the state is a terminal state
        """
        # Perform action
        x, y, player = action
        self.board[x, y] = player

        # Determine if terminal state
        self._actions.discard(action)
        self._is_terminal = bool(self._actions)

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

    def get_observation(self):
        """Returns the agents' perceivable state of the environment."""
        return self.board.astype(bool).tobytes()

    def get_legal_actions(self):
        return self._actions

    def reset(self):
        """Resets the environment."""
        self.board = self._board.copy()
        self._actions = set(map(tuple, np.argwhere(self.board == 0)))
        self._is_terminal = False
        self._step = 0

    def is_terminal(self):
        """Determines whether the given state is a terminal state."""
        # False if minimum number of moves for a terminal state hasnt been played
        if self.board.sum()
        return self._is_terminal

    def decode_state(self, state):
        # bytestring -> np.array([0, 1, 1, 0, 0, 1])
        return np.frombuffer(state, dtype=np.uint8)
