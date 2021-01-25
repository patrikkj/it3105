from abc import abstractmethod


class Environment:
    def __init__(self):
        self.action_space = None
        self.state_space = None

    @abstractmethod
    def step(self, action):
        """
        Apply action to this environment.
        Returns: 
            observation (object): an environment-specific object representing your observation of the environment.
            reward (float): amount of reward achieved by the previous action. 
        """

    @abstractmethod
    def get_current_step(self):
        """Returns the number of steps for the active episode."""

    @abstractmethod
    def get_observation_spec(self):
        """Returns the specifications for the observation space."""

    @abstractmethod
    def get_action_spec(self):
        """Returns the specifications for the action space."""

    @abstractmethod
    def get_legal_actions(self):
        """Returns the legal actions in the current state."""

    @abstractmethod
    def reset(self):
        """Resets the environment."""

    @abstractmethod
    def is_terminal(self):
        """Determines whether the given state is a terminal state."""

    