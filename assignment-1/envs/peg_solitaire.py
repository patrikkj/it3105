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
            done (boolean): whether it’s time to reset the environment again. 
            info (dict): diagnostic information useful for debugging. 
        """

    @abstractmethod
    def reset(self):
        """Resets the environment."""

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial state for the environment."""

    @abstractmethod
    def is_terminal(self):
        """Determines whether the given state is a terminal state."""