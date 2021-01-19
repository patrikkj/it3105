from abc import abstractmethod

import numpy as np


class Policy:
    """The policy defines a mapping from state to 
    probabilities of selecting each possible action.

    From lecture rl-intro.pdf:
    --------------------------
    Greedy → pick action a* in s with max Q∗ value, or a* that leads to next state with max V∗ value.
    """

    def __init__(self, env):
        self.policy = np.zeros((env.state_space.size, env.action_space.size))
        self.env = env

    def get_initial_state(self):
        return self.env.get_initial_state()

    @abstractmethod
    def __call__(self, state):
        """
        Returns action or probabilities over action space.
        Output: np.array of shape equal to action space.
        """
        pass

    @abstractmethod
    def update(self, s, a, diff):
        pass

    @abstractmethod
    def update_all(self, diff, e_matrix):
        pass



class GreedyPolicy(Policy):
    def __init__(self, env, greedy_arg):
        super().__init__(env)
        self.greedy_arg = greedy_arg

    @abstractmethod
    def __call__(self, state):
        """
        Returns action or probabilities over action space.
        Output: np.array of shape equal to action space.
        """
        pass

    @abstractmethod
    def update(self, s, a, diff):
        pass

    @abstractmethod
    def update_all(self, diff, e_matrix):
        pass
