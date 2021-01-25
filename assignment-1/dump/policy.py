from abc import abstractmethod

import numpy as np


class Policy:
    """The policy defines a mapping from state to 
    probabilities of selecting each possible action.

    From lecture rl-intro.pdf:
    --------------------------
    Greedy → pick action a* in s with max Q∗ value, or a* that leads to next state with max V∗ value.
    """

    def __init__(self, env, epsilon=0.5):
        self.epsilon = epsilon
        self.policy = np.zeros((env.state_space.size, env.action_space.size)) #Martin: hva betyr dette?
        self.env = env

    def get_initial_state(self): #Er denne nødvendig? np.zeros fikser vel det vi ønsker?
        """ Sets pi(s,a) = 0 (probability distribution) for all combinations of (s,a)"""
        return self.env.get_initial_state() #blir dette riktig? Den skal vel initialisere all policy til "0"?

    def __call__(self, state):
        """
        Returns action or probabilities over action space.
        Output: np.array of shape equal to action space.
            Martin: blir dette riktig? Skal ikke den returnere en enkelt action?
        """

    def update(self, s, a, diff):
        pass

    def update_all(self, diff, e_matrix):
        pass
