import numpy as np


class Actor:
    """
    The actor manages interactions between a policy and an environment.
    The actor evaluates policies during training.

    Args:
        policy(object):     Object with the following methods:
                                __call__(self, state):              Evaluates state
                                initialize(self):                   Initializes policy to zero
                                update(self, state, action, diff):  Updates valuation of state by diff  
                                update_all(self, diff):             Updates valuation of all states-action pairs by diff

    The actor only sees an error term, not V(s).
    """

    def __init__(self, env, policy, epsilon=0.5):
        self.policy = policy
        self.e_matrix = np.zeros((env.state_space.size, env.action_space.size))

    def __call__(self, state):
        """Returns the actors' proposed action for a given state."""
        #if np.random.random() < self.epsilon:
        #    return 
        return self.policy(state)

    def set_eligibility(self, state, action, value=1):
        self.e_matrix[state, action] = value

    def reset_eligibility(self):
        self.e_matrix = np.zeros_like(self.e_matrix)

    def iter(self, state, action):
        self.policy.update(state, action)

    def update_all(self, error):
        self.policy.update_all(error, self.e_matrix)
