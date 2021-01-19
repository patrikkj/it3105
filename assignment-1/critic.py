import numpy as np


class Critic:
    def __init__(self, evaluator, n_states, alpha, lambda_, gamma):
        """
        Args:
            evaulator(object):      Object with the following methods:
                                        __call__(self, state):              Evaluates state
                                        initialize(self):                   Initializes evaluator with small random values
                                        update(self, state, diff):          Updates valuation of state by diff
                                        update_all(self, diff):             Updates valuation of all states by diff
            n_states:               The state space dimensionality
            alpha:                  Critic's learning rate.
            lambda_:                Trace decay rate (eligibility decay / decay for prev. states)
            gamma:                  Discount rate / future decay rate for depreciating future rewards.
        """
        self.evaluator = evaluator
        self.n_states = n_states
        self.e = np.zeros(n_states)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma


    def __call__(self, r, s, s_next):
        """Returns the critics' evaluation of a given state (TD error)."""
        return r + self.gamma * self.evaluator(s_next) - self.evaluator(s)

    def set_eligibility(self, s, value):
        self.e[s] = value

    def reset_eligibility(self):
        self.e = np.zeros_like(self.e)

    def update_all(self, error):
        self.e = self.gamma * self.lambda_ * self.e     # Eligibility decay
        self.evaluator.update_all(self, error)          # Update valuation function
