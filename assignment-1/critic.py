import numpy as np
from abc import abstractmethod


class AbstractCritic:
    pass
    # @abstractmethod
    # def __call__(self, r, s, s_next):
    #     """Returns the critics' evaluation of a given state (TD error)."""
# 
    # @abstractmethod
    # def set_eligibility(self, s, value):
    #     """Sets the eligibility for a state to a specified value."""
# 
    # @abstractmethod
    # def reset_eligibility(self):
    #     """Resets the eligibility for all states."""
# 
    # @abstractmethod
    # def update_weights(self, error):
    #     """Updates the eligibility for all states by the specified error."""


class CriticTable(AbstractCritic):
    def __init__(self, env, alpha=0.01, decay_rate=0.9, discount_rate=0.99):
        """
        Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Critic's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
        """
        self.env = env
        self.V = {}                 # Format:   state -> value
        self.eligibility = {}
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

    def __call__(self, r, s, s_next):
        """Returns the critics' evaluation of a given state (TD error)."""
        v = self.V.setdefault(s, np.random.uniform(0.05, 0.1))
        v_next = self.V.setdefault(s_next, np.random.uniform(0.05, 0.1))
        self.eligibility.setdefault(s, 0)
        self.eligibility.setdefault(s_next, 0)
        return r + self.discount_rate * v_next - v

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, state):
        #print("critic_elig", len(self.eligibility))
        for state_, value in self.eligibility.items():
            self.eligibility[state_] = self.discount_rate * self.decay_rate * value     # Eligibility decay
        self.eligibility[state] = 1

    def update_value_func(self, error):
        #print("critic_values", len(self.V))
        for state, value in self.eligibility.items():
            self.V[state] += value * self.alpha * error

    def update_all(self, state, error, is_exploring):
        #if not is_exploring:
        self.update_eligibility(state)
        self.update_value_func(error)