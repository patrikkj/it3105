import numpy as np
from utils import SAP


class Actor:
    """
    Har i oppgave å bestemme nye actions.
    The actor manages interactions between a policy and an environment.
    The actor evaluates policies during training.

    Args:
            env:                    Environment, used to fetch state space specifications.
            alpha:                  Actor's learning rate.
            decay_rate:             Trace decay rate (λ) (eligibility decay / decay for prev. states)
            discount_rate:          Discount rate (𝛾) / future decay rate for depreciating future rewards.
            epsilon:                Rate of exploration.
            epsilon_decay:          Rate at which the exploration rate decay.
    """

    def __init__(self, alpha=0.01, decay_rate=0.9, discount_rate=0.9, 
                 epsilon=0.5, epsilon_min=0, epsilon_decay=0.999, reset_on_explore=True):

        # Eligibility
        self.eligibility = {}  # Format: sap -> elig_value between 0 and 1
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

        # Policy
        self.policy = {}  # FORMAT: sap -> desirability of doing action 'a' in state 's'
        self.alpha = alpha
        self.episode = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reset_on_explore = reset_on_explore

    def __call__(self, state, legal_actions, training=True):
        """Returns the actors' proposed action for a given state."""
        # Fetch legal actions from environment
        saps = [SAP(state, action) for action in legal_actions]
        for sap in saps:
            self.policy.setdefault(sap, 0)
            self.eligibility.setdefault(sap, 0)
        # Determine action based on policy
        self._current_epsilon = max(self.epsilon * self.epsilon_decay ** self.episode, self.epsilon_min)
        if (np.random.random() < self._current_epsilon) and training:
            action = legal_actions[np.random.choice(len(legal_actions))]
            return action, True
        else:
            #action = np.random.choice(saps, weights=[self.policy[sap] for sap in saps]).action
            action = max(saps, key=lambda sap: self.policy.get(sap, 0)).action
            return action, False

    def set_episode(self, episode):
        self.episode = episode

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, sap):
        for sap_, value in self.eligibility.items():
            self.eligibility[sap_] = self.decay_rate * self.discount_rate * value
        self.eligibility[sap] = 1

    def update_policy(self, error):
        for sap, elig in self.eligibility.items():
            self.policy[sap] += self.alpha * error * elig

    def update(self, sap, error, is_exploring):
        if is_exploring and self.reset_on_explore:
            self.reset_eligibility()
        self.update_eligibility(sap)
        self.update_policy(error)
