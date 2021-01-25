import numpy as np
from utils import SAP


class Actor:
    """
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

    def __init__(self, env, alpha=0.01, decay_rate=0.9, discount_rate=0.9, epsilon=0.5, epsilon_decay=0.999):
        self.env = env

        # Eligibility
        self.eligibility = {}  # Format: sap -> elig_value between 0 and 1
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

        # Policy
        self.policy = {}  # FORMAT: sap -> desirability of doing action 'a' in state 's'
        self.alpha = alpha
        self.episode = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def __call__(self, state):
        """Returns the actors' proposed action for a given state."""
        adjusted_epsilon = self.epsilon * self.epsilon_decay ** self.episode
        #print(adjusted_epsilon)
        # Fetch legal actions from environment
        legal_actions = self.env.get_legal_actions()
        saps = [SAP(state, action) for action in legal_actions]
        for sap in saps:
            self.policy.setdefault(sap, 0)
            self.eligibility.setdefault(sap, 0)

        # Determine action based on policy
        if (is_exploring := np.random.random() < adjusted_epsilon):
            action = legal_actions[np.random.choice(len(legal_actions))]
        else:
            action = max(saps, key=lambda sap: self.policy.get(sap, 0)).action
        return action, is_exploring

    def set_episode(self, episode):
        self.episode = episode

    def reset_eligibility(self):
        self.eligibility = {}

    def update_eligibility(self, sap):
        #print("actor_elig", len(self.eligibility))
        for sap_, value in self.eligibility.items():
            self.eligibility[sap_] = self.decay_rate * self.discount_rate * value
        self.eligibility[sap] = 1

    def update_weights(self, error):
        #print("actor_policy", len(self.policy))
        for sap, value in self.eligibility.items():
            self.policy[sap] += self.alpha * error * value

    def update_all(self, sap, error, is_exploring):
        #if not is_exploring:
        self.update_eligibility(sap)
        self.update_weights(error)
