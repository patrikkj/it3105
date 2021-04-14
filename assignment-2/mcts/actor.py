import random

import numpy as np
from base import Actor


class MCTSActor(Actor):
    def __init__(self, env, network):
        self.env = env
        self.network = network
        self._action_space = set(range(self.env.spec.actions))

    def __call__(self, state, env=None, use_probs=False):
        env = env or self.env
        state = env.decode_state(state)
        probs = self.network(state).numpy().flatten()
        
        # Remove illegal actions (no need to normalize probabilities)
        legal_actions = env.get_legal_actions()
        illegal_actions = legal_actions ^ self._action_space
        probs[list(illegal_actions)] = 0
        if use_probs:
            probs = probs / probs.sum()
            return np.random.choice(range(probs.size), p=probs)
        else:
            return np.argmax(probs)
