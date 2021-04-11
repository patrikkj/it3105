import numpy as np

from base import Actor


class MCTSActor(Actor):
    def __init__(self, env, network):
        self.env = env
        self.network = network
        self._action_space = set(range(self.env.spec.actions))

    def __call__(self, state, env=None):
        env = env or self.env
        state = np.array(state).reshape(1, -1)
        probs = self.network(np.where(state==2, -1, state)).numpy().flatten()
        
        # Remove illegal actions (no need to normalize probabilities)
        legal_actions = env.get_legal_actions()
        illegal_actions = legal_actions ^ self._action_space
        probs[list(illegal_actions)] = 0
        return np.argmax(probs)
