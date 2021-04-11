import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from base import Learner

from .tree import MCNode, MCTree


class MCTSLearner(Learner):
    def __init__(self, env, tree_policy, target_policy, network, n_episodes, n_simulations, save_interval, batch_size, replay_buffer):
        self.env = env                          # Game environment
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy)
        self.network = network                  # Neural net encoding a mapping from state space to action space
        self.n_episodes = n_episodes            # Number of search games
        self.n_simulations = n_simulations      # Number of simulated games from leaf node
        self.save_interval = save_interval      # Number of games between network checkpoints
        self.batch_size = batch_size            # Number of examples per batch from replay buffer
        self.replay_buffer = replay_buffer

    def get_action(self, state):
        """Ask the target policy for the probability distribution over actions."""
        return self.target_policy.get_action(state)

    def learn(self):
        """
        Does 'self.n_episodes' iterations of learning.
        Also saves network state every now and then.
        """
                                                                        
        self.replay_buffer.clear()
        self.network.save()

        for ep in range(self.n_episodes):
            env_actual = self.env.reset()
            self.step(env_actual)

            if ep % self.save_interval == 0:
                self.network.save()

    def step(self, env):
        """Does one iteration of learning."""
        self.root = root = MCNode.from_state(env.get_observation())
        self.mct = mct = MCTree(root, self.tree_policy, self.target_policy)

        while not env.is_finished():
            for _ in range(self.n_simulations):
                env_sim = env.copy()
                leaf = mct.search(env_sim)
                reward = mct.rollout(env_sim, leaf)
                mct.backpropagate(leaf, reward)

            D = mct.get_distribution(env, root)
            self.replay_buffer.add(root.state, D)
            action = np.argmax(D)
            env.move(action, root.player)
            self.root = root = root.successors[action]
            mct.change_root(root)

        x_batch, y_batch = self.replay_buffer.fetch_minibatch(batch_size=self.batch_size)
        self.network.train(x_batch, y_batch)
