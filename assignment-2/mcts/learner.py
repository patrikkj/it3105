import json
import pickle
import numpy as np
from base import Learner

from .tree import MCNode, MCTree


class MCTSLearner(Learner):
    def __init__(self, env, tree_policy, target_policy, network, replay_buffer, n_episodes, n_simulations, save_interval, batch_size, agent_dir):
        self.env = env                          # Game environment
        self._tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self._target_policy = target_policy      # Used for rollout simulation (default policy)
        self._network = network                  # Neural net encoding a mapping from state space to action space
        self._replay_buffer = replay_buffer
        self.n_episodes = n_episodes            # Number of search games
        self.n_simulations = n_simulations      # Number of simulated games from leaf node
        self.save_interval = save_interval      # Number of games between network checkpoints
        self.batch_size = batch_size            # Number of examples per batch from replay buffer
        self.agent_dir = agent_dir

    def save(self, episode):
        """Creates a checkpoint of the network and state of replay buffer."""
        self._network.save(self.agent_dir, episode)
        with open(f"{self.agent_dir}/checkpoints/{episode}_replay_buffer.p", "wb") as f:
            pickle.dump(self._replay_buffer, f)

    def learn(self):
        """
        Does 'self.n_episodes' iterations of learning.
        Also saves network state every now and then.
        """
        self._replay_buffer.clear()
        self.save(episode=0)

        for ep in range(1, self.n_episodes + 1):
            env_actual = self.env.reset()
            self.step(env_actual)

            if ep % self.save_interval == 0:
                self.save(episode=ep)

    def step(self, env):
        """Does one iteration of learning."""
        self.root = root = MCNode.from_state(env.get_initial_observation())
        self.mct = mct = MCTree(root, self._tree_policy, self._target_policy)

        while not env.is_finished():
            for _ in range(self.n_simulations):
                env_sim = env.copy()
                leaf = mct.search(env_sim)
                leaf = mct.node_expansion(env_sim, leaf)
                reward = mct.rollout(env_sim, leaf)
                mct.backpropagate(leaf, reward)

            D = mct.get_distribution(env, root)
            self._replay_buffer.add(root.state, D)
            action = np.argmax(D)
            env.move(action, root.player)
            self.root = root = root.successors[action]
            mct.change_root(root)

        x_batch, y_batch = self._replay_buffer.fetch_minibatch(batch_size=self.batch_size)
        self._network.train(x_batch, y_batch)


    # -------------------- #
    # Object serialization #
    # -------------------- #
    def serialize(self):
        # Save state of current instance (Exclude private members and env. reference)
        config = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        del config['env']
        with open(f"{self.agent_dir}/learner_config.json", 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def deserialize(cls, env, tree_policy, target_policy, network, replay_buffer, agent_dir):
        with open(f"{agent_dir}/learner_config.json") as f:
            config = json.load(f)
        return cls(env, tree_policy, target_policy, network, replay_buffer, **config)
