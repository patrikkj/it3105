import json
import pickle

import numpy as np
from base import Learner

from .tree import MCNode, MCTree


class MCTSLearner(Learner):
    def __init__(self, 
                 env, 
                 tree_policy, 
                 target_policy, 
                 network, 
                 replay_buffer, 
                 n_episodes, 
                 n_simulations, 
                 n_simulations_decay, 
                 n_simulations_step_decay, 
                 save_interval, 
                 batch_size, 
                 agent_dir,
                 epsilon,
                 epsilon_decay,
                 _total_episodes=0):
        self.env = env                          # Game environment
        self._tree_policy = tree_policy         # Used for tree traversal (which is usually highly exploratory)
        self._target_policy = target_policy     # Used for rollout simulation (default policy)
        self._network = network                 # Neural net encoding a mapping from state space to action space
        self._replay_buffer = replay_buffer
        self.n_episodes = n_episodes                                # Number of search games
        self.n_simulations = n_simulations                          # Number of simulated games from leaf node
        self.n_simulations_decay = n_simulations_decay              # Number of simulated games from leaf node
        self.n_simulations_step_decay = n_simulations_step_decay    # Number of simulated games from leaf node
        self.save_interval = save_interval      # Number of games between network checkpoints
        self.batch_size = batch_size            # Number of examples per batch from replay buffer
        self.agent_dir = agent_dir
        self.epsilon = epsilon                  # Initial rate of exploration
        self.epsilon_decay = epsilon_decay      # Exploration decay rate
        
        # Internal variables for book-keeping
        self._total_episodes = _total_episodes
        self._episode = 1
        self._step = 1

    def save(self):
        """Creates a checkpoint of the network and state of replay buffer."""
        self._network.save(self._total_episodes)
        with open(f"{self.agent_dir}/checkpoints/{self._total_episodes}_replay_buffer.p", "wb") as f:
            pickle.dump(self._replay_buffer, f)

    def learn(self, episodes=None):
        """
        Does 'self.n_episodes' iterations of learning.
        Also saves network state every now and then.
        """
        episodes = episodes or self.n_episodes
        self._replay_buffer.clear()
        if self._total_episodes == 0:
            self.save()

        for _ in range(episodes):
            env_actual = self.env.reset()
            self.episode(env_actual)
            self._episode += 1
            self._total_episodes += 1
            if self._total_episodes % self.save_interval == 0:
                self.save()

    def episode(self, env):
        """Does one iteration of learning."""
        # Compute adaptive parameters
        epsilon = self.epsilon * self.epsilon_decay**self._episode
        n_simulations = self.n_simulations * self.n_simulations_decay**self._episode
        
        # Build a new monte-carlo tree for this particular game
        self.root = root = MCNode.from_state(env.get_initial_observation())
        self.mct = mct = MCTree(root, self._tree_policy, self._target_policy, epsilon=epsilon)

        self._step = 1
        while not env.is_finished():
            for _ in range(int(n_simulations)):
                env_sim = env.copy()
                leaf = mct.search(env_sim)
                leaf = mct.node_expansion(env_sim, leaf)
                reward = mct.rollout(env_sim, leaf)
                mct.backpropagate(leaf, reward)

            # Determine what action to make, and add this action to buffer
            D = mct.get_distribution(env, root)
            self._replay_buffer.add(root.state, D)
            action = np.argmax(D)
            env.move(action, root.player)
            self.root = root = root.successors[action]
            mct.change_root(root)

            # Update adaptive parameters
            n_simulations *= self.n_simulations_step_decay
            self._step += 1

        # At the end of each game, fetch training examples and update network weights
        x_batch, y_batch = self._replay_buffer.fetch_minibatch(batch_size=self.batch_size)
        self._network.train(x_batch, y_batch)
        self._network.decay_learning_rate()


    # -------------------- #
    # Object serialization #
    # -------------------- #
    def serialize(self, agent_dir):
        # Save state of current instance (Exclude private members and env. reference)
        config = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        del config['env']
        with open(f"{self.agent_dir}/learner_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        self._checkpoint_dir = f"{agent_dir}/checkpoints"
        self._network.serialize(agent_dir)

    @classmethod
    def deserialize(cls, env, tree_policy, target_policy, network, replay_buffer, agent_dir):
        config_path = f"{agent_dir}/learner_config.json"
        with open(config_path) as f:
            config = json.load(f)
        print(f"Successfully loaded MCTSLearner [config={config_path}]")
        return cls(env, tree_policy, target_policy, network, replay_buffer,
                   _total_episodes=network._episodes, **config)
