import os
import pickle
from datetime import datetime

from base import Agent, LearningAgent
from network import ActorNetwork
from replay import ReplayBuffer

from .actor import MCTSActor
from .learner import MCTSLearner
from .policy import random_policy, tree_policy
from .tree import MCNode, MCTree


class MCTSAgent(LearningAgent):
    def __init__(self, env, actor, learner, name=None, export_dir='saves'):
        self.env = env
        self.actor = actor
        self.learner = learner

        # Variables for serialization/deserialization
        self.name = name or f"mctsagent__{datetime.now():%Y_%m_%d__%H_%M_%S}"
        self.export_dir = export_dir
        self.agent_dir = f"{export_dir}/{name}"

    def get_action(self, state):
        return self.actor(state, env=self.env)

    def learn(self):
        """
        Start learning via monte carlo simulations.
        Creates a directory for storing agent state along with 
        checkpoints during the learning process.
        """
        os.makedirs(self.agent_dir, exist_ok=False)
        self.learner.serialize()
        self.learner.learn()


    # -------------------- #
    # Object serialization #
    # -------------------- #
    @classmethod
    def from_config(cls, env, config):
        export_dir = config.get("export_dir", "saves")
        name = config.get("name", None) or f"mctsagent__{datetime.now():%Y_%m_%d__%H_%M_%S}"
        agent_dir = f"{export_dir}/{name}"

        network = ActorNetwork(env, **config["network_params"])
        replay_buffer = ReplayBuffer(**config["buffer_params"])
        actor = MCTSActor(env, network)
        learner = MCTSLearner(
            env=env,
            tree_policy=tree_policy,
            target_policy=actor,
            network=network,
            replay_buffer=replay_buffer,
            agent_dir=agent_dir,
            **config["learner_params"])
        return cls(env, actor, learner, name=name, export_dir=export_dir)

    @classmethod
    def from_checkpoint(cls, env, export_dir, name, episode=0):
        agent_dir = f"{export_dir}/{name}"
        network = ActorNetwork.from_checkpoint(env, agent_dir, episode)
        with open(f"{agent_dir}/checkpoints/{episode}_replay_buffer.p", "rb") as f:
            replay_buffer = pickle.load(f)
        actor = MCTSActor(env, network)
        learner = MCTSLearner.deserialize(
            env=env,
            tree_policy=tree_policy,
            target_policy=actor,
            network=network,
            replay_buffer=replay_buffer,
            agent_dir=agent_dir)
        return cls(env, actor, learner, name=name, export_dir=export_dir)


class NaiveMCTSAgent(Agent):
    """Implements a subset of the functionality for the MCTS agent.
    Performs a sequence of rollouts from the current state as a basis for evaluation.
    """
    def __init__(self, env, n_simulations=10_000):
        self.env = env
        self.n_simulations = n_simulations
        self.tree_policy = tree_policy
        self.target_policy = random_policy

    def get_action(self, state):
        """Ask the tree policy for the probability distribution over actions."""
        # Does one iteration of learning.
        self.root = root = MCNode.from_state(state)
        self.mct = mct = MCTree(root, self.tree_policy, self.target_policy)

        for sim in range(self.n_simulations):
            env_sim = self.env.copy()
            leaf = mct.search(env_sim)
            leaf = mct.node_expansion(env_sim, leaf)
            reward = mct.rollout(env_sim, leaf)
            mct.backpropagate(leaf, reward)
        #mct.visualize()
        return self.tree_policy(root, c=0)
