from base import Agent, LearningAgent
from network import ActorNetwork
from replay import ReplayBuffer

from .actor import MCTSActor
from .learner import MCTSLearner
from .policy import tree_policy, random_policy
from .tree import MCNode, MCTree


class MCTSAgent(LearningAgent):
    def __init__(self, env, actor, learner):
        self.env = env
        self.actor = actor
        self.learner = learner

    def get_action(self, state):
        return self.actor(state, env=self.env)
    
    def learn(self):
        self.learner.learn()

    def save(self):
        self.actor.save()

    def load(self, path):
        self.actor.load()

    @classmethod
    def from_config(cls, env, config):
        network = ActorNetwork(env, **config["network_params"])
        replay_buffer = ReplayBuffer(**config["buffer_params"])
        actor = MCTSActor(env, network)
        learner = MCTSLearner(
            env=env,
            tree_policy=tree_policy,
            target_policy=actor,
            network=network,
            replay_buffer=replay_buffer,
            **config["learner_params"])
        return cls(env, actor, learner)

    @staticmethod
    def from_file(config_path):
        pass


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
