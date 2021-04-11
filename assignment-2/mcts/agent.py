from base import Agent, LearningAgent

from .tree import MCNode, MCTree, default_policy, tree_policy


class MCTSAgent(LearningAgent):
    def __init__(self, env, actor, learner, mode='network'):
        self.env = env
        self.actor = actor
        self.learner = learner
        self.mode = mode

    def get_action(self, state):
        if self.mode == 'network':
            return self.actor.get_action(state)
        elif self.mode == 'mcts':
            return self.learner.mct.tree_policy(self.learner.root, c=0)
    
    def learn(self):
        self.learner.learn()

    def save(self):
        ...

    def load(self):
        ...


class NaiveMCTSAgent(Agent):
    def __init__(self, env, n_simulations=1_000):
        self.env = env
        self.n_simulations = n_simulations
        self.tree_policy = tree_policy
        self.target_policy = default_policy

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
