import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from base import Learner


@dataclass
class MCNode:
    state: tuple
    parent: 'MCNode' = None
    successors: dict = field(default_factory=dict)
    Q: int = 0
    N: int = 0
    player: int = 0
    is_leaf: bool = True

    @staticmethod
    @lru_cache(maxsize=None)      # Meomizes node generation by state
    def from_state(state):
        return MCNode(state=state, player=state[0])

    @property
    def q(self):
        """Expected reward"""
        return (self.Q / self.N) if self.N else 0

    @property
    def u(self):
        """Exploration bonus"""
        if self.parent is None:
            return 0
        return math.log(self.parent.N)**0.5 / (1 + self.N)

    def uct(self, c, is_max=False):
        """Returns the Upper Confidence Bound for Trees (UCT) metric."""
        return self.q + c * (self.u if is_max else -self.u)

    # def uct(self, c, is_max=False):
    #     """Returns the Upper Confidence Bound for Trees (UCT) metric."""
    #     if is_max:
    #         return (self.Q / self.N) if self.N else 0 + \
    #                 c * (math.log(self.parent.N)**0.5 / (1 + self.N)) if self.parent is not None else 0
    #     else:
    #         return (self.Q / self.N) if self.N else 0 - \
    #                 c * (math.log(self.parent.N)**0.5 / (1 + self.N)) if self.parent is not None else 0


def tree_policy(node, c=1):
    """UCT search policy."""
    is_max = node.player == 1
    ucts = [s.uct(c, is_max=is_max) for s in node.successors.values()]
    func = max if is_max else min
    return func(zip(ucts, node.successors.keys()))[1]
    #return func(node.successors.items(), key=lambda node: node[1].uct(c, is_max=is_max))[0]

def default_policy(actions):
    return random.choice(list(actions))

class MCTree:
    """
    This implementation assumes that the state representation starts with
    the ID of the player eligible to make the next move.
    """
    def __init__(self, root, tree_policy, target_policy):
        MCNode.from_state.cache_clear()
        self.root = root
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy) (ActorNetwork in this case)

    def search(self, env):
        """ (1) - Tree Search
        Traversing the tree from the root to a leaf node by using the tree policy.
        """
        node = self.root
        while not node.is_leaf:
            action = self.tree_policy(node)
            env.move(action, node.player)
            node = node.successors[action]

        # Node is now a leaf node; expand if leaf has been visited before
        if node.N > 0 and not env.is_finished():
            node = self.node_expansion(env, node)
            #action = next(env.get_legal_actions(), None)
            action = self.tree_policy(node)
            env.move(action, node.player)
            node = node.successors[action]
        return node


    def node_expansion(self, env, node):
        """ (2) - Node expansion
        Generating some or all child states of a parent state, and then connecting the tree node housing
        the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
        """
        node.is_leaf = False    # Node is being expanded
        actions = env.get_legal_actions()
        for action in actions:
            state = env.apply(node.state, action)     # Yields the new state without affecting env
            successor = MCNode.from_state(state)
            successor.is_leaf = True
            node.successors[action] = successor
        return node

    def rollout(self, env, node):
        """ (3) - Leaf evaluation
        Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
        policy from the leaf node’s state to a final state.
        """
        if env.is_finished():
            return env.calculate_reward()

        player = node.player
        while not env.is_finished():
            action = self.target_policy(env.get_legal_actions())
            _, reward, _ = env.move(action, player)
            player = player % 2 + 1
            #node = node.successors[action]
        return reward

    def backpropagate(self, node, score):
        """ (4) - Backpropagation
        Passing the evaluation of a final state back up the tree, updating relevant data (see course
        lecture notes) at all nodes and edges on the path from the final state to the tree root.
        """
        while node is not None:
            node.Q += score
            node.N += 1
            node = node.parent

    def change_root(self, root):
        """
        Changes the root of the Monte Carlo Tree, ensuring that siblings of 
        the previous root are deteched from the remaining subtree.
        """
        #print(MCNode.from_state.cache_info())
        #MCNode.from_state.cache_clear() # Clears the node state memoization cache
        del self.root                   # Remove reference to parent such that it can be garbage collected
        self.root = root
        self.root.parent = None

    def get_distribution(self, env, node):
        """
        Returns the probability distribution generated by the monte carlo tree.
        Represented as a 1-d np.ndarray of length equal to |action space|.
        """
        probs = np.zeros(env.spec.actions)
        for i, successor in node.successors.items():
            probs[i] = successor.N
        return probs / probs.sum() 


class MCTSLearner(Learner):
    def __init__(self, env, tree_policy, target_policy, n_episodes, n_simulations, save_interval, batch_size, replay_buffer):
        self.env = env                          # Game environment
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy)
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
                                                                        #   1. is = save interval for ANET (the actor network) parameters
        self.replay_buffer.clear()                                      #   2. Clear Replay Buffer (RBUF)
        #self.network.initialize()                                       #   3. Randomly initialize parameters (weights and biases) of ANET
        #self.network.save()     # Save initial network

        for ep in range(self.n_episodes):                                      #   4. For ga in number actual games:
            env_actual = self.env.reset()                               #       (a) Initialize the actual game board (Ba) to an empty board.
            self.step(env_actual)                                       #       (e) Train ANET on a random minibatch of cases from RBUF

            #if ep % self.save_interval == 0:                            #       (f) if ga modulo is == 0:
                #self.network.save()                                     #           • Save ANET’s current parameters for later use in tournament play.

    def step(self, env):
        """Does one iteration of learning."""
        root = MCNode.from_state(env.get_observation())             #       (b) sinit ← starting board state
        mct = MCTree(root, self.tree_policy, self.target_policy)    #       (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit

        while not env.is_finished():                                #       (d) While Ba not in a final state:
            for sim in range(self.n_simulations):                          #           • For gs in number search games:
                env_sim = env.copy()                                #           • Initialize Monte Carlo game board (Bmc) to same state as root.
                leaf = mct.search(env_sim)                          #               – Use tree policy Pt to search from root to a leaf (L) of MCT. Update Bmc with each move.
                reward = mct.rollout(env_sim, leaf)                 #               – Use ANET to choose rollout actions from L to a final state (F). Update Bmc with each move.
                mct.backpropagate(leaf, reward)                     #               – Perform MCTS backpropagation from F to root.

            D = mct.get_distribution(env, root)                     #           • D = distribution of visit counts in MCT along all arcs emanating from root.
            self.replay_buffer.add(root.state, D)                   #           • Add case (state, D) to RBUF
            action = np.argmax(D)                                   #           • Choose actual move (a*) based on D
            env.move(action, root.player)                           #           • Update Ba to s*
            root = root.successors[action]                          #           • Perform a* on root to produce successor state s*
            mct.change_root(root)                                   #           • In MCT, retain subtree rooted at s*; discard everything else.

        batch = self.replay_buffer.fetch_minibatch(batch_size=self.batch_size)
        #self.network.train(batch)
