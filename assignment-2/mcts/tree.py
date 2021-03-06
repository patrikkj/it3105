import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import utils

from .policy import random_policy


@dataclass
class MCNode:
    state: tuple
    parent: 'MCNode' = None
    successors: dict = field(default_factory=dict)
    Q: float = 0            # Expected reward
    E: float = 0            # Total reward (evaluation)
    N: int = 0              # Num. visits
    player: int = 0

    #@lru_cache(maxsize=None)      # Meomizes node generation by state
    @staticmethod
    def from_state(state):
        return MCNode(state=state, player=state[0])

    def uct(self, c, is_max=False): # TODO: Maybe try math.ln(...)
        """Returns the Upper Confidence Bound for Trees (UCT) metric."""
        if is_max:
            return self.Q + c * (math.log(self.parent.N) / (1 + self.N))**0.5
        else:
            return self.Q - c * (math.log(self.parent.N) / (1 + self.N))**0.5

    def __str__(self):
        if self.N == 0:
            return ""
        board = self.state[1:]
        sides = int(len(board)**0.5)
        board = np.array(board, dtype=int).reshape(sides, sides)
        return "\n".join([
            np.array2string(np.array(board).reshape(sides, sides)),
            f"Player: {self.player}",
            f"Q: {self.Q}",
            f"E: {self.E}",
            f"N: {self.N}"
        ])

    def __hash__(self):
        return hash(self.state + self.parent.state if self.parent else tuple())


class MCTree:
    """
    This implementation assumes that the state representation starts with
    the ID of the player eligible to make the next move.
    """
    def __init__(self, root, tree_policy, target_policy, epsilon=0, uct_coeff=1):
        self.root = root
        self.tree_policy = tree_policy          # Used for tree traversal (which is usually highly exploratory)
        self.target_policy = target_policy      # Used for rollout simulation (default policy) (ActorNetwork in this case)
        self.epsilon = epsilon                  # Rate of exploration for this particular simulation tree
        self.uct_coeff = uct_coeff              # Exploration coefficient for tree policy
        #MCNode.from_state.cache_clear()         # Clears the node state memoization cache
        #print(MCNode.from_state.cache_info())
        
    def search(self, env):
        """ (1) - Tree Search
        Traversing the tree from the root to a leaf node by using the tree policy.
        """
        node = self.root
        while node.successors:
            action = self.tree_policy(node, c=self.uct_coeff)
            env.move(action, node.player)
            node = node.successors[action]
        return node

    def node_expansion(self, env, node):
        """ (2) - Node expansion
        Generating some or all child states of a parent state, and then connecting the tree node housing
        the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
        """
        if env.is_finished():
            return node

        # We don't expand a node if the branch has yet to be discovered
        if node.N == 0:
            return node

        # Perform expansion
        for action in env.get_legal_actions():
            state = env.apply(node.state, action)   # Returns new state without affecting env
            successor = MCNode.from_state(state)
            successor.parent = node
            node.successors[action] = successor

        # Traverse to the most suitable among the newly expanded nodes
        action = self.tree_policy(node, c=self.uct_coeff)
        env.move(action, node.player)
        node = node.successors[action]
        return node

    def rollout(self, env, node):
        """ (3) - Leaf evaluation
        Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
        policy from the leaf node’s state to a final state.
        """
        if env.is_finished():
            return env.calculate_reward()

        player = node.player
        state = node.state
        while not env.is_finished():
            if random.random() < self.epsilon:
                action = random_policy(state, env=env)
            else:
                action = self.target_policy(state, env=env)
            state, reward, _ = env.move(action, player)
            player = player % 2 + 1
        return reward

    def backpropagate(self, node, score):
        """ (4) - Backpropagation
        Passing the evaluation of a final state back up the tree, updating relevant data (see course
        lecture notes) at all nodes and edges on the path from the final state to the tree root.
        """
        while node is not None:
            node.E += score
            node.N += 1
            node.Q = node.E / node.N
            node = node.parent

    def change_root(self, root):
        """
        Changes the root of the Monte Carlo Tree, ensuring that siblings of
        the previous root are deteched from the remaining subtree.
        """
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

    def visualize(self):
        utils.visualize_graph(self.root, successor_attr='successors', show=True)
