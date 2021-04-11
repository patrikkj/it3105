

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import viz


@dataclass
class MCNode:
    state: tuple
    parent: 'MCNode' = None
    successors: dict = field(default_factory=dict)
    Q: float = 0            # Expected reward
    E: float = 0            # Total reward (evaluation)
    N: int = 0              # Num. visits
    player: int = 0
    is_leaf: bool = True

    #@lru_cache(maxsize=None)      # Meomizes node generation by state
    @staticmethod
    def from_state(state):
        return MCNode(state=state, player=state[0])

    @property
    def u(self):
        """Exploration bonus"""
        if self.parent is None:
            return 0
        return math.log(self.parent.N)**0.5 / (1 + self.N)

    def uct(self, c, is_max=False):
        """Returns the Upper Confidence Bound for Trees (UCT) metric."""
        return self.Q + c * (self.u if is_max else -self.u)

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if k not in ("successors", "parent")) if self.N > 0 else "😃"
    
    def __hash__(self):
        return hash(self.state + self.parent.state if self.parent else tuple())

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
        #MCNode.from_state.cache_clear()
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
            successor.parent = node
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
            node.E += score
            node.N += 1
            node.Q = node.E / node.N
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

    def visualize(self):
        viz.visualize_graph(self.root, successor_attr='successors', show=True)
