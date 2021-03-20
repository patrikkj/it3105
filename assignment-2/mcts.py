# The exact details of the MCTS algorithm that you decide to
# implement may vary slightly from these sources, but your code must perform these four basic processes:


# 1. Tree Search - Traversing the tree from the root to a leaf node by using the tree policy.
def tree_search(root):
    pass

# 2. Node Expansion - Generating some or all child states of a parent state, and then connecting the tree node housing
# the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
def generate_successors(node):
    pass

@staticmethod
def successor_gen(state, agent_index):
    actions = state.getLegalActions(agent_index)
    yield from ((state.generateSuccessor(agent_index, action), action) for action in actions)



# 3. Leaf Evaluation - Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
# policy from the leaf node’s state to a final state.


# 4. Backpropagation - Passing the evaluation of a final state back up the tree, updating