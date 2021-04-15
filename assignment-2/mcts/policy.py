import random


def tree_policy(node, c=2):
    """UCT search policy."""
    is_max = node.player == 1
    ucts = [s.uct(c, is_max=is_max) for s in node.successors.values()]
    func = max if is_max else min
    best = func(ucts)
    return random.choice([k for k, v in zip(node.successors.keys(), ucts) if v == best])
    # best = func(zip(ucts, node.successors.keys()))[1]
    # return random.choice()
    # return func(zip(ucts, node.successors.keys()))[1]

def random_policy(state, env=None):
    return random.choice(list(env.get_legal_actions()))
