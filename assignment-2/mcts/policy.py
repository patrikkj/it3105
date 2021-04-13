import random


def tree_policy(node, c=1):
    """UCT search policy."""
    is_max = node.player == 1
    ucts = [s.uct(c, is_max=is_max) for s in node.successors.values()]
    func = max if is_max else min
    return func(zip(ucts, node.successors.keys()))[1]
    #return func(node.successors.items(), key=lambda node: node[1].uct(c, is_max=is_max))[0]

def random_policy(state, env=None):
    return random.choice(list(env.get_legal_actions()))
