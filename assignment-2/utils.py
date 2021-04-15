import cProfile
import time
from functools import wraps

from graphviz import Digraph

i = 0

def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            out = func(*args, **kwargs)
        pr.print_stats(sort=1)
        return out
    return wrapper

def timeit(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        diff = te - ts
        print(f"{method.__name__}: {diff:.8f} s")
        return result
    return timed

def flatten_nodes(node, successor_attr='children'):
    # base case
    successors = getattr(node, successor_attr)
    if isinstance(successors, dict):
        successors = successors.values()
    if not successors:
        return [node]
    return [node] + [n for successor in successors for n in flatten_nodes(successor, successor_attr=successor_attr)]

def visualize_graph(root, 
                    directory='exported_graphs', 
                    file_prefix="graph", 
                    successor_attr='', 
                    label_func=str, 
                    edge_label_func=None, 
                    root_to_none=True, 
                    show=False):
    g = Digraph()

    #g = Digraph(format='png')
    nodes = flatten_nodes(root, successor_attr=successor_attr)
    node_to_id = {node : str(id_) for id_, node in enumerate(nodes)}

    # Add all nodes to graph
    for idx, node in enumerate(nodes):
        if root_to_none:
            g.node(node_to_id[node], label=label_func(node) if idx != 0 else 'None', ordering='in')
        else:
            g.node(node_to_id[node], label=label_func(node), ordering='in')


    # Add all edges to graph
    for node in nodes:
        successors = getattr(node, successor_attr)
        if isinstance(successors, dict):
            successors = successors.values()
        for successor in successors:
            if edge_label_func:
                g.edge(node_to_id[node], node_to_id[successor], label=edge_label_func(node, successor))
            else:
                g.edge(node_to_id[node], node_to_id[successor])
        
    # Export graph
    global i
    #g.render(filename=f"{directory}/{i:03}_{file_prefix}", format="png", cleanup=True)
    i += 1

    # Plot graph
    if show:
        g.view()
