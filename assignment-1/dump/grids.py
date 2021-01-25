import networkx as nx

G = nx.generators.lattice.triangular_lattice_graph(3, 3)
print(f"Nodes: {G.nodes}")
print(f"Edges: {G.edges}")
for node in G.nodes:
    print(type(node))