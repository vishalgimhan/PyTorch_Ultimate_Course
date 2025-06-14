#pip install networkx

# %% Packages
import networkx as nx

# %% Create the graph
my_nodes = list(range(5))
H = nx.DiGraph()

# %%
H.add_nodes_from(my_nodes)

# %% Add edges
H.add_edges_from([
    (1, 0),
    (1, 2),
    (1, 3),
    (3, 2),
    (3, 4),
    (4, 0)
])

# %% Visualize Graph
nx.draw(H, with_labels=True)

# %% Get Dense Adjacency Matrix
nx.adjacency_matrix(H).todense()

# %%
