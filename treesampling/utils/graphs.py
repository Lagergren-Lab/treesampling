import random
import itertools
from operator import mul
from functools import reduce
import numpy as np
import networkx as nx


# copied from VICTree
def tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)


def random_uniform_graph(n_nodes, log_probs=False) -> nx.DiGraph:
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    for u, v in graph.edges():
        if u == v:
            w = 0
        else:
            w = random.random()
        graph.edges()[u, v]['weight'] = w if not log_probs else np.log(w)
    return graph


def normalize_graph_weights(graph, log_probs=False, rowwise=True) -> nx.DiGraph:
    adj_mat = nx.to_numpy_array(graph)
    axis = 1 if rowwise else 0
    if not log_probs:
        adj_mat = adj_mat / adj_mat.sum(axis=axis, keepdims=True)
    else:
        adj_mat = adj_mat - np.logaddexp.reduce(adj_mat, axis=axis, keepdims=True)
    norm_graph = reset_adj_matrix(graph, adj_mat)
    return norm_graph


def reset_adj_matrix(graph: nx.DiGraph, matrix: np.ndarray) -> nx.DiGraph:
    weights_dict = [(i, j, matrix[i, j]) for i, j in itertools.product(range(matrix.shape[0]), repeat=2)]
    new_graph = graph.copy()
    new_graph.add_weighted_edges_from(weights_dict)
    return new_graph


def graph_weight(graph: nx.DiGraph, log_probs=False):
    if log_probs:
        # sum of weights
        w = graph.size(weight='weight')
    else:
        # product of weights
        w = reduce(mul, list(graph.edges()[e]['weight'] for e in graph.edges()), 1)

    return w


def cayleys_formula(n):
    assert n > 1
    return n**(n-2)