import time
import networkx as nx
import numpy as np
import scipy.special as sp

from treesampling.algorithms.castaway import CastawayRST
from treesampling.algorithms.wilson import wilson_rst
from treesampling.algorithms.colbourn import colbourn_rst

from treesampling.utils.graphs import graph_weight, tuttes_tot_weight, reset_adj_matrix

from treesampling.utils.graphs import random_uniform_graph, normalize_graph_weights
from warnings import warn


def random_spanning_tree(graph: nx.DiGraph, root=0, trick: bool = True) -> nx.DiGraph:
    smplr = CastawayRST(graph, root, log_probs=False, trick=trick)
    warn('Use the new object ' + smplr.__class__.__name__ + ' with log_probs=False', DeprecationWarning, stacklevel=2)
    return smplr.sample_tree()


def random_spanning_tree_log(graph: nx.DiGraph, root=0, trick: bool = True) -> nx.DiGraph:
    smplr = CastawayRST(graph, root, log_probs=True, trick=trick)
    warn('Use the new object ' + smplr.__class__.__name__ + ' with log_probs=True', DeprecationWarning, stacklevel=2)
    return smplr.sample_tree()


def kirchoff_rst(graph: nx.DiGraph, root=0, log_probs: bool = False) -> nx.DiGraph:
    """
    Implementation of Kulkarni A8 algorithm for directed graphs.
    """
    # set root column to ones (it only matters it's not zero)
    matrix = nx.to_numpy_array(graph)
    matrix[:, root] = 1.
    graph = reset_adj_matrix(graph, matrix)
    # normalize graph weights
    graph = normalize_graph_weights(graph, log_probs=log_probs)
    graph.remove_edges_from([(u, v) for u, v in graph.edges() if u == v or v == root])
    if log_probs:
        min_log_weight = np.log(np.nextafter(0, 1))
        matrix[matrix < min_log_weight] = -np.inf
        W = np.exp(matrix)
        # after exp, there might be columns with all zeros, there uniform distribution is assumed
        null_col = ~np.any(W > 0, axis=0)
        W[:, null_col] = 1.
        W[np.diag_indices(W.shape[0])] = 0. # remove self loops

        graph = reset_adj_matrix(graph, W)
        graph = normalize_graph_weights(graph, log_probs=False)

    # initialize empty digraph
    tree = nx.DiGraph()
    arc_candidates = sorted([e for e in graph.edges() if e[0] != e[1]], key=lambda x: graph.edges()[x]['weight'],
                            reverse=True)
    aa = tuttes_tot_weight(graph, root)
    deleted_arcs = []
    while len(tree.edges()) < graph.number_of_nodes() - 1:
        # pick edge (sorted by weight so to increase acceptance ratio)
        # print("candidates: ", arc_candidates)
        # print("current tree: ", tree.edges())
        arc = arc_candidates.pop(0)
        # sample uniform (0,1) and check if edge is added
        # sum of weights of trees including tree edges
        a = aa
        # sum of weights of trees including tree edges + e (a' in Kulkarni A8)
        # Leverage score of arc
        aa = 1. if not tree.edges() else graph_weight(tree)
        aa *= graph.edges()[arc]['weight'] * tuttes_tot_weight(graph, root,
                                                               contracted_arcs=[e for e in tree.edges()] + [arc],
                                                               deleted_arcs=deleted_arcs)
        acceptance_ratio = aa / a
        # print(f"a: {a}, aa: {aa}")
        # print(f"acceptance ratio: {acceptance_ratio}")
        if np.random.random() < acceptance_ratio:
            # print(f"adding edge {arc}")
            tree.add_edge(*arc)
            tree.edges()[arc]['weight'] = graph.edges()[arc]['weight']
            # remove edges going in edge[1] from candidates or opposite
            for e in arc_candidates.copy():
                if e[1] == arc[1] or e[::-1] == arc:
                    arc_candidates.remove(e)
        else:
            # print(f"excluding edge {arc}")
            # exclude edge from future consideration
            deleted_arcs.append(arc)
            # recompute aa
            aa = 1. if not tree.edges() else graph_weight(tree)
            aa *= tuttes_tot_weight(graph, root, contracted_arcs=[e for e in tree.edges()], deleted_arcs=deleted_arcs)

    return tree


def stable_matrix_exp(W, root=0):
    # normalize graph weights
    W = W - sp.logsumexp(W, axis=0)
    # filter probs that are too low and take exp
    min_log_weight = np.log(np.nextafter(0, 1))
    # if there exists a column with all values below min_log_weight, set the max weight to log(1)
    null_col = np.nonzero(~np.any(W > min_log_weight, axis=0))[0].tolist()
    for nc in null_col:
        max_idx = np.argmax(W[:, nc])
        W[max_idx, nc] = 0.

    W = np.exp(W)
    W[np.diag_indices(W.shape[0])] = 0  # remove self loops
    W[:, root] = 0
    return W




if __name__ == '__main__':
    # repeat for different number of nodes
    root = 1
    for n_nodes in [8, 9, 10]:
        trees_sample = {}
        graph = random_uniform_graph(n_nodes, normalize=True)
        sample_size = 500

        # our vs wilson: uniform graph
        start = time.time()
        for s in range(sample_size):
            tree = random_spanning_tree(graph, root)
        print(f"our time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        start = time.time()
        for s in range(sample_size):
            tree = wilson_rst(graph, root)
        print(f"wilson time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        start = time.time()
        for s in range(sample_size):
            tree = colbourn_rst(graph, root)
        print(f"colbourn time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        # print(tree_to_newick(tree))

    # 2-components weakly connected graph
    print("2 weakly connected components test")
    n_nodes = 8
    trees_sample = {}
    weights = np.random.random((n_nodes, n_nodes))
    component2 = [5, 6, 7]
    # divide into two components
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (i in component2) ^ (j in component2):
                weights[i, j] = np.random.random(1).item() * 1e-3
    np.fill_diagonal(weights, 0)

    graph = nx.from_numpy_array(weights)
    sample_size = 500

    # our vs wilson: uniform graph
    start = time.time()
    for s in range(sample_size):
        tree = random_spanning_tree(graph, 0)
    print(f"our time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
    start = time.time()
    for s in range(sample_size):
        tree = wilson_rst(graph, 0)
    print(f"wilson time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        # print(tree_to_newick(tree))
