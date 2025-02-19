import time
import networkx as nx
import numpy as np

from treesampling.algorithms.castaway import CastawayRST

from treesampling.utils.math import gumbel_max_trick_sample
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
    if log_probs:
        min_log_weight = np.log(np.nextafter(0, 1))
        matrix[matrix < min_log_weight] = -np.inf
        W = np.exp(matrix)
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
        # print(arc_candidates)
        # print(tree.edges())
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
            aa *= tuttes_tot_weight(graph, root, contracted_arcs=[e for e in tree.edges()],
                                    deleted_arcs=deleted_arcs)

    return tree


def wilson_rst(graph: nx.DiGraph, root=0, log_probs: bool = False) -> nx.DiGraph:
    n_nodes = graph.number_of_nodes()
    # normalize graph
    norm_graph = normalize_graph_weights(graph, log_probs=log_probs)
    weights = nx.to_numpy_array(norm_graph)
    tree = nx.DiGraph()
    t_set = {root}
    x_set = {x for x in list(range(n_nodes))}.difference(t_set)
    # pick random node from x
    prev = [None] * n_nodes
    while x_set:
        i = np.random.choice(list(x_set))
        u = i
        while u not in t_set:
            # loop-erased random walk
            if log_probs:
                prev[u] = gumbel_max_trick_sample(weights[:, u])
            else:
                prev[u] = np.random.choice(np.arange(n_nodes), size=1, p=weights[:, u])[0]
            u = prev[u]
        u = i
        while u not in t_set:
            # add to tree
            tree.add_edge(prev[u], u, weight=weights[prev[u], u])
            x_set.remove(u)
            t_set.add(u)
            u = prev[u]
    return tree

def colbourn_rst(graph: nx.DiGraph, root=0, log_probs: bool = False):
    """
    Re-adapted from rycolab/treesample implementation
    :param graph:
    :param root:
    :param log_probs:
    :return:
    """
    # normalize graph weights
    graph = normalize_graph_weights(graph, log_probs=log_probs)
    W = nx.to_numpy_array(graph)
    if log_probs:
        # filter probs that are too low and take exp
        min_log_weight = np.log(np.nextafter(0, 1))
        W[W < min_log_weight] = -np.inf
        W = np.exp(W)
        graph = reset_adj_matrix(graph, W)
        graph = normalize_graph_weights(graph, log_probs=False)

    nodes_perm = [i for i in range(W.shape[1])]
    if root != 0:
        nodes_perm = [root] + [i for i in range(W.shape[1]) if i != root]
        W = W[:, nodes_perm]
    tree = _colbourn_tree_from_matrix(W)

    tree = nx.relabel_nodes(tree, {i: nodes_perm[i] for i in range(W.shape[1])})
    for e in tree.edges():
        tree.edges()[e]['weight'] = graph.edges()[e]['weight']
    return tree


def _sample_edge(j, B, A, r) -> tuple[int, float]:

    # compute the marginals
    n = A.shape[0]
    marginals = np.zeros(n)
    for i in range(n):
        if i == j:
            marginals[i] = B[0, i] * r[i]
        else:
            if j != 0:
                marginals[i] += B[j, j] * A[i, j]
            if i != 0:
                marginals[i] -= B[i, j] * A[i, j]
    # correct very small numbers to 0 due to float precision leading to
    # subtractions a - a != 0
    marginals[marginals < 1e-50] = 0
    # re-normalize
    marginals /= np.sum(marginals)
    out = np.random.choice(np.arange(n), p=marginals)
    return out, float(marginals[out])


def _update_BL(i, j, B, L, A, r) -> tuple[np.ndarray, np.ndarray]:
    # code is copied from rycolab/treesample/colbourn.py - credits to
    # condition the laplacian so that i -> j is in any tree
    # K is the laplacian
    n = B.shape[0]
    uj = np.zeros(n)
    if i == j:
        uj[0] = r[j]
    else:
        if j != 0:
            uj[j] = A[i, j]
        if i != 0:
            uj[i] = -A[i, j]
    # update B and L
    u = uj - L[:, j]
    L[:, j] = uj
    bj = B[:, j]
    ub = u.T @ bj
    s = 1 + ub
    B -= np.outer(bj, u.T @ B) / s
    return B, L


def _colbourn_tree_from_matrix(W: np.ndarray) -> nx.DiGraph:
    """
    Assumes root is 0. Wrapper can permute nodes so to arbitrarily set the root. See main function colbourn_rst
    :param W: weight matrix
    :return: nx.DiGraph with tree edges only (is_arborescence = True)
    """
    # nodes
    n = W.shape[0] - 1
    r = W[0, 1:]
    A = W[1:, 1:]
    np.fill_diagonal(A, 0)
    # Kirchoff matrix
    L = _koo_laplacian(A, r)
    B = np.linalg.inv(L).transpose()
    tree = nx.DiGraph()

    for j in range(n):
        i, p_i = _sample_edge(j=j, B=B, A=A, r=r)
        if i == j:
            # i is root
            tree.add_edge(0, j + 1)
        else:
            tree.add_edge(i + 1, j + 1)
        B, L = _update_BL(i, j, B, L, A, r)

    assert nx.is_arborescence(tree)
    return tree


def _koo_laplacian(A, r):
    """
    Root-weighted Laplacian of Koo et al. (2007)
    A is the adjacency matrix and r is the root weight
    """
    L = -A + np.diag(np.sum(A, 0))
    L[0] = r
    return L


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
