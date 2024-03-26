import numpy as np
import networkx as nx

import treesampling.algorithms as ta
from treesampling.utils.graphs import enumerate_rooted_trees, cayleys_formula, kirchhoff_tot_weight, \
    normalize_graph_weights, tree_to_newick
from treesampling.utils.math import generate_random_matrix


def test_enumerate_rooted_trees():
    n_nodes = 5
    trees = enumerate_rooted_trees(n_nodes)
    tot_trees = cayleys_formula(n_nodes)
    assert len(trees) == tot_trees


def test_kirchhoff_tot_weight():
    n_nodes = 5
    weights = np.random.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0)
    graph = nx.from_numpy_array(weights)
    det_matrix = np.zeros_like(weights)
    for r in range(n_nodes):
        for c in range(n_nodes):
            det_matrix[r, c] = kirchhoff_tot_weight(graph, r, c)

    # print(np.abs(det_matrix))

    # ill conditioned
    weights = np.ones((n_nodes, n_nodes)) * 1000
    component2 = [2, 3, 4]
    # divide into two components
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (i in component2) ^ (j in component2):
                weights[i, j] = np.ones(1) * 1e-8
    weights[0, 2] = 1e-3
    weights[2, 0] = 1e-3
    np.fill_diagonal(weights, 0)

    graph = nx.from_numpy_array(weights)
    # graph = normalize_graph_weights(graph, rowwise=True)
    print("============ WEIGHTS ===============")
    print(nx.to_numpy_array(graph))
    det_matrix = np.zeros_like(weights)
    for r in range(n_nodes):
        for c in range(n_nodes):
            det_matrix[r, c] = np.abs(kirchhoff_tot_weight(graph, r, c))

    print("========== DETERMINANT OF LAPLACIAN MINORS =========")
    print(det_matrix)
    print(f"mean difference between det[L_00] and det[L_ij] {np.mean(det_matrix[0, 0] - det_matrix)}")

    print("sampling arborescences")
    ss = 1000
    sample = {}
    for s in range(ss):
        tree = ta.random_spanning_tree(graph, root=0)
        tree_nwk = tree_to_newick(tree)
        if tree_nwk not in sample:
            sample[tree_nwk] = 1
        else:
            sample[tree_nwk] += 1
    print(sorted(sample.items(), key=lambda u: u[1], reverse=True))
    # high condition number
    # L should be conditioned...
    weights = generate_random_matrix(n_nodes, condition_number=1e10)
    weights -= np.min(weights)
    # print(weights)
    graph = nx.from_numpy_array(weights)
    graph = normalize_graph_weights(graph, rowwise=True)
    det_matrix = np.zeros_like(weights)
    for r in range(n_nodes):
        for c in range(n_nodes):
            det_matrix[r, c] = np.abs(kirchhoff_tot_weight(graph, r, c))

    # print(det_matrix)
    # print(np.mean(det_matrix[0, 0] - det_matrix))



