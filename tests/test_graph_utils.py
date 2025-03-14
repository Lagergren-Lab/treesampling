import numpy as np
import networkx as nx
from statsmodels.stats.proportion import proportions_ztest

import treesampling.algorithms as ta
from treesampling.utils.graphs import enumerate_rooted_trees, cayleys_formula, kirchhoff_tot_weight, \
    normalize_graph_weights, tree_to_newick, reset_adj_matrix, tuttes_tot_weight, adjoint, graph_weight
from treesampling.utils.math import generate_random_matrix


def test_enumerate_rooted_trees():
    n_nodes = 5
    trees = enumerate_rooted_trees(n_nodes)
    tot_trees = cayleys_formula(n_nodes)
    assert len(trees) == tot_trees

def test_tuttes_tot_weight():
    n_nodes = 5
    weights = np.random.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0)
    graph = nx.from_numpy_array(weights)
    graph = normalize_graph_weights(graph)
    root = 0
    tot_weight = tuttes_tot_weight(graph, root)
    # enumerate all trees
    trees = enumerate_rooted_trees(n_nodes, root, graph)
    acc = 0
    for t in trees:
        acc += graph_weight(t)

    assert np.isclose(tot_weight, acc)

def test_edge_contraction():
    np.random.seed(0)
    n_nodes = 8
    # generate random adjacency matrix
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    weights = np.random.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0.)
    graph = reset_adj_matrix(graph, weights)
    graph = normalize_graph_weights(graph)
    e = (1, 2)
    e_weight = graph.edges()[e]['weight']
    # print(f"adj matrix: {weights}")
    print(f"contracted edge weight: {e_weight}")
    tot_weight = tuttes_tot_weight(graph, root=0)
    tot_contracted_weight = tuttes_tot_weight(graph, root=0, contracted_arcs=[e])
    tot_deleted_weight = tuttes_tot_weight(graph, root=0, deleted_arcs=[e])
    # compare tot weight and sum of contracted and deleted weight
    assert np.isclose(tot_weight, tot_contracted_weight * e_weight + tot_deleted_weight)
    tuttes_proportion = tot_contracted_weight * e_weight / tot_weight

    # now sample many trees and check the proportion of trees which contain that edge
    sample_size = 5000
    tree_proportion_with_e = 0
    for s in range(sample_size):
        tree = ta.random_spanning_tree(graph, root=0)
        if e in tree.edges:
            tree_proportion_with_e += 1
    tree_proportion_with_e /= sample_size
    # print the two proportions with annotations
    print(f"the following should be as equal as possible (sample size {sample_size})")
    print(f"empirical proportion of trees with edge e {e}: {tree_proportion_with_e}")
    print(f"proportion of weight of trees with edge e {e} (tuttes): {tuttes_proportion}")

    # prop test to verify whether the difference is statistically significant
    stat, pval = proportions_ztest(tree_proportion_with_e, sample_size, value=tuttes_proportion)
    # assert pval > 0.95
    print(f"pval: {pval}")

    # test with adjoint
    # compute the kirchhoff matrix
    # FIXME: why the adjoint method is not working?
    Din = np.diag(np.sum(weights, axis=0))
    K = Din - weights
    adjoint_K = adjoint(K)
    tot_weight_with_e = (adjoint_K[e[0], e[0]] - adjoint_K[e[1], e[0]])  # * e_weight
    print("with adjoint formula given in the Colbourn paper (Lemma 3.1)")
    print(f"{tot_weight_with_e / tot_weight} == {tuttes_proportion}")


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
    # FIXME: the condition number should be high for the laplacian matrix not the weight matrix
    #   issue: how to generate a matrix whose Laplacian has a high condition number?
    # L should be conditioned...
    weights = generate_random_matrix(n_nodes, condition_number=1e60)
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



