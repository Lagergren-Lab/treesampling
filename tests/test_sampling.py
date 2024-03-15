import numpy as np
import networkx as nx
import itertools
from scipy.stats import chisquare

from treesampling import algorithms
import treesampling.utils.graphs as tg


def test_log_random_uniform_graph():
    log_graph = tg.random_uniform_graph(5, log_probs=True)
    weight_matrix = nx.to_numpy_array(log_graph)
    # all weights except diagonals are < 0
    diag_mask = np.eye(weight_matrix.shape[0], dtype=bool)
    assert np.all(weight_matrix[diag_mask] == 0)
    assert np.all(weight_matrix[~diag_mask] < 0)


def test_random_k_trees_graph():
    n_nodes = 5
    root = 0
    graph, mst_tree = tg.random_tree_skewed_graph(n_nodes, 10)
    matrix = nx.to_numpy_array(graph)
    tg.reset_adj_matrix(graph, matrix)
    norm_graph = tg.normalize_graph_weights(graph)

    # Kirchhoff theorem for directed graphs, to get total weight of all trees
    tot_weight = tg.tuttes_tot_weight(norm_graph, root)
    log_graph = tg.reset_adj_matrix(graph, np.log(nx.to_numpy_array(norm_graph)))

    assert tg.tree_to_newick(nx.maximum_spanning_arborescence(norm_graph)) == tg.tree_to_newick(mst_tree)

    sample_size = 5000
    sample = {}
    acc = 0
    num = 0
    for i in range(sample_size):
        tree = algorithms.random_spanning_tree_log(log_graph, root=root)
        tree_nwk = tg.tree_to_newick(tree)
        if tree_nwk not in sample:
            weight = np.exp(tg.graph_weight(tree, log_probs=True))
            sample[tree_nwk] = weight
            num += 1
            acc += weight
    assert acc / tot_weight > 0.99, "sample did not cover all probability mass"
    extra_sample = 5000
    residual = 1 - acc / tot_weight
    unseen_freq = 0
    for i in range(extra_sample):
        tree = algorithms.random_spanning_tree_log(log_graph, root=root)
        if tg.tree_to_newick(tree) not in sample:
            unseen_freq += 1

    assert np.isclose(unseen_freq / extra_sample, residual, atol=2e-3), ("residual probability mass after large "
                                                                         "sample does not match with unseen "
                                                                         "trees occurrence")


def test_laplacian():
    n_nodes = 6

    # laplacian for number of undirected trees (no weight)
    weights = np.ones((n_nodes, n_nodes))
    graph = nx.from_numpy_array(weights)
    lap_mat = nx.laplacian_matrix(graph, weight='weight').toarray()
    print(lap_mat)
    assert np.isclose(np.linalg.det(lap_mat[1:, 1:]), tg.cayleys_formula(n_nodes))

    # laplacian for total weight of undirected trees
    weights = np.random.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0)
    graph = nx.from_numpy_array(weights)
    norm_graph = tg.normalize_graph_weights(graph)
    lap2 = nx.laplacian_matrix(norm_graph).toarray()
    lap_tot_weight = np.linalg.det(lap2[1:, 1:])

    tot_weight = 0
    for pruf_seq in itertools.product(range(n_nodes), repeat=n_nodes - 2):
        tree = nx.from_prufer_sequence(list(pruf_seq))
        for e in tree.edges():
            tree.edges()[e]['weight'] = norm_graph.edges()[e]['weight']
        tw = tg.graph_weight(tree)
        tot_weight += tw

    assert np.isclose(tot_weight, lap_tot_weight)

    # laplacian for directed rooted trees
    # root is 0
    weights[:, 0] = 0
    graph = nx.from_numpy_array(weights)
    norm_graph = tg.normalize_graph_weights(graph)
    tot_weight0 = 0
    all_trees = tg.enumerate_rooted_trees(n_nodes, root=0, weighted_graph=norm_graph)
    for tree in all_trees:
        tw = tg.graph_weight(tree)
        tot_weight0 += tw

    laplacian_tot_weight = tg.tuttes_tot_weight(norm_graph, 0)
    assert np.isclose(tot_weight0, laplacian_tot_weight)


def test_uniform_graph_sampling():
    n_nodes = 7
    adj_mat = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adj_mat, 0)
    graph = nx.from_numpy_array(adj_mat)
    # cardinality of tree topology
    tot_trees = tg.cayleys_formula(n_nodes)

    sample_size = 3 * tot_trees
    sample_dict = {}
    for s in range(sample_size):
        tree = algorithms.random_spanning_tree(graph, root=0)
        tree_nwk = tg.tree_to_newick(tree)
        if tree_nwk not in sample_dict:
            sample_dict[tree_nwk] = 0
        sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1

    unique_trees_obs = len(sample_dict)
    freqs = np.pad(np.array([v for k, v in sample_dict.items()]) / sample_size,
                   (0, tot_trees - unique_trees_obs))
    # test against uniform distribution
    test_result = chisquare(f_obs=freqs)
    assert test_result.pvalue >= 0.95, f"chisq test not passed: evidence that distribution is not uniform"






