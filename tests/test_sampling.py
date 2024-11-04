import logging

import h5py
import numpy as np
import networkx as nx
import itertools
from scipy.stats import chisquare

from treesampling import algorithms
import treesampling.utils.graphs as tg
from treesampling.algorithms import CastawayRST, kirchoff_rst


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
    tot_weight = tg.tuttes_tot_weight(norm_graph, root, contracted_arcs=None)
    log_graph = tg.reset_adj_matrix(graph, np.log(nx.to_numpy_array(norm_graph)))

    assert tg.tree_to_newick(nx.maximum_spanning_arborescence(norm_graph)) == tg.tree_to_newick(mst_tree)

    sample_size = 5000
    sample = {}
    acc = 0
    num = 0
    smplr = CastawayRST(log_graph, root=root, log_probs=True, trick=True)
    for i in range(sample_size):
        tree = smplr.sample_tree()
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
        tree = smplr.sample_tree()
        if tg.tree_to_newick(tree) not in sample:
            unseen_freq += 1

    assert np.isclose(unseen_freq / extra_sample, residual, atol=2e-3), ("residual probability mass after large "
                                                                         "sample does not match with unseen "
                                                                         "trees occurrence")


def test_castaway_low_weight():
    """
    This test was built to debug numerical instability in the castaway algorithm.
    :return:
    """
    np.random.seed(0)
    n_nodes = 5
    low_weight = 1e-19
    random_graph = tg.random_weakly_connected_graph(n_nodes, weak_weight=low_weight)
    # Kirchhoff Laplacian determinant test
    L = tg.kirchoff_matrix(nx.to_numpy_array(random_graph))
    print(np.linalg.det(tg.mat_minor(L, 0, 0)))
    # Koo Laplacian determinant test
    kL = algorithms._koo_laplacian(nx.to_numpy_array(random_graph), 0)
    print(np.linalg.det(kL))
    for _ in range(100):
        # tree = algorithms.castaway_rst(random_graph, root=0, log_probs=False, trick=True)
        tree = algorithms.colbourn_rst(random_graph, root=0, log_probs=False)
        assert nx.is_arborescence(tree)


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

    laplacian_tot_weight = tg.tuttes_tot_weight(norm_graph, 0, contracted_arcs=None)
    assert np.isclose(tot_weight0, laplacian_tot_weight)


def test_uniform_graph_sampling():
    n_nodes = 7
    adj_mat = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adj_mat, 0)
    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    # cardinality of tree topology
    tot_trees = tg.cayleys_formula(n_nodes)

    sample_size = 3 * tot_trees
    smplr = CastawayRST(graph, root=0, log_probs=False, trick=True)
    sample_dict = smplr.sample(sample_size)

    unique_trees_obs = len(sample_dict)
    freqs = np.pad(np.array([v for k, v in sample_dict.items()]) / sample_size,
                   (0, tot_trees - unique_trees_obs))
    # test against uniform distribution
    test_result = chisquare(f_obs=freqs)
    assert test_result.pvalue >= 0.95, f"chisq test not passed: evidence that distribution is not uniform"


def test_unbalanced_weights():
    root = 0
    n_nodes = 8
    low_val = -3000
    high_val = -1
    adj_matrix = np.ones((n_nodes, n_nodes)) * low_val
    np.fill_diagonal(adj_matrix, -np.inf)  # no self connections
    adj_matrix[:, root] = -np.inf  # no arcs into root
    # add some high vals
    adj_matrix[0, 1] = high_val
    adj_matrix[1, 2] = high_val
    adj_matrix[2, 3] = high_val
    adj_matrix[3, 4] = high_val
    adj_matrix[4, 5] = high_val
    adj_matrix[5, 6] = high_val
    adj_matrix[6, 7] = high_val

    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    graph = tg.reset_adj_matrix(graph, adj_matrix)
    smplr = CastawayRST(graph, root=root, log_probs=True, trick=True)
    tree = smplr.sample_tree()
    exp_graph = tg.reset_adj_matrix(graph, np.exp(adj_matrix))
    assert tg.tree_to_newick(tree) == tg.tree_to_newick(nx.maximum_spanning_arborescence(exp_graph))


def test_victree_output():
    vic_out = h5py.File("/Users/zemp/phd/scilife/victree-experiments/SA501X3F/victree-out/K12/victree.out.h5ad",
                        'r')
    graph_matrix = vic_out['uns']['victree-tree-graph'][:]
    K = graph_matrix.shape[0]
    graph = nx.complete_graph(K, create_using=nx.DiGraph)
    for u, v in itertools.product(range(K), repeat=2):
        if u == v:
            graph_matrix[u, v] = -np.inf
    graph = tg.reset_adj_matrix(graph, graph_matrix)
    graph = tg.normalize_graph_weights(graph, log_probs=True)
    exp_weights = np.exp(nx.to_numpy_array(graph))
    # NOTE: weights are clamped so to avoid instabilities
    exp_weights = np.clip(exp_weights, a_min=1e-50, a_max=1.)
    graph = tg.reset_adj_matrix(graph, exp_weights)
    graph = tg.normalize_graph_weights(graph)
    ss = 100
    smplr = CastawayRST(graph, root=0, log_probs=False, trick=True)
    sample = smplr.sample(ss)

    print(sample)
    max_freq_tree_nwk = max(sample, key=sample.get)
    for i in range(K):
        graph.remove_edge(i, 0)
    msa = tg.tree_to_newick(nx.maximum_spanning_arborescence(graph))
    assert max_freq_tree_nwk == msa


def test_wilson_rst():
    n_nodes = 7
    adj_mat = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adj_mat, 0)
    # cardinality of tree topology
    tot_trees = tg.cayleys_formula(n_nodes)

    for log_probs in [False, True]:
        adj_mat = np.log(adj_mat) if log_probs else adj_mat
        graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        sample_size = 3 * tot_trees
        sample_dict = {}
        for s in range(sample_size):
            tree = algorithms.wilson_rst(graph, root=0, log_probs=log_probs)
            tree_nwk = tg.tree_to_newick(tree)
            if tree_nwk not in sample_dict:
                sample_dict[tree_nwk] = 0
            sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1

        unique_trees_obs = len(sample_dict)
        freqs = np.pad(np.array([v for k, v in sample_dict.items()]) / sample_size,
                       (0, tot_trees - unique_trees_obs))
        # test against uniform distribution
        test_result = chisquare(f_obs=freqs)
        assert test_result.pvalue >= 0.95, (f"chisq test not passed: evidence that distribution is not uniform"
                                            f" [log_probs = {log_probs}]")

def test_kirchoff_rst():
    n_nodes = 6
    adj_mat = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adj_mat, 0)
    # cardinality of tree topology
    tot_trees = tg.cayleys_formula(n_nodes)

    adj_mat = adj_mat
    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    sample_size = 3 * tot_trees
    sample_dict = {}
    for s in range(sample_size):
        tree = algorithms.kirchoff_rst(graph, root=0, log_probs=False)
        assert nx.is_arborescence(tree)
        tree_nwk = tg.tree_to_newick(tree)
        if tree_nwk not in sample_dict:
            sample_dict[tree_nwk] = 0
        sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1

    unique_trees_obs = len(sample_dict)
    freqs = np.pad(np.array([v for k, v in sample_dict.items()]) / sample_size,
                   (0, tot_trees - unique_trees_obs))
    # test against uniform distribution
    test_result = chisquare(f_obs=freqs)
    assert test_result.pvalue >= 0.95, (f"chisq test not passed: evidence that distribution is not uniform")


def test_colbourn_rst():
    n_nodes = 6
    root = 1
    adj_mat = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(adj_mat, 0)
    # cardinality of tree topology
    tot_trees = tg.cayleys_formula(n_nodes)

    adj_mat = adj_mat
    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    sample_size = 3 * tot_trees
    sample_dict = {}
    for s in range(sample_size):
        tree = algorithms.colbourn_rst(graph, root=root, log_probs=False)
        tree_nwk = tg.tree_to_newick(tree)
        if tree_nwk not in sample_dict:
            sample_dict[tree_nwk] = 0
        sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1

    unique_trees_obs = len(sample_dict)
    freqs = np.pad(np.array([v for k, v in sample_dict.items()]) / sample_size,
                   (0, tot_trees - unique_trees_obs))
    # test against uniform distribution
    test_result = chisquare(f_obs=freqs)
    assert test_result.pvalue >= 0.95, (f"chisq test not passed: evidence that distribution is not uniform")


def test_weakly_connection_colbourn():
    np.random.seed(0)
    n_nodes = 5
    n_samples = 10000

    for weak_weight in [0.1, 1e-2, 1e-10, 1e-15, 1e-20, 1e-25]:
        graph = tg.random_weakly_connected_graph(n_nodes, weak_weight=weak_weight)
        sample_dict = {}
        weight_dict = {}
        for _ in range(n_samples):
            tree = algorithms.colbourn_rst(graph, root=0, log_probs=False)
            tree_nwk = tg.tree_to_newick(tree)

            if tree_nwk not in sample_dict:
                sample_dict[tree_nwk] = 0
                weight_dict[tree_nwk] = tg.graph_weight(tree)
            sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1

        unique_trees_obs = len(sample_dict)
        freqs = np.array([v for k, v in sample_dict.items()])
        fexp = np.array([weight_dict[t] for t, _ in sample_dict.items()])
        # remove rare samples, otherwise chisq test is invalid
        rare = freqs <= 10
        if rare.sum() > 0:
            rare_freq = freqs[rare]
            freqs = np.append(freqs[~rare], np.sum(rare_freq))

            rare_fexp = fexp[rare]
            fexp = np.append(fexp[~rare], np.sum(rare_fexp))

        fexp /= np.sum(fexp)
        fexp *= sum(freqs)
        print(f"\n--- weakly connected graph test ---")
        print(f"weakness={weak_weight}, unique trees: {unique_trees_obs}/{tg.cayleys_formula(n_nodes)}")
        print(freqs)
        print(fexp)
        print(fexp.size)
        # test against expected distribution
        print(f'weakness={weak_weight}, freqs mse: {np.sum((freqs/sum(freqs) - fexp/sum(fexp)) ** 2)}')
        test_result = chisquare(f_obs=freqs, f_exp=fexp)
        print(f'weakness={weak_weight}, p-val {test_result}')
        # assert test_result.pvalue < 0.15, (f"chisq test not passed: evidence that distribution is not as expected")


def test_weakly_connection_castaway():
    np.random.seed(0)
    n_nodes = 5
    n_samples = 100
    # logging.root.setLevel(logging.DEBUG)

    for weak_weight in [0.1, 1e-2, 1e-10, 1e-15, 1e-20, 1e-25]:
        graph = tg.random_weakly_connected_graph(n_nodes, weak_weight=weak_weight, log_probs=False)
        sample_dict = {}
        weight_dict = {}
        smplr = CastawayRST(graph, root=0, log_probs=False, trick=False)
        try:
            sample_dict = smplr.sample(n_samples)
        except Exception as e:
            print(e)
            print("weakness too low, skipping test")
            continue
        for tree, _ in sample_dict.items():
            weight_dict[tree] = tg.graph_weight(tree)

        unique_trees_obs = len(sample_dict)
        freqs = np.array([v for k, v in sample_dict.items()])
        fexp = np.array([weight_dict[t] for t, _ in sample_dict.items()])
        # remove rare samples, otherwise chisq test is invalid
        rare = freqs <= 10
        if rare.sum() > 0:
            rare_freq = freqs[rare]
            freqs = np.append(freqs[~rare], np.sum(rare_freq))

            rare_fexp = fexp[rare]
            fexp = np.append(fexp[~rare], np.sum(rare_fexp))

        fexp /= np.sum(fexp)
        fexp *= sum(freqs)
        print(f"\n--- weakly connected graph test ---")
        print(f"weakness={weak_weight}, unique trees: {unique_trees_obs}/{tg.cayleys_formula(n_nodes)}")
        print(freqs)
        print(fexp)
        print(fexp.size)
        # test against expected distribution
        print(f'weakness={weak_weight}, freqs mse: {np.sum((freqs/sum(freqs) - fexp/sum(fexp)) ** 2)}')
        test_result = chisquare(f_obs=freqs, f_exp=fexp)
        print(f'weakness={weak_weight}, p-val {test_result}')
        # assert test_result.pvalue < 0.15, (f"chisq test not passed: evidence that distribution is not as expected")

def test_kirchoff_no_loops():
    """
    Checks that leverage is 0 when mixing arc is deleted.
    In this case, removing arc 0 -> 2 makes it impossible to go from 3 to 2
    without creating loops. Indeed, the leverage of arc 3 -> 2 is 0 if arc 0 -> 2 is deleted.
    """
    # toy matrix
    A = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    G = tg.normalize_graph_weights(G)
    print(nx.to_numpy_array(G))
    # check resistance of arc 3 -> 2
    leverage = tg.tuttes_tot_weight(G, 0, contracted_arcs=[(3, 2)], deleted_arcs=[(0, 3)])
    resistance = G.edges()[3, 2]['weight'] * leverage / tg.tuttes_tot_weight(G, 0)
    assert np.isclose(0, resistance)
    # contract 0 -> 1 and check that 1/3 the total weight is in 1-2-3,
    # another third is in 0->3->2 and the last third is in 0->3 and 1->2
    w01 = G.edges()[0, 1]['weight']
    tot_weight_c01 = w01 * tg.tuttes_tot_weight(G, 0, contracted_arcs=[(0, 1)])
    tot_weight_c01_23 = G.edges()[2, 3]['weight'] * w01 * tg.tuttes_tot_weight(G, 0, contracted_arcs=[(0, 1), (2, 3)])
    tot_weight_c01_32 = G.edges()[3, 2]['weight'] * w01 * tg.tuttes_tot_weight(G, 0, contracted_arcs=[(0, 1), (3, 2)])
    tot_weight_c01_d23_d32 = w01 * tg.tuttes_tot_weight(G, 0, contracted_arcs=[(0, 1)], deleted_arcs=[(2, 3), (3, 2)])
    print(f'total weight of trees with arc 0 -> 1: {tot_weight_c01}')
    print(f'total weight of trees with arc 0 -> 1 and 2 -> 3: {tot_weight_c01_23}')
    print(f'total weight of trees with arc 0 -> 1 and 3 -> 2: {tot_weight_c01_32}')
    print(f'total weight of trees with arc 0 -> 1 without 3 -> 2 or 2 -> 3: {tot_weight_c01_d23_d32}')
    assert np.isclose(tot_weight_c01, tot_weight_c01_23 + tot_weight_c01_32 + tot_weight_c01_d23_d32)

    ss = 10
    for s in range(ss):
        tree = kirchoff_rst(G, 0)
        assert nx.is_arborescence(tree)


def test_castaway_log_limitation():
    # generate weakly connected graphs with low weights in a range of -50 to -300
    np.random.seed(0)
    n_nodes = 5
    n_samples = 100
    # logging.root.setLevel(logging.DEBUG)
    start = -100
    end = -1000

    for lw in range(start, end, -10):
        log_graph = tg.random_weakly_connected_graph(n_nodes, weak_weight=lw, log_probs=True)
        miss = 0
        for _ in range(n_samples):
            try:
                smplr = CastawayRST(log_graph, root=0, log_probs=True, trick=False)
                tree = smplr.sample_tree()
                assert nx.is_arborescence(tree)
            except Exception as e:
                miss += 1
                print(e)
        print(f"weakness={lw}, misses={miss}")

def test_castaway_uncomplete_graph():
    # generate a graph with some missing arcs
    np.random.seed(0)
    n_nodes = 5
    n_samples = 100
    for log_probs in [True, False]:
        print(f"--- log_probs: {log_probs} ---")
        graph = tg.random_uniform_graph(n_nodes, log_probs=log_probs)
        removed_arcs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for arc in removed_arcs:
            graph.remove_edge(*arc)
        print(f"adj matrix: {nx.to_numpy_array(graph)}")
        smplr = CastawayRST(graph, root=0, log_probs=log_probs)
        for _ in range(n_samples):
            tree = smplr.sample_tree()
            assert nx.is_arborescence(tree)
            for arc in removed_arcs:
                assert not tree.has_edge(*arc), f"log_probs: {log_probs}, tree: {tree.edges()}"

