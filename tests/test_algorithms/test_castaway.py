import networkx as nx
import numpy as np
import scipy.special as sp

from treesampling.algorithms import CastawayRST
from treesampling.algorithms.castaway import WxTable
import treesampling.utils.graphs as tg
from tests.test_sampling import chi_square_goodness, tree_sample_dist_correlation
from treesampling.utils.graphs import normalize_graph_weights, random_tree_skewed_graph


def test_wx_table():
    """
    Check that wx table is independent of the order of the nodes in x and that
    the trick outputs the same W table as the computation from scratch
    """
    # random graph
    g = tg.random_uniform_graph(5, log_probs=False, normalize=True)
    wx = WxTable(x=[1, 2, 3, 4], graph=g, log_probs=False)
    print(wx.to_array())

    # test update
    # remove 4 from x
    wx.x = [1, 2, 3]
    wx_4 = wx._build()
    print(wx_4)

    # remove 4 and change order
    wx.x = [1, 3, 2]
    wx_4_order = wx._build()
    print(wx_4_order)

    # test update with trick
    wx_4_trick = wx._update_trick(4)
    print(wx_4_trick)

    for k, v in wx_4.items():
        assert np.isclose(v, wx_4_order[k])
        assert np.isclose(v, wx_4_trick[k])


def test_wx_table_log():
    """
    LOG version of previous test:
    Check that wx table is independent of the order of the nodes in x and that
    the trick outputs the same W table as the computation from scratch
    """
    # random graph
    g = tg.random_uniform_graph(5, log_probs=True, normalize=True)
    wx = WxTable(x=[1, 2, 3, 4], graph=g, log_probs=True)
    print(wx.to_array())

    # test update
    # remove 4 from x
    wx.x = [1, 2, 3]
    wx_4 = wx._build()
    print(wx_4)

    # remove 4 and change order
    wx.x = [1, 3, 2]
    wx_4_order = wx._build()
    print(wx_4_order)

    # test update with trick
    wx_4_trick = wx._update_trick(4)
    print(wx_4_trick)

    for k, v in wx_4.items():
        assert np.isclose(v, wx_4_order[k])
        assert np.isclose(v, wx_4_trick[k])

def test_wx_table_log_accuracy():
    """
    Check that the Wx table is computed the same way in log scale and in linear scale
    """
    np.random.seed(42)
    # compute wx table in log scale and in linear scale
    # compare the results
    g = tg.random_uniform_graph(5, log_probs=False, normalize=True)
    # zero is not included in x as it's the root node
    wx = WxTable(x=[1, 2, 3, 4], graph=g, log_probs=False)
    print("g matrix")
    print(nx.to_numpy_array(g))

    g_log = tg.reset_adj_matrix(g, np.log(nx.to_numpy_array(g)))
    # g_log = normalize_graph_weights(g_log, log_probs=True)
    # normalize by col
    marginal = np.logaddexp.reduce(nx.to_numpy_array(g_log), axis=0)
    assert np.allclose(marginal, np.zeros(5))
    print("g_log matrix")
    print(nx.to_numpy_array(g_log))
    wx_log = WxTable(x=[1, 2, 3, 4], graph=g_log, log_probs=True)

    # test that tables are equal both when computed in log scale and in linear scale
    # mat minor excludes the root node row/col from the table
    wx_arr = tg.mat_minor(wx.to_array(), 0, 0)
    wx_log_arr = tg.mat_minor(wx_log.to_array(), 0, 0)
    assert np.allclose(np.log(wx_arr), wx_log_arr)
    # print("log(wx_arr)")
    # print(np.log(wx_arr))
    # print("wx_log_arr")
    # print(wx_log_arr)

    assert np.allclose(np.exp(wx_log_arr), wx_arr)
    # print("wx_arr")
    # print(wx_arr)
    # print("exp(wx_log_arr)")
    # print(np.exp(wx_log_arr))

def test_wx_table_log_accuracy_update():
    """
    Check that the trick update output the same W table both in log and linear scale
    """
    np.random.seed(42)
    # compute wx table in log scale and in linear scale
    # compare the results
    g = tg.random_uniform_graph(5, log_probs=False, normalize=True)
    # zero is not included in x as it's the root node

    wx = WxTable(x=[1, 2, 3, 4], graph=g, log_probs=False)

    g_log = tg.reset_adj_matrix(g, np.log(nx.to_numpy_array(g)))
    wx_log = WxTable(x=[1, 2, 3, 4], graph=g_log, log_probs=True)

    wx.update(1, trick=True)
    wx_log.update(1, trick=True)
    # remove both row/col 1 and 0
    wx_arr = tg.mat_minor(tg.mat_minor(wx.to_array(), 1, 1), 0, 0)
    wx_log_arr = tg.mat_minor(tg.mat_minor(wx_log.to_array(), 1, 1), 0, 0)
    assert np.allclose(np.log(wx_arr), wx_log_arr)
    assert np.allclose(np.exp(wx_log_arr), wx_arr)

def test_wx_table_forced_edges():
    """
    Test that the Wx table is computed correctly when there are forced edges.
    """
    np.random.seed(42)
    n_nodes = 5
    g = tg.random_uniform_graph(n_nodes, log_probs=True, normalize=True)
    # force edges from 0 to 1
    g.edges[(0, 1)]['weight'] = 0.0
    for i in range(2, n_nodes):
        g.edges[(i, 1)]['weight'] = -np.inf
    print(nx.to_numpy_array(g))

    wx = WxTable(x=[1, 2, 3, 4], graph=g, log_probs=True)
    print(wx.to_array())
    wx.update(1, trick=True)
    print(wx.to_array())

def test_wx_table_weakly_connected_subgraphs():
    """
    Test that the Wx table is computed correctly when the graph has weakly connected subgraphs.
    """
    np.random.seed(42)
    n_nodes = 6
    g = tg.random_weakly_connected_k_subgraphs(n_nodes, k=2, weak_weight=-16, log_probs=True, normalized=True)
    print(nx.to_numpy_array(g))
    wx = WxTable(x=[1, 2, 3, 4, 5], graph=g, log_probs=True)
    print(wx.to_array())
    wx.update(1, trick=True)
    print(wx.to_array())
    wx.update(3, trick=True)
    print(wx.to_array())


def test_castaway_random_uniform_counts():
    np.random.seed(42)
    n_nodes = 5
    # random graph
    g = tg.random_uniform_graph(n_nodes, normalize=True)
    print(nx.to_numpy_array(g))
    print(g.edges(data=True))

    sampler = CastawayRST(graph=g, root=0, trick=True)
    # tutte's weight for actual prob computation
    tot_weight = tg.tuttes_tot_weight(g, root=0)

    # sample trees
    n_samples = 10000
    counts = {}
    tree_weight = {}
    for _ in range(n_samples):
        tree = sampler.sample_tree()
        tnwk = tg.tree_to_newick(tree)  # into dict for counting
        counts[tnwk] = counts.get(tnwk, 0) + 1 / n_samples
        if tnwk not in tree_weight:
            tree_weight[tnwk] = tg.graph_weight(tree) / tot_weight

    # check missing trees
    all_trees = tg.enumerate_rooted_trees(n_nodes, 0, g)
    set_all_trees = {tg.tree_to_newick(t): tg.graph_weight(t) / tot_weight for t in all_trees}
    assert np.isclose(sum([w for w in set_all_trees.values()]), 1.), "tot weight is not correct"

    # check that all frequencies in the sample are close to the true probability
    for tnwk, freq in counts.items():
        assert tnwk in set_all_trees, f"Tree {tnwk} not in set of all trees"
        set_all_trees.pop(tnwk)
        # assert np.isclose(freq, tree_weight[tnwk]), f"Tree {tnwk} has different weights ({freq} != {tree_weight[tnwk]})"
        print(f"Tree: {tnwk}, count: {freq}, weight: {tree_weight[tnwk]}")

    missing_trees = set_all_trees
    for tnwk in missing_trees:
        print(f"Missing tree: {tnwk} (weight: {set_all_trees[tnwk]}")


def test_castaway_uniform():
    """
    Random graph test
    """
    np.random.seed(42)
    n_nodes = 5
    # random graph
    g = tg.random_uniform_graph(n_nodes, normalize=True)
    print(nx.to_numpy_array(g))
    print(g.edges(data=True))

    sampler = CastawayRST(graph=g, root=0)
    tree = sampler.sample_tree()
    print(tg.tree_to_newick(tree))

    # sample trees
    n_samples = 10000
    trees = []
    for _ in range(n_samples):
        tree = sampler.sample_tree()
        # print(tg.tree_to_newick(tree))
        trees.append(tree)  # into list for chi2 test
    # chi square test
    test_result = chi_square_goodness(trees, g, root=0)
    print(f"chi2: {test_result.statistic}, p-value: {test_result.pvalue}")
    assert test_result.pvalue >= 0.05, f"chisq test not passed: evidence that distribution is not as expected"

    # test correlation coeff
    corr = tree_sample_dist_correlation(trees, g, root=0)
    print(f"correlation: {corr}")
    assert corr >= 0.90, f"correlation test not passed: evidence that distribution is not as expected"

def test_castaway_uniform_log():
    np.random.seed(42)
    n_nodes = 5
    # random graph
    g = tg.random_uniform_graph(n_nodes, log_probs=True, normalize=True)
    print(g.edges(data=True))
    print(np.exp(nx.to_numpy_array(g)))

    sampler = CastawayRST(graph=g, root=0, log_probs=True)
    tree = sampler.sample_tree()
    print(tg.tree_to_newick(tree))

    # test correlation
    n_samples = 5000
    trees = []
    for _ in range(n_samples):
        tree = sampler.sample_tree()
        # print(tg.tree_to_newick(tree))
        trees.append(tree)

    # transform to linear scale
    g = tg.reset_adj_matrix(g, np.exp(nx.to_numpy_array(g)))
    for t in trees:
        for e in t.edges():
            t.edges()[e]['weight'] = np.exp(t.edges()[e]['weight'])
    # check normalization
    assert np.allclose(nx.to_numpy_array(g).sum(axis=0), np.ones(n_nodes))

    # test correlation coeff
    corr = tree_sample_dist_correlation(trees, g, root=0)
    print(f"correlation: {corr}")
    assert corr >= 0.90, f"correlation test not passed: evidence that distribution is not as expected"

    # test chisquare
    test_result = chi_square_goodness(trees, g, root=0)
    print(f"chi2: {test_result.statistic}, p-value: {test_result.pvalue}")
    assert test_result.pvalue > 0.05, f"chisq test not passed: evidence that distribution is not as expected"
    # FIXME: correlation test passes slightly (as opposed to the linear scale)
    #   and chisquare fails completely

def test_lexit_probs():
    """
    Check that the probabilities are computed in the same way in log scale and in linear scale
    """
    np.random.seed(42)
    # random graph
    g = tg.random_uniform_graph(6, log_probs=False, normalize=True)
    sampler = CastawayRST(graph=g, root=0, log_probs=False)
    log_g = tg.reset_adj_matrix(g, np.log(nx.to_numpy_array(g)))
    log_sampler = CastawayRST(graph=log_g, root=0, log_probs=True)

    # update wx table to have x = [1, 2, 4] and tree_nodes = [0, 3]
    tree_nodes = [0, 3]
    sampler.wx.update(3, trick=True)
    sampler.wx.update(5, trick=True)
    log_sampler.wx.update(3, trick=True)
    log_sampler.wx.update(5, trick=True)
    assert sampler.wx.x == log_sampler.wx.x == [1, 2, 4]

    lexit, _ = sampler._compute_lexit_probs(5, tree_nodes)
    log_lexit, _ = log_sampler._compute_lexit_probs(5, tree_nodes)

    print(f"l_exit: {lexit}")
    print(f"log l_exit: {log_lexit}")
    assert np.allclose(np.exp(log_lexit), lexit)
    assert np.allclose(np.log(lexit), log_lexit)

def test_lexit_forced_edges():
    """
    Test that the exit probabilities are computed correctly when there are forced edges.
    """
    np.random.seed(42)
    n_nodes = 6
    g = tg.random_uniform_graph(n_nodes, log_probs=False, normalize=True)
    # force edge from 0 to 1
    g.edges[(0, 1)]['weight'] = 1.0
    for i in range(2, n_nodes):
        g.edges[(i, 1)]['weight'] = 0.
    print(nx.to_numpy_array(g))
    sampler = CastawayRST(graph=g, root=0, log_probs=False)
    log_g = tg.reset_adj_matrix(g, np.log(nx.to_numpy_array(g)))
    log_sampler = CastawayRST(graph=log_g, root=0, log_probs=True)

    # update wx table to have x = [1, 2, 4] and tree_nodes = [0, 3]
    tree_nodes = [0, 3]
    sampler.wx.update(3, trick=True)
    sampler.wx.update(5, trick=True)
    log_sampler.wx.update(3, trick=True)
    log_sampler.wx.update(5, trick=True)
    assert sampler.wx.x == log_sampler.wx.x == [1, 2, 4]

    lexit, _ = sampler._compute_lexit_probs(5, tree_nodes)
    log_lexit, _ = log_sampler._compute_lexit_probs(5, tree_nodes)

    print(f"l_exit: {lexit}")
    print(f"log l_exit: {log_lexit}")
    assert np.allclose(np.exp(log_lexit), lexit)
    assert np.allclose(np.log(lexit), log_lexit)

def test_lexit_weakly_connected_subgraphs_log():
    np.random.seed(42)
    log_probs = True
    n_nodes = 6
    g = tg.random_weakly_connected_k_subgraphs(n_nodes, k=2, weak_weight=-3, log_probs=log_probs, normalized=True)
    print(nx.to_numpy_array(g))
    sampler = CastawayRST(graph=g, root=0, log_probs=log_probs)

    # update wx table to have x = [1, 2, 4] and tree_nodes = [0, 3]
    tree_nodes = [0, 3]
    sampler.wx.update(3, trick=True)
    sampler.wx.update(5, trick=True)
    assert sampler.wx.x == [1, 2, 4]

    lexit, _ = sampler._compute_lexit_probs(5, tree_nodes)

    print(f"l_exit: {lexit}")
    assert not np.any(np.isnan(lexit)), "exit probs must be finite"



def test_castaway_log_low_weight():
    """
    This test was built to debug numerical instability in the castaway algorithm. Log-prob version.
    """
    np.random.seed(0)
    n_nodes = 6
    # random_graph = tg.random_weakly_connected_graph(n_nodes, weak_weight=low_weight, log_probs=True)
    random_graph = tg.random_weakly_connected_k_subgraphs(n_nodes, k=2, weak_weight=-3, log_probs=True)
    print(nx.to_numpy_array(random_graph))
    # sort edges by weight in increasing order without including self connections
    # in order to check which weak connections are most likely to be sampled
    edges = sorted(random_graph.edges(data=True), key=lambda e: e[2]['weight'])
    for e in edges:
        if e[0] != e[1]:
            print(f"{e[0]} -> {e[1]}: {e[2]['weight']}")

    # generate a sample and save both the tree weights and the frequencies at which they occur
    sampler = CastawayRST(random_graph, root=0, log_probs=True, trick=False, debug=True)
    matrix = nx.to_numpy_array(sampler.graph)
    assert np.all(np.isclose(sp.logsumexp(matrix, axis=0), 0.)), "not normalized"
    # FIXME: graph presents 0.0 weights between nodes in the same component [0, 1]
    # if any edge is 0, assert that the other incoming edges are -inf
    # if np.any(np.isclose(matrix, 0.)):
    #     for i in range(n_nodes):
    #         if np.any(np.isclose(matrix[:, i], 0.)):
    #             null_src = np.ones(n_nodes, dtype=bool)
    #             null_src[np.argwhere(np.isclose(matrix[:, i], 0.)).flatten()] = False
    #             assert np.all(np.isclose(matrix[null_src, i], -np.inf)), "null source edges must be -inf"
    # else:
    assert (np.all(nx.to_numpy_array(sampler.graph)[~np.eye(n_nodes, dtype=bool)] > -np.inf),
            "no self connections must be finite")

    assert np.all(nx.to_numpy_array(sampler.graph)[np.eye(n_nodes, dtype=bool)] == -np.inf), "self connections must be -inf"
    print(f"graph after castaway adjust: {nx.to_numpy_array(sampler.graph)}")
    n = 1000
    trees_dict = sampler.sample(n)
    print(sorted(trees_dict.items(), key=lambda x: x[1], reverse=True))

    # convert to linear scale
    lin_random_graph = tg.reset_adj_matrix(random_graph, np.exp(nx.to_numpy_array(random_graph)))
    assert np.all(np.diag(nx.to_numpy_array(lin_random_graph)) == 0)

def test_caching():
    # test caching
    # cache size should constrain cache_tables dict size
    n_nodes = 15
    cache_size = 10
    graph = random_tree_skewed_graph(n_nodes, 5, root=0)[0]
    sampler = CastawayRST(graph, root=0, log_probs=True, trick=True, debug=True, cache_size=10)
    n = 100
    cache_hits = {}
    for _ in range(n):
        sampler.sample_tree()

        # check that the hit counts doesn't reset (monotonic increase)
        for k, v in sampler.wx._cache_hits.items():
            if k in cache_hits:
                assert v >= cache_hits[k]
            cache_hits[k] = v

        # check that tables are also in the "hits" dict
        for k in sampler.wx._cached_tables:
            assert k in cache_hits

        assert len(sampler.wx._cached_tables) <= cache_size
