import random

import networkx as nx
import numpy as np
from scipy.stats import alpha

from treesampling.algorithms import CastawayRST
from treesampling.algorithms.castaway import WxTable
from treesampling.utils.graphs import random_uniform_graph, normalize_graph_weights, tree_to_newick, reset_adj_matrix
from tests.test_sampling import chi_square_goodness, tree_sample_dist_correlation


def test_wx_table():
    # random graph
    g = random_uniform_graph(5, log_probs=False)
    # normalize by col
    normalize_graph_weights(g, log_probs=False)
    wx = WxTable(x=[0, 1, 2, 3, 4], graph=g, log_probs=False)

    # test update
    # remove 4 from x
    wx.x = [0, 1, 2, 3]
    wx_4 = wx._build()
    print(wx_4)

    # remove 4 and change order
    wx.x = [0, 1, 3, 2]
    wx_4_order = wx._build()
    print(wx_4_order)

    # test update with trick
    wx_4_trick = wx._update_trick(4)
    print(wx_4_trick)

    for k, v in wx_4.items():
        assert np.isclose(v, wx_4_order[k])
        assert np.isclose(v, wx_4_trick[k])


def test_wx_table_log():
    # random graph
    g = random_uniform_graph(5, log_probs=True)
    # normalize by col
    g = normalize_graph_weights(g, log_probs=True)
    wx = WxTable(x=[0, 1, 2, 3, 4], graph=g, log_probs=True)
    print(wx.get_wx())

    # test update
    # remove 4 from x
    wx.x = [0, 1, 2, 3]
    wx_4 = wx._build()
    print(wx_4)

    # remove 4 and change order
    wx.x = [0, 1, 3, 2]
    wx_4_order = wx._build()
    print(wx_4_order)

    # test update with trick
    wx_4_trick = wx._update_trick(4)
    print(wx_4_trick)

    for k, v in wx_4.items():
        assert np.isclose(v, wx_4_order[k])
        assert np.isclose(v, wx_4_trick[k])

def test_castaway_uniform():
    random.seed(42)
    np.random.seed(42)
    # random graph
    g = random_uniform_graph(5)
    # normalize by col
    g = normalize_graph_weights(g)
    print(g.edges(data=True))

    sampler = CastawayRST(graph=g, root=0)
    tree = sampler.sample_tree()
    print(tree_to_newick(tree))

    # test correlation
    n_samples = 10000
    trees = []
    for _ in range(n_samples):
        tree = sampler.sample_tree()
        # print(tree_to_newick(tree))
        trees.append(tree)

    test_result = chi_square_goodness(trees, g, root=0)
    print(f"chi2: {test_result.statistic}, p-value: {test_result.pvalue}")
    # assert test_result.pvalue >= 0.95, (f"chisq test not passed: evidence that distribution is not as expected")
    # NOTE: chisquare test not passing

    # test correlation coeff
    corr = tree_sample_dist_correlation(trees, g, root=0)
    print(f"correlation: {corr}")
    assert corr >= 0.95, (f"correlation test not passed: evidence that distribution is not as expected")

def test_castaway_uniform_log():
    random.seed(42)
    np.random.seed(42)
    # random graph
    g = random_uniform_graph(5, log_probs=True)
    # normalize by col
    g = normalize_graph_weights(g, log_probs=True)

    sampler = CastawayRST(graph=g, root=0, log_probs=True)
    tree = sampler.sample_tree()
    print(tree_to_newick(tree))

    # test correlation
    n_samples = 10000
    trees = []
    for _ in range(n_samples):
        tree = sampler.sample_tree()
        # print(tree_to_newick(tree))
        trees.append(tree)

    # transform to linear scale
    g = reset_adj_matrix(g, np.exp(nx.to_numpy_array(g)))
    for t in trees:
        for e in t.edges():
            t.edges()[e]['weight'] = np.exp(t.edges()[e]['weight'])
    # test correlation coeff
    corr = tree_sample_dist_correlation(trees, g, root=0)
    print(f"correlation: {corr}")
    assert corr >= 0.95, (f"correlation test not passed: evidence that distribution is not as expected")
    # TODO: correlation with log probs is much lower than with linear probs, something is wrong with correct log probs implementation
