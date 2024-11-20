from random import random

import networkx as nx
import numpy as np

from treesampling.algorithms import CastawayRST
from treesampling.algorithms.castaway import WxTable
from treesampling.utils.graphs import tree_to_newick, reset_adj_matrix, random_uniform_graph, normalize_graph_weights, \
    tuttes_tot_weight, graph_weight, enumerate_rooted_trees


def compute_lexit_probs(graph, wx: WxTable, i_vertex, tree_nodes):
    op = wx.op
    gw = nx.to_numpy_array(graph)
    nodes = []  # tuples (node, source) - source can be 'x' or 't'
    w_choice = []  # weights for random choice at each node

    # probability of any u in V(T) U X to be the next connection to i
    # attachment from tree to any node in X
    pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
    # O(n^2)
    for v in wx.x:
        pattach[v] = op.add([gw[w, v] for w in tree_nodes])
    # for tree nodes as origin, probability is barely the weight of the arc
    for u in tree_nodes:
        nodes.append((u, 't'))
        w_choice.append(gw[u, i_vertex])
    # for x nodes as origin, probability is the sum of all paths from any w in T to u (through any v in X)
    # O(n^2)
    for u in wx.x:
        # lexit_i(u) = (sum_{v in X} (sum_{w in T} Ww,v) * Wv,u ) * p(u, i)
        # any w, v (from tree to X) * any v, u (from X to i) * p(u, i)
        p_tree_u_i = op.mul(
            [op.add([op.mul([pattach[v], wx.wx_dict[v, u]]) for v in wx.x]), gw[u, i_vertex]])
        nodes.append((u, 'x'))
        w_choice.append(p_tree_u_i)
    w_choice = op.normalize(w_choice)
    return w_choice, nodes


def sample_tree_pair(graph: nx.DiGraph, root: int, n_nodes: int, seed: int | None = None) -> nx.DiGraph:
    if seed is None:
        seed = np.random.randint(0, 2 ** 32)
    graph_log = reset_adj_matrix(graph, np.log(nx.to_numpy_array(graph)))

    # graph_log = normalize_graph_weights(graph_log, log_probs=True)
    x = list(set(graph.nodes()).difference({root}))

    wxp = WxTable(x=x, graph=graph, log_probs=False)
    wxl = WxTable(x=x, graph=graph_log, log_probs=True)

    # initialize tree with root
    tp = nx.DiGraph()
    tp.add_node(root)

    # log-version tree
    tl = nx.DiGraph()
    tl.add_node(root)

    # dangling path plain and log
    dpp: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
    dpl: list[tuple] = []
    # iterate for each node
    assert set(x) == set(wxp.x) == set(wxl.x), f"nodes x: {x} != {wxp.x} != {wxl.x}"
    while len(wxp.x) > 0:
        assert set(wxp.x) == set(wxl.x), f"[{n_nodes - len(wxp.x)}] nodes x: {wxp.x} != {wxl.x}"
        # choose new i from x if no dangling nodes
        assert tuple(dpp) == tuple(dpl), f"dangling path {dpp} != {dpl}"
        if not dpp:
            i_vertex = wxp.x[0]
        else:
            latest_edge = dpp[-1]
            i_vertex = latest_edge[0]

        # update Wx table and remove x from x set
        wxp.update(i_vertex, trick=True)
        wxl.update(i_vertex, trick=True)
        assert np.allclose(wxp.to_log_array(), wxl.to_array()),\
            f"Wx tables differ: {wxp.to_log_array()} != {wxl.to_array()}"

        # compute lexit probs
        assert i_vertex not in wxp.x
        assert i_vertex not in wxl.x

        w_choice, nodes = compute_lexit_probs(graph, wxp, i_vertex, [n for n in tp.nodes()])
        w_choice_log, nodes_log = compute_lexit_probs(graph_log, wxl, i_vertex, [n for n in tl.nodes()])

        # pick u proportionally to lexit_i(u)
        assert np.allclose(w_choice, np.exp(w_choice_log)), f"weights: {w_choice} != {np.exp(w_choice_log)}"
        # control random seed
        np.random.seed(seed)
        u_idx = wxp.op.random_choice(w_choice) # random choice (if log_probs, uses gumbel trick)
        np.random.seed(seed)
        u_idx_log = wxp.op.random_choice(np.exp(w_choice_log))  # !!! this is the only difference
        assert u_idx == u_idx_log, f"u_idx: {u_idx} != {u_idx_log}"

        u_vertex, origin_lab = nodes[u_idx]

        dpp.append((u_vertex, i_vertex, graph.edges()[u_vertex, i_vertex]['weight']))
        dpl.append((nodes_log[u_idx_log][0], i_vertex, graph.edges()[nodes_log[u_idx_log][0], i_vertex]['weight']))
        if origin_lab == 't':
            tp.add_weighted_edges_from(dpp)
            tl.add_weighted_edges_from(dpl)
            dpp = []
            dpl = []

    # no more nodes in x set, the tree is complete
    return tp, tl

def run():
    root = 0
    n_nodes = 5
    n_trees = 10000
    graph = random_uniform_graph(n_nodes, normalize=True)
    print(nx.to_numpy_array(graph))
    counts = {}
    tree_weight = {}
    tot_weight = tuttes_tot_weight(graph, root)
    for i in range(n_trees):
        tree_plain, tree_log = sample_tree_pair(graph, root, n_nodes, seed=None)
        newick_plain = tree_to_newick(tree_plain)
        newick_log = tree_to_newick(tree_log)
        # print(f"Plain: {newick_plain}")
        # print(f"Log: {newick_log}")
        if newick_plain != newick_log:
            print("Trees are different!")
        else:
            counts[newick_plain] = counts.get(newick_plain, 0) + 1 / n_trees
            if newick_plain not in tree_weight:
                tree_weight[newick_plain] = graph_weight(tree_plain) / tot_weight

    # compute the actual weight for all possible trees
    all_trees = enumerate_rooted_trees(n_nodes, root, graph)
    set_all_trees = {tree_to_newick(t): graph_weight(t) / tot_weight for t in all_trees}
    assert np.isclose(sum([w for w in set_all_trees.values()]), 1.), "tot weight is not correct"

    # check that all frequencies in the sample are close to the true probability
    for tnwk, freq in counts.items():
        assert tnwk in set_all_trees, f"Tree {tnwk} not in set of all trees"
        set_all_trees.pop(tnwk)
        # assert np.isclose(freq, tree_weight[tnwk]), f"Tree {tnwk} has different weights ({freq} != {tree_weight[tnwk]})"
        print(f"Tree: {tnwk}, count: {freq}, weight: {tree_weight[tnwk]}")

    missing_trees = set_all_trees
    for tnwk in missing_trees:
        print(f"Missing tree: {tnwk} (weight: {set_all_trees[tnwk]})")

if __name__ == "__main__":
    run()
