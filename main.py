import itertools
import random
import time
from operator import mul
from functools import reduce

import networkx as nx
import numpy as np

from utils import tree_to_newick


def random_graph(n_nodes) -> nx.DiGraph:
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    for u, v in graph.edges():
        if u == v:
            w = 0
        else:
            w = random.random()
        graph.edges()[u, v]['weight'] = w
    return graph


def wilson_rst(graph) -> nx.DiGraph:
    # TODO: implement
    stoch_graph = normalize_graph_weights(graph)
    random_tree = nx.DiGraph()
    return random_tree


def _compute_lexit_table(i, x_list: list,  wx_table: dict, tree: nx.DiGraph, graph: nx.DiGraph) -> tuple[list, list]:
    nodes = []  # tuples (node, source) - source can be 'x' or 't'
    w_choice = []  # weights for random choice at each node

    # probability of any u in V(T) U X to be the next connection to i
    pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
    for v in x_list:
        p_treetou = 0
        for w in tree.nodes():
            p_treetou += graph.edges()[w, v]['weight']
        pattach[v] = p_treetou

    for u in tree.nodes():
        nodes.append((u, 't'))
        w_choice.append(graph.edges()[u, i]['weight'])
    for u in x_list:
        p_treetou = 0
        for v in x_list:
            p_treetou += pattach[v] * wx_table[v, u]
        nodes.append((u, 'x'))
        w_choice.append(p_treetou * graph.edges()[u, i]['weight'])

    return nodes, w_choice  # (u, origin) list and choice weights list


def _update_wx(wy_table, u) -> dict:
    # speed up trick
    wx_table = {}
    for (v, w) in wy_table.keys():
        if v != u and w != u:
            wx_table[v, w] = wy_table[v, w] - wy_table[v, u] * wy_table[u, w] / wy_table[u,u]
    return wx_table


def jens_rst(in_graph: nx.DiGraph, root=0, trick=True) -> nx.DiGraph:
    # normalize out arcs (rows)
    # print("BEGIN ALGORITHM")
    graph = normalize_graph_weights(in_graph)

    # algorithm variables
    tree = nx.DiGraph()
    tree.add_node(root)
    dangling_path: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
    x_list = list(set(graph.nodes()).difference([root]))  # X set
    # precompute Wx table
    # build wx with X = V \ {root}
    wx_table = _compute_wx_table(graph, x_list)
    # iterate for each node
    while len(x_list) > 1:
        # print(f"+ TREE NODES: {tree.nodes()}")
        # print(f"\tX set: {x_list}")
        # print(f"\tdangling path: {dangling_path}")
        # choose new i if no dangling nodes
        if not dangling_path:
            i_vertex = random.choice(x_list)
            x_list.remove(i_vertex)
        # or set last node
        else:
            last_edge = dangling_path[-1]
            i_vertex = last_edge[0]
            x_list.remove(i_vertex)

        # update Wx table
        if trick:
            wx_table = _update_wx(wx_table, i_vertex)
        else:
            wx_table = _compute_wx_table(graph, x_list)
        nodes, w_choice = _compute_lexit_table(i_vertex, x_list, wx_table, tree, graph)

        # pick next node
        u_vertex, origin_lab = random.choices(nodes, k=1, weights=w_choice)[0]
        dangling_path.append((u_vertex, i_vertex, graph.edges()[u_vertex, i_vertex]['weight']))
        if origin_lab == 't':
            # if u picked from tree, attach dangling path and reset
            # print(f"\t TREE ATTACHMENT! selected u {u_vertex} from tree -> i {i_vertex}")
            # print(f"\t attached path: {dangling_path}")
            # add dangling path edges to tree
            tree.add_weighted_edges_from(dangling_path)
            dangling_path = []

    assert len(x_list) == 1
    i_vertex = x_list[0]
    # print(tree.nodes())
    u_vertex = random.choices(list(tree.nodes()), k=1,
                              weights=[graph.edges()[u, i_vertex]['weight'] for u in tree.nodes()])[0]
    if dangling_path:
        dangling_path.append((u_vertex, i_vertex, graph.edges()[u_vertex, i_vertex]['weight']))

        # print(f"\t LAST TREE ATTACHMENT! selected u {u_vertex} from tree -> i {i_vertex}")
        # print(f"\t attached path: {dangling_path}")
        tree.add_weighted_edges_from(dangling_path)
    else:
        # print(f"\t LAST TREE ATTACHMENT! selected u {u_vertex} from tree -> i {i_vertex}")
        tree.add_edge(u_vertex, i_vertex, weight=graph.edges()[u_vertex, i_vertex]['weight'])

    return tree


def _compute_wx_table(graph: nx.DiGraph, x_set: list) -> dict:
    # print(f"w(G): {nx.to_numpy_array(graph)}")
    # print(f"x_set: {list(graph.nodes())}")
    # trivial case when X set has only one node
    if len(x_set) == 1:
        u = x_set[0]
        wx = {(u, u): 1}
        return wx

    # base step: x_set = [u, v] (two nodes)
    u, v = x_set[:2]
    wx = {
        (u, v): graph.edges()[u, v]['weight'],
        (v, u): graph.edges()[v, u]['weight'],
        (v, v): graph.edges()[v, u]['weight'] * graph.edges()[u, v]['weight'],
        (u, u): graph.edges()[u, v]['weight'] * graph.edges()[v, u]['weight'],
    }
    for i in range(2, len(x_set)):
        # print(f"current wx: {wx}")
        x = x_set[:i]
        # print(f"x: {x}")
        u = x_set[i]
        # print(f"new vertex u: {u}")
        # Y = X U { Vi }
        wy = {}
        # compute Ry(u) where u is Y \ X (u)
        ry_1 = 0
        for (v, w) in wx.keys():
            ry_1 += graph.edges()[u, v]['weight'] * wx[v, w] * graph.edges()[w, u]['weight']
        ry = 1 / (1 - ry_1)

        # compute Wy
        # partial computations: Wxy and Wyx
        wxy = {}  # Wxy (all paths from any v to new vertex u = Y \ X)
        wyx = {}  # Wxy (all paths from the new vertex u to any v in X)
        for v in x:
            wxy[v] = 0
            wyx[v] = 0
            for vv in x:
                wxy[v] += graph.edges()[vv, u]['weight'] * wx[v, vv]
                wyx[v] += wx[vv, v] * graph.edges()[u, vv]['weight']

        for v, w in wx.keys():
            # main update: either stay in X or pass through u (Y \ X)
            wy[v, w] = wx[v, w] + wxy[v] * ry * wyx[w]

            # special case: start or end in new vertex u
            wy[u, w] = ry * wyx[w]
            wy[v, u] = wxy[v] * ry

            # new self returning random path
            wy[u, u] = ry

        wx = wy

    return wx


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


if __name__ == '__main__':
    n_nodes = 5
    sample_size = 5000
    log_scale_weights = False  # change
    graph = random_graph(n_nodes)

    # start = time.time()
    # # networkx
    # trees = [nx.random_spanning_tree(graph, weight='weight',
    #                                  multiplicative=not log_scale_weights) for _ in range(sample_size)]
    # end = time.time() - start
    # print(f"K = {n_nodes}: sampled {sample_size} trees in {end}s")
    # for tree in trees:
    #     print(f"\t{[e for e in tree.edges()]}")
    #

    # try jens_rst
    start = time.time()
    trees_sample = {}
    for i in range(sample_size):
        tree = jens_rst(graph)
        tree_newick = tree_to_newick(tree)
        if tree_newick not in trees_sample:
            trees_sample[tree_newick] = (1, tree)
        else:
            trees_sample[tree_newick] = (trees_sample[tree_newick][0] + 1, tree)
    end = time.time() - start

    for t_nwk, (prop, t) in trees_sample.items():
        print(f"\t{prop / sample_size} : {reduce(mul, list(t.edges()[e]['weight'] for e in t.edges()), 1)}"
              f" newick: {t_nwk}")
    # print time
    print(f"K = {n_nodes}: sampled {sample_size} trees in {end}s")
    # TODO: save big results, print correlation and assess correctness (some proportions don't agree)
