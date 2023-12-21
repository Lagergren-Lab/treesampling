import itertools
import random
import time
from operator import mul
from functools import reduce

import networkx as nx
import numpy as np


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


def jens_rst(in_graph: nx.DiGraph, root=0) -> nx.DiGraph:
    # normalize out arcs (rows)
    print("BEGIN ALGORITHM")
    graph = normalize_graph_weights(in_graph)
    tree = nx.DiGraph()
    tree.add_node(root)
    x_list = list(set(graph.nodes()).difference([root]))
    i_vertex = None
    o_vertex = None
    while tree.number_of_nodes() < in_graph.number_of_nodes() - 1:
        print(f"+ TREE NODES: {tree.nodes()}")
        print(f"\t x_list: {x_list}")
        # build wx with X = V \ {root}
        wx_table = _compute_wx_table(graph, x_list)
        # choose i vertex
        if i_vertex is None:
            i_vertex = random.choice(x_list)
            print(f"\t picked new i {i_vertex}")
            # pick arc (v, o) with v \in V(T) and o \in X
            pick_vo = []
            for o in x_list:
                for v in list(tree.nodes()):
                    pick_vo.append(((v, o), graph.edges()[v, o]['weight'] * wx_table[o, i_vertex]))
        else:
            # force previous o to be tree attachment
            print(f"\t continue previous i {i_vertex}")
            v = o_vertex
            pick_vo = []
            for o in x_list:
                pick_vo.append(((v, o), graph.edges()[v, o]['weight'] * wx_table[o, i_vertex]))

        edges, weights = zip(*pick_vo)
        # FIXME: i gets picked as o most of the times
        v, o_vertex = random.choices(edges, k=1, weights=weights)[0]
        print(f"\t selected v {v} from tree -> o {o_vertex} from x")
        # add o to tree, with arc from v (either newly selected tree node, or previous o)
        tree.add_edge(v, o_vertex)
        tree.edges()[v, o_vertex]['weight'] = graph.edges()[v, o_vertex]['weight']
        # remove o from X
        if o_vertex == i_vertex:
            i_vertex = None
        x_list = list(set(x_list).difference([o_vertex]))
    tree.add_edge(o_vertex, x_list[0])
    tree.edges()[o_vertex, x_list[0]]['weight'] = graph.edges[o_vertex, x_list[0]]['weight']

    return tree


def _compute_wx_table(graph: nx.DiGraph, x_set: list):
    # print(f"w(G): {nx.to_numpy_array(graph)}")
    # print(f"x_set: {list(graph.nodes())}")
    # base step: x_set = [u, v] (two nodes)
    u, v = x_set[:2]
    wx = {
        (u, v): graph.edges()[u, v]['weight'],
        (v, u): graph.edges()[v, u]['weight'],
        (v, v): 1.,
        (u, u): 1.
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
                if vv != v:
                    wxy[v] += graph.edges()[vv, u]['weight'] * wx[v, vv]
                    wyx[v] += wx[vv, v] * graph.edges()[u, vv]['weight']

        for v, w in wx.keys():
            if v == w:
                wy[v, w] = 1.
                continue

            # main update: either stay in X or pass through u (Y \ X)
            wy[v, w] = wx[v, w] + wxy[v] * ry * wyx[w]

            # special case: start or end in new vertex u
            wy[u, w] = ry * wyx[w]
            wy[v, u] = wxy[v] * ry

        wy[u, u] = 1.
        wx = wy

    return wx


def normalize_graph_weights(graph, log_probs=False) -> nx.DiGraph:
    adj_mat = nx.to_numpy_array(graph)
    if not log_probs:
        adj_mat = adj_mat / adj_mat.sum(axis=1, keepdims=True)
    else:
        adj_mat = adj_mat - np.logaddexp.reduce(adj_mat, axis=1, keepdims=True)
    norm_graph = reset_adj_matrix(graph, adj_mat)
    return norm_graph


def reset_adj_matrix(graph: nx.DiGraph, matrix: np.ndarray) -> nx.DiGraph:
    weights_dict = [(i, j, matrix[i, j]) for i, j in itertools.product(range(matrix.shape[0]), repeat=2)]
    new_graph = graph.copy()
    new_graph.add_weighted_edges_from(weights_dict)
    return new_graph


if __name__ == '__main__':
    n_nodes = 10
    sample_size = 100
    log_scale_weights = False  # change
    graph = random_graph(n_nodes)

    start = time.time()
    # networkx
    trees = [nx.random_spanning_tree(graph, weight='weight',
                                     multiplicative=not log_scale_weights) for _ in range(sample_size)]
    end = time.time() - start
    print(f"K = {n_nodes}: sampled {sample_size} trees in {end}s")
    for tree in trees:
        print(f"\t{[e for e in tree.edges()]}")

    # compute wx attempt
    norm_graph = normalize_graph_weights(graph)
    wx_table = _compute_wx_table(norm_graph, x_set=list(range(1, n_nodes)))
    print(wx_table)

    # try jens_rst
    trees_sample = {}
    for i in range(sample_size):
        tree = jens_rst(graph)
        if tree not in trees_sample:
            trees_sample[tree] = 1
        else:
            trees_sample[tree] += 1

    for t, prop in trees_sample.items():
        print(f"\t{prop / sample_size} : {reduce(mul, list(t.edges()[e]['weight'] for e in t.edges()), 1)} edges: {t.edges()}")
