import time
import random
import networkx as nx
import numpy as np
import csv

from treesampling.utils.math import logsubexp, gumbel_max_trick_sample
from treesampling.utils.graphs import tree_to_newick, graph_weight

from treesampling.utils.graphs import random_uniform_graph, normalize_graph_weights


def wilson_rst(graph) -> nx.DiGraph:
    # TODO: implement
    stoch_graph = normalize_graph_weights(graph)
    random_tree = nx.DiGraph()
    return random_tree


# IMPLEMENTATION IN NORMAL SCALE
def _compute_lexit_table(i, x_list: list,  wx_table: dict, tree: nx.DiGraph, graph: nx.DiGraph) -> tuple[list, list]:
    nodes = []  # tuples (node, source) - source can be 'x' or 't'
    w_choice = []  # weights for random choice at each node

    # probability of any u in V(T) U X to be the next connection to i
    pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
    for v in x_list:
        p_treetou = 0
        for w in tree.nodes():
            p_treetou = p_treetou + graph.edges()[w, v]['weight']
        pattach[v] = p_treetou

    for u in tree.nodes():
        nodes.append((u, 't'))
        w_choice.append(graph.edges()[u, i]['weight'])
    for u in x_list:
        p_treetou = 0
        for v in x_list:
            p_treetou = p_treetou + pattach[v] * wx_table[v, u]
        nodes.append((u, 'x'))
        w_choice.append(p_treetou * graph.edges()[u, i]['weight'])

    return nodes, w_choice  # (u, origin) list and choice weights list


def _update_wx(wy_table, u) -> dict:
    # speed up trick
    wx_table = {}
    for (v, w) in wy_table.keys():
        if v != u and w != u:
            wx_table[v, w] = wy_table[v, w] - wy_table[v, u] * wy_table[u, w] / wy_table[u, u]
    return wx_table


def random_spanning_tree(in_graph: nx.DiGraph, root, trick=True) -> nx.DiGraph:
    """
    Sample one tree from a given graph with fast arborescence sampling algorithm.
    :param in_graph: must have log-scale weights
    :param root: root node
    :param trick: if false, Wx gets re-computed every time
    :return: nx.DiGraph with tree edges only
    """
    # normalize out arcs (cols)
    # print("BEGIN ALGORITHM")
    graph = normalize_graph_weights(in_graph, rowwise=False, log_probs=False)

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
        # print(f"+ TREE: {tree_to_newick(tree)}")
        # print(f"\tX set: {x_list}")
        # print(f"\tdangling path: {dangling_path}")
        # choose new i if no dangling nodes
        if not dangling_path:
            # NOTE: stochastic vs sorted choice (should not matter)
            # i_vertex = random.choice(x_list)
            i_vertex = x_list[0]
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
        nodes_lab, w_choice = _compute_lexit_table(i_vertex, x_list, wx_table, tree, graph)

        # pick next node
        rnd_idx = random.choices(list(range(len(w_choice))), w_choice, k=1)
        u_vertex, origin_lab = nodes_lab[rnd_idx[0]]
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
    tree_nodes = list(tree.nodes())
    rnd_idx = random.choices(list(range(len(tree_nodes))), [graph.edges()[u, i_vertex]['weight'] for u in tree_nodes],
                             k=1)
    u_vertex = tree_nodes[rnd_idx[0]]

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
    # base step: x_set = [v] (one node)
    v = x_set[0]
    wx = {(v, v): 1}

    for i in range(1, len(x_set)):
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
            ry_1 = ry_1 + graph.edges()[u, v]['weight'] * wx[v, w] * graph.edges()[w, u]['weight']
        ry = 1 / (1 - ry_1)

        # compute Wy
        # partial computations: Wxy and Wyx
        wxy = {}  # Wxy (all paths from any v to new vertex u = Y \ X)
        wyx = {}  # Wxy (all paths from the new vertex u to any v in X)
        for v in x:
            wxy[v] = 0
            wyx[v] = 0
            for vv in x:
                wxy[v] = wxy[v] + graph.edges()[vv, u]['weight'] * wx[v, vv]
                wyx[v] = wyx[v] + wx[vv, v] * graph.edges()[u, vv]['weight']

        # write new W table
        for v in x:
            # special case: start or end in new vertex u
            wy[u, v] = ry * wyx[v]
            wy[v, u] = wxy[v] * ry
            for w in x:
                # main update: either stay in X or pass through u (Y \ X)
                wy[v, w] = wx[v, w] + wxy[v] * ry * wyx[w]
        # new self returning random path
        wy[u, u] = ry

        wx = wy

    return wx

# IMPLEMENTATION IN LOG SCALE
def _compute_lexit_table_log(i, x_list: list,  wx_table: dict, tree: nx.DiGraph, graph: nx.DiGraph) -> tuple[list, list]:
    nodes = []  # tuples (node, source) - source can be 'x' or 't'
    w_choice = []  # weights for random choice at each node

    # probability of any u in V(T) U X to be the next connection to i
    pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
    for v in x_list:
        p_treetou = - np.infty
        for w in tree.nodes():
            p_treetou = np.logaddexp(p_treetou, graph.edges()[w, v]['weight'])
        pattach[v] = p_treetou

    for u in tree.nodes():
        nodes.append((u, 't'))
        w_choice.append(graph.edges()[u, i]['weight'])
    for u in x_list:
        p_treetou = - np.infty
        for v in x_list:
            p_treetou = np.logaddexp(p_treetou, pattach[v] + wx_table[v, u])
        nodes.append((u, 'x'))
        w_choice.append(p_treetou + graph.edges()[u, i]['weight'])

    return nodes, w_choice  # (u, origin) list and choice weights list


def _update_wx_log(wy_table, u) -> dict:
    # speed up trick
    wx_table = {}
    if wy_table[u, u] == - np.inf:
        wx_table = wy_table
    else:
        for (v, w) in wy_table.keys():
            if v != u and w != u:
                if wy_table[v, w] == - np.inf:
                    # wx can't be any less than that (avoids logsubexp(-inf, a) where -np.inf < a << 0 )
                    a = - np.inf
                else:
                    a = logsubexp(wy_table[v, w], wy_table[v, u] + wy_table[u, w] - wy_table[u, u])
                wx_table[v, w] = a
    return wx_table


def random_spanning_tree_log(in_graph: nx.DiGraph, root, trick=True) -> nx.DiGraph:
    """
    Sample one tree from a given graph with fast arborescence sampling algorithm.
    :param in_graph: must have log-scale weights
    :param root: root node
    :param trick: if false, Wx gets re-computed every time
    :return: nx.DiGraph with tree edges only
    """
    # normalize out arcs (cols)
    # print("BEGIN ALGORITHM")
    # set non-existing edge weights to -inf
    missing_edges = nx.difference(nx.complete_graph(in_graph.number_of_nodes()), in_graph)
    in_graph.add_edges_from([(u, v, {'weight': -np.inf}) for u, v in missing_edges.edges()])
    graph = normalize_graph_weights(in_graph, rowwise=False, log_probs=True)

    # algorithm variables
    tree = nx.DiGraph()
    tree.add_node(root)
    dangling_path: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
    x_list = list(set(graph.nodes()).difference([root]))  # X set
    # precompute Wx table
    # build wx with X = V \ {root}
    wx_table = _compute_wx_table_log(graph, x_list)
    # iterate for each node
    while len(x_list) > 1:
        # print(f"+ TREE: {tree_to_newick(tree)}")
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
            wx_table = _update_wx_log(wx_table, i_vertex)
        else:
            wx_table = _compute_wx_table_log(graph, x_list)
        nodes_lab, w_choice = _compute_lexit_table_log(i_vertex, x_list, wx_table, tree, graph)

        # pick next node
        # --- original
        rnd_idx = gumbel_max_trick_sample(w_choice)

        # --- exponentiate version
        # w_choice_p = ss.softmax(np.array(w_choice))
        # rnd_idx = random.choices(list(range(len(w_choice))), w_choice_p, k=1)
        # rnd_idx = rnd_idx[0]
        # ---
        u_vertex, origin_lab = nodes_lab[rnd_idx]
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
    tree_nodes = list(tree.nodes())
    rnd_idx = gumbel_max_trick_sample([graph.edges()[u, i_vertex]['weight'] for u in tree_nodes])
    u_vertex = tree_nodes[rnd_idx]

    if dangling_path:
        dangling_path.append((u_vertex, i_vertex, graph.edges()[u_vertex, i_vertex]['weight']))

        # print(f"\t LAST TREE ATTACHMENT! selected u {u_vertex} from tree -> i {i_vertex}")
        # print(f"\t attached path: {dangling_path}")
        tree.add_weighted_edges_from(dangling_path)
    else:
        # print(f"\t LAST TREE ATTACHMENT! selected u {u_vertex} from tree -> i {i_vertex}")
        tree.add_edge(u_vertex, i_vertex, weight=graph.edges()[u_vertex, i_vertex]['weight'])

    return tree


def _compute_wx_table_log(graph: nx.DiGraph, x_set: list) -> dict:
    # print(f"w(G): {nx.to_numpy_array(graph)}")
    # print(f"x_set: {list(graph.nodes())}")
    # base step: x_set = [v] (one node)
    v = x_set[0]
    wx = {(v, v): 0}

    for i in range(1, len(x_set)):
        # print(f"current wx: {wx}")
        x = x_set[:i]
        # print(f"x: {x}")
        u = x_set[i]
        # print(f"new vertex u: {u}")
        # Y = X U { Vi }
        wy = {}
        # compute Ry(u) where u is Y \ X (u)
        ry_1 = - np.infty
        for (v, w) in wx.keys():
            # this might lead to ry_1 being slightly larger than 1,
            # but only ry_1 < 0 (strictly) is allowed
            ry_1 = np.logaddexp(ry_1, graph.edges()[u, v]['weight'] + wx[v, w] + graph.edges()[w, u]['weight'])
        # TODO: check if the following commented hack is necessary or not
        # ry_1 = np.clip(ry_1, a_min=None, a_max=-1e-10)
        # stable exponentiation: if ry_1 << 0 in log scale, then geometric series will be = 1 anyway
        ry = - np.log(1 - np.exp(ry_1))

        # compute Wy
        # partial computations: Wxy and Wyx
        wxy = {}  # Wxy (all paths from any v to new vertex u = Y \ X)
        wyx = {}  # Wxy (all paths from the new vertex u to any v in X)
        for v in x:
            wxy[v] = - np.infty
            wyx[v] = - np.infty
            for vv in x:
                wxy[v] = np.logaddexp(wxy[v], graph.edges()[vv, u]['weight'] + wx[v, vv])
                wyx[v] = np.logaddexp(wyx[v], wx[vv, v] + graph.edges()[u, vv]['weight'])

        # write new W table
        # FIXME: when ry is infty, then normalize all wx entries by removing ry
        #   i.e. set ry to zero and wx to zero (cause all new paths will pass through u
        for v in x:
            # special case: start or end in new vertex u
            wy[u, v] = ry + wyx[v]
            wy[v, u] = wxy[v] + ry
            for w in x:
                # main update: either stay in X or pass through u (Y \ X)
                wy[v, w] = np.logaddexp(wx[v, w], wxy[v] + ry + wyx[w])
        # new self returning random path
        wy[u, u] = ry

        wx = wy

    return wx


def kirchoff_rst(graph: nx.DiGraph, root=0, log_probs: bool = False):
    if log_probs:
        raise ValueError("Kirchoff RST not implemented for log-probabilities")
    # initialize empty digraph
    tree = nx.DiGraph()
    # for each edge, contract that edge and compute the acceptance probability
    # TODO: implement

    return tree

def wilson_rst(graph: nx.DiGraph, root=0, log_probs: bool = False):
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
        i = random.choice(list(x_set))
        u = i
        while u not in t_set:
            # loop-erased random walk
            if log_probs:
                prev[u] = gumbel_max_trick_sample(weights[:, u])
            else:
                prev[u] = random.choices(range(n_nodes), k=1, weights=weights[:, u])[0]
            u = prev[u]
        u = i
        while u not in t_set:
            # add to tree
            tree.add_edge(prev[u], u, weight=weights[prev[u], u])
            x_set.remove(u)
            t_set.add(u)
            u = prev[u]
    return tree


if __name__ == '__main__':
    # repeat for different number of nodes
    for n_nodes in [8, 9, 10]:
        trees_sample = {}
        graph = random_uniform_graph(n_nodes)
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
                weights[i, j] = np.random.random(1) * 1e-3
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
