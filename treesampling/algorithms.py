import time
import random
import networkx as nx
import numpy as np
import csv

from treesampling.utils.math import logsubexp, gumbel_max_trick_sample
from utils.graphs import tree_to_newick, graph_weight

from utils.graphs import random_uniform_graph, normalize_graph_weights


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


def _update_wx(wy_table, u) -> dict:
    # speed up trick
    wx_table = {}
    for (v, w) in wy_table.keys():
        if v != u and w != u:
            wx_table[v, w] = logsubexp(wy_table[v, w], wy_table[v, u] + wy_table[u, w] - wy_table[u, u])
    return wx_table


def jens_rst(in_graph: nx.DiGraph, root=0, trick=True) -> nx.DiGraph:
    # normalize out arcs (rows)
    # print("BEGIN ALGORITHM")
    graph = normalize_graph_weights(in_graph, log_probs=True)

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
        nodes_lab, w_choice = _compute_lexit_table(i_vertex, x_list, wx_table, tree, graph)

        # pick next node
        rnd_idx = gumbel_max_trick_sample(w_choice)
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


def _compute_wx_table(graph: nx.DiGraph, x_set: list) -> dict:
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
            ry_1 = np.logaddexp(ry_1, graph.edges()[u, v]['weight'] + wx[v, w] + graph.edges()[w, u]['weight'])
        ry = - logsubexp(0, ry_1)

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

        for v, w in wx.keys():
            # main update: either stay in X or pass through u (Y \ X)
            wy[v, w] = np.logaddexp(wx[v, w], wxy[v] + ry + wyx[w])

            # special case: start or end in new vertex u
            wy[u, w] = ry + wyx[w]
            wy[v, u] = wxy[v] + ry

            # new self returning random path
            wy[u, u] = ry

        wx = wy

    return wx


if __name__ == '__main__':
    log_scale_weights = True  # change
    results_csv_path = "../output/uniform_log_graph_corr_time.csv"
    # write header
    with open(results_csv_path, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['n_nodes', 'sample_size', 'time', 'correlation'])

    # repeat for different number of nodes
    for n_nodes in [5, 6, 7, 8, 9, 10]:
        trees_sample = {}
        graph = random_uniform_graph(n_nodes, log_scale_weights)
        times = []
        sample_sizes = [100, 500, 1000, 2000]
        results = []
        for i in range(len(sample_sizes)):
            prev_time = 0 if i == 0 else times[-1]
            prev_ss = 0 if i == 0 else sample_sizes[i-1]
            sample_size = sample_sizes[i]

            start = time.time()
            for i in range(sample_size - prev_ss):
                tree = jens_rst(graph)
                tree_newick = tree_to_newick(tree)
                if tree_newick not in trees_sample:
                    trees_sample[tree_newick] = (1, tree)
                else:
                    trees_sample[tree_newick] = (trees_sample[tree_newick][0] + 1, tree)
            end = time.time() - start
            times.append(end + prev_time)

            tree_freqs = []  # frequency of tree in a sample
            tree_weights = []  # weight of tree in the sample
            for t_nwk, (prop, t) in trees_sample.items():
                tree_freqs.append(prop / sample_size)
                tree_weights.append(np.exp(graph_weight(t, log_probs=log_scale_weights)))
                # print(f"\t{tree_freqs[-1]} : {tree_weights[-1]}"
                #       f" newick: {t_nwk}")
            correlation = np.corrcoef(tree_freqs, tree_weights)[0, 1]
            # print(f"Correlation coeff: {correlation}")
            # # print time
            # print(f"K = {n_nodes}: sampled {sample_size} trees in {times[-1]}s")
            results.append([n_nodes, sample_size, times[-1], correlation])
            print(f"{results[-1]}")
        with open(results_csv_path, 'a') as fd:
            writer = csv.writer(fd)
            for r in results:
                writer.writerow(r)
