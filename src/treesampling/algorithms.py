import time
import logging
import networkx as nx
import numpy as np

from treesampling.utils.math import logsubexp, gumbel_max_trick_sample
from treesampling.utils.graphs import graph_weight, tuttes_tot_weight, reset_adj_matrix

from treesampling.utils.graphs import random_uniform_graph, normalize_graph_weights
from warnings import warn


def random_spanning_tree(graph: nx.DiGraph, root=0) -> nx.DiGraph:
    warn('Use the new function ' + castaway_rst.__name__ + ' with log_probs=False', DeprecationWarning, stacklevel=2)
    return castaway_rst(graph, root, log_probs=False)


def random_spanning_tree_log(graph: nx.DiGraph, root=0) -> nx.DiGraph:
    warn('Use the new function ' + castaway_rst.__name__ + ' with log_probs=True', DeprecationWarning, stacklevel=2)
    return castaway_rst(graph, root, log_probs=True)


def castaway_rst(graph: nx.DiGraph, root=0, log_probs: bool = False, trick: bool = True) -> nx.DiGraph:
    """
    Wrapper for the original random spanning tree sampler inspired by Wilson algorithm.
    :param graph: nx.DiGraph, with weights on arcs
    :param root: label of the root in the graph
    :param log_probs: if the graph has log-weights
    :param trick: if True, the algorithm runs in O(n^3) by efficiently updating the W table. Otherwise W
        is computed from scratch every time a new arc is added to the tree
    :return:
    """

    if log_probs:
        return _castaway_rst_log(graph, root, trick)
    else:
        return _castaway_rst_plain(graph, root, trick)


def _castaway_rst_plain(in_graph: nx.DiGraph, root, trick=True) -> nx.DiGraph:
    """
    Sample one tree from a given graph with fast arborescence sampling algorithm.
    :param in_graph: must have log-scale weights
    :param root: root node
    :param trick: if false, Wx gets re-computed every time
    :return: nx.DiGraph with tree edges only
    """
    # normalize out arcs (cols)
    logging.debug("\n[[[ BEGIN ALGORITHM ]]]\n")
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
        logging.debug(f"\n[[[ CASTAWAY IT {graph.number_of_nodes() - len(x_list)}]]]\n")
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
        assert not np.any([np.isnan(wx_table[k]) for k in wx_table]), f"NaN values in wx table: {wx_table} at node {i_vertex}"
        nodes_lab, w_choice = _compute_lexit_table(i_vertex, x_list, wx_table, tree, graph)

        # pick next node
        rnd_idx = np.random.choice(np.arange(len(w_choice)), size=1, p=w_choice)[0]
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
    pchoice = np.array([graph.edges()[u, i_vertex]['weight'] for u in tree_nodes])
    rnd_idx = np.random.choice(np.arange(len(tree_nodes)), p=pchoice / np.sum(pchoice),
                               size=1)[0]
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


def _compute_lexit_table(i, x_list: list,  wx_table: dict,
                         tree: nx.DiGraph, graph: nx.DiGraph) -> tuple[list, np.ndarray]:
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
    w_choice = np.array(w_choice) / np.sum(w_choice)

    return nodes, w_choice  # (u, origin) list and choice weights list


def _update_wx(wy_table, u) -> dict:
    # speed up trick
    logging.debug(f"\n--- UPDATE WX node: {u} ---\n")
    wx_table = {}
    for (v, w) in wy_table.keys():
        if v != u and w != u:
            wx_table[v, w] = wy_table[v, w] - wy_table[v, u] * wy_table[u, w] / wy_table[u, u]
    logging.debug(f"wx_table: {wx_table}")
    return wx_table


def _compute_wx_table(graph: nx.DiGraph, x_set: list) -> dict:
    logging.debug(f"w(G): {nx.to_numpy_array(graph)}")
    logging.debug(f"x_set: {list(graph.nodes())}")
    # base step: x_set = [v] (one node)
    v = x_set[0]
    wx = {(v, v): 1}

    for i in range(1, len(x_set)):
        logging.debug(f"\n-------- COMPUTE WX IT: {i} -----------\n")
        logging.debug(f"current wx: {wx}")
        x = x_set[:i]
        logging.debug(f"x: {x}")
        u = x_set[i]
        logging.debug(f"new vertex u: {u}")
        # Y = X U { Vi }
        wy = {}
        # compute Ry(u) where u is Y \ X (u)
        ry_1 = 0
        for (v, w) in wx.keys():
            ry_1 = ry_1 + graph.edges()[u, v]['weight'] * wx[v, w] * graph.edges()[w, u]['weight']
            if ry_1 >= 1:
                logging.debug(f"{(v, w)}: wx[v,w] = {wx[v, w]} makes ry_1 = {ry_1} be 1")
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

        # normalize wy

        wx = wy

    return wx


def _castaway_rst_log(in_graph: nx.DiGraph, root, trick=True) -> nx.DiGraph:
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
            i_vertex = np.random.choice(x_list, size=1)[0]
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


def kirchoff_rst(graph: nx.DiGraph, root=0, log_probs: bool = False) -> nx.DiGraph:
    """
    Implementation of Kulkarni A8 algorithm for directed graphs.
    """
    if log_probs:
        raise ValueError("Kulkarni RST not implemented for log-probabilities")
    # set root column to ones (it only matters it's not zero)
    matrix = nx.to_numpy_array(graph)
    matrix[:, root] = 1.
    graph = reset_adj_matrix(graph, matrix)
    # normalize graph weights
    graph = normalize_graph_weights(graph, log_probs=log_probs)

    # initialize empty digraph
    tree = nx.DiGraph()
    arc_candidates = sorted([e for e in graph.edges() if e[0] != e[1]], key=lambda x: graph.edges()[x]['weight'],
                            reverse=True)
    aa = tuttes_tot_weight(graph, root)
    deleted_arcs = []
    while len(tree.edges()) < graph.number_of_nodes() - 1:
        # pick edge (sorted by weight so to increase acceptance ratio)
        # print(arc_candidates)
        # print(tree.edges())
        arc = arc_candidates.pop(0)
        # sample uniform (0,1) and check if edge is added
        # sum of weights of trees including tree edges
        a = aa
        # sum of weights of trees including tree edges + e (a' in Kulkarni A8)
        # Leverage score of arc
        aa = 1. if not tree.edges() else graph_weight(tree)
        aa *= graph.edges()[arc]['weight'] * tuttes_tot_weight(graph, root,
                                                               contracted_arcs=[e for e in tree.edges()] + [arc],
                                                               deleted_arcs=deleted_arcs)
        acceptance_ratio = aa / a
        # print(f"a: {a}, aa: {aa}")
        # print(f"acceptance ratio: {acceptance_ratio}")
        if np.random.random() < acceptance_ratio:
            # print(f"adding edge {arc}")
            tree.add_edge(*arc)
            tree.edges()[arc]['weight'] = graph.edges()[arc]['weight']
            # remove edges going in edge[1] from candidates or opposite
            for e in arc_candidates.copy():
                if e[1] == arc[1] or e[::-1] == arc:
                    arc_candidates.remove(e)
        else:
            # print(f"excluding edge {arc}")
            # exclude edge from future consideration
            deleted_arcs.append(arc)
            # recompute aa
            aa = 1. if not tree.edges() else graph_weight(tree)
            aa *= tuttes_tot_weight(graph, root, contracted_arcs=[e for e in tree.edges()],
                                    deleted_arcs=deleted_arcs)

    return tree


def wilson_rst(graph: nx.DiGraph, root=0, log_probs: bool = False) -> nx.DiGraph:
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
        i = np.random.choice(list(x_set))
        u = i
        while u not in t_set:
            # loop-erased random walk
            if log_probs:
                prev[u] = gumbel_max_trick_sample(weights[:, u])
            else:
                prev[u] = np.random.choice(np.arange(n_nodes), size=1, p=weights[:, u])[0]
            u = prev[u]
        u = i
        while u not in t_set:
            # add to tree
            tree.add_edge(prev[u], u, weight=weights[prev[u], u])
            x_set.remove(u)
            t_set.add(u)
            u = prev[u]
    return tree


def colbourn_rst(graph: nx.DiGraph, root=0, log_probs: bool = False):
    """
    Re-adapted from rycolab/treesample implementation
    :param graph:
    :param root:
    :param log_probs:
    :return:
    """
    if log_probs:
        raise ValueError("Colbourne RST not implemented for log-probabilities")
    # normalize graph weights
    graph = normalize_graph_weights(graph, log_probs=log_probs)
    W = nx.to_numpy_array(graph)
    nodes_perm = [i for i in range(W.shape[1])]
    if root != 0:
        nodes_perm = [root] + [i for i in range(W.shape[1]) if i != root]
        W = W[:, nodes_perm]
    tree = _colbourn_tree_from_matrix(W)

    tree = nx.relabel_nodes(tree, {i: nodes_perm[i] for i in range(W.shape[1])})
    for e in tree.edges():
        tree.edges()[e]['weight'] = graph.edges()[e]['weight']
    return tree


def _sample_edge(j, B, A, r) -> tuple[int, float]:

    # compute the marginals
    n = A.shape[0]
    marginals = np.zeros(n)
    for i in range(n):
        if i == j:
            marginals[i] = B[0, i] * r[i]
        else:
            if j != 0:
                marginals[i] += B[j, j] * A[i, j]
            if i != 0:
                marginals[i] -= B[i, j] * A[i, j]
    # correct very small numbers to 0 due to float precision leading to
    # subtractions a - a != 0
    marginals[marginals < 1e-50] = 0
    # re-normalize
    marginals /= np.sum(marginals)
    out = np.random.choice(np.arange(n), p=marginals)
    return out, float(marginals[out])


def _update_BL(i, j, B, L, A, r) -> tuple[np.ndarray, np.ndarray]:
    # code is copied from rycolab/treesample/colbourn.py - credits to
    # condition the laplacian so that i -> j is in any tree
    # K is the laplacian
    n = B.shape[0]
    uj = np.zeros(n)
    if i == j:
        uj[0] = r[j]
    else:
        if j != 0:
            uj[j] = A[i, j]
        if i != 0:
            uj[i] = -A[i, j]
    # update B and L
    u = uj - L[:, j]
    L[:, j] = uj
    bj = B[:, j]
    ub = u.T @ bj
    s = 1 + ub
    B -= np.outer(bj, u.T @ B) / s
    return B, L


def _colbourn_tree_from_matrix(W: np.ndarray) -> nx.DiGraph:
    """
    Assumes root is 0. Wrapper can permute nodes so to arbitrarily set the root. See main function colbourn_rst
    :param W: weight matrix
    :return: nx.DiGraph with tree edges only (is_arborescence = True)
    """
    # nodes
    n = W.shape[0] - 1
    r = W[0, 1:]
    A = W[1:, 1:]
    np.fill_diagonal(A, 0)
    # Kirchoff matrix
    L = _koo_laplacian(A, r)
    B = np.linalg.inv(L).transpose()
    tree = nx.DiGraph()

    for j in range(n):
        i, p_i = _sample_edge(j=j, B=B, A=A, r=r)
        if i == j:
            # i is root
            tree.add_edge(0, j + 1)
        else:
            tree.add_edge(i + 1, j + 1)
        B, L = _update_BL(i, j, B, L, A, r)

    assert nx.is_arborescence(tree)
    return tree


def _koo_laplacian(A, r):
    """
    Root-weighted Laplacian of Koo et al. (2007)
    A is the adjacency matrix and r is the root weight
    """
    L = -A + np.diag(np.sum(A, 0))
    L[0] = r
    return L


if __name__ == '__main__':
    # repeat for different number of nodes
    root = 1
    for n_nodes in [8, 9, 10]:
        trees_sample = {}
        graph = random_uniform_graph(n_nodes)
        sample_size = 500

        # our vs wilson: uniform graph
        start = time.time()
        for s in range(sample_size):
            tree = castaway_rst(graph, root)
        print(f"our time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        start = time.time()
        for s in range(sample_size):
            tree = wilson_rst(graph, root)
        print(f"wilson time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        start = time.time()
        for s in range(sample_size):
            tree = colbourn_rst(graph, root)
        print(f"colbourn time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
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
        tree = castaway_rst(graph, 0)
    print(f"our time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
    start = time.time()
    for s in range(sample_size):
        tree = wilson_rst(graph, 0)
    print(f"wilson time ss = {sample_size}, k = {n_nodes}: {time.time() - start}")
        # print(tree_to_newick(tree))
