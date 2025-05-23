import itertools
import random
import networkx as nx
import numpy as np

from treesampling.utils.graphs import reset_adj_matrix, tuttes_determinant, tree_weight
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
    weight_matrix = nx.to_numpy_array(graph)

    if log_probs:
        return _castaway_rst_log(weight_matrix, root, trick)
    else:
        return _castaway_rst_plain(weight_matrix, root, trick)


def _castaway_rst_plain(weight_matrix: np.ndarray, root, trick=True) -> nx.DiGraph:
    """
    Sample one tree from a given graph with fast arborescence sampling algorithm.
    :param weight_matrix: np.ndarray, weighted adjacency matrix
    :param root: root node
    :param trick: if false, Wx gets re-computed every time
    :return: nx.DiGraph with tree edges only
    """
    # normalize out arcs (cols)
    # print("BEGIN ALGORITHM")
    n_nodes = weight_matrix.shape[0]
    graph = nx.DiGraph()
    # initialize edges
    for u, v in itertools.product(range(n_nodes), repeat=2):
        if u != v and v != root and weight_matrix[u, v] != 0:
            graph.add_edge(u, v, weight=weight_matrix[u, v])
    graph = normalize_graph_weights(graph, rowwise=False, log_probs=False)

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


def _castaway_rst_log(weight_matrix: np.ndarray, root, trick=True) -> nx.DiGraph:
    """
    Sample one tree from a given graph with fast arborescence sampling algorithm.
    :param weight_matrix: np.ndarray, weighted adjacency matrix
    :param root: root node
    :param trick: if false, Wx gets re-computed every time
    :return: nx.DiGraph with tree edges only
    """
    # ----------------------------
    # PREPARE MATRIX FOR SAMPLING
    n_nodes = weight_matrix.shape[0]
    weights = np.copy(weight_matrix)
    weights[np.diag_indices(n_nodes)] = -np.inf
    # normalize out arcs (cols)
    weights = weights - np.logaddexp.reduce(weights, axis=0, keepdims=True)
    weights[:, root] = -np.inf

    graph = nx.DiGraph()
    # initialize edges
    for u, v in itertools.product(range(n_nodes), repeat=2):
        if u != v and v != root and weight_matrix[u, v] != -np.inf:
            graph.add_edge(u, v, weight=weight_matrix[u, v])
    # ----------------------------
    # ALGORITHM
    # variables
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
                    try:
                        a = logsubexp(wy_table[v, w], wy_table[v, u] + wy_table[u, w] - wy_table[u, u])
                    except ValueError as ve:
                        print(f"[DBG] logsubexp failed at update: removing {u} from y_set = {set([x for x, y in wy_table.keys()])}")
                        raise ve
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

# OLDER HELP FUNCTIONS

def logsubexp(l1, l2):
    """
    OLD version: Subtraction in linear scale of log terms.
    """
    if np.isclose(l1, l2):
        # includes also case l1 == l2 == - np.inf
        res = - np.inf
    else:
        if not l1 > l2:
            raise ValueError(f"l1: {l1}, l2: {l2}, l1 should be greater than l2")
        dx = -l1 + l2
        exp_x = np.exp(dx)
        res = l1 + np.log(1 - exp_x)

    return res

def normalize_graph_weights(graph, log_probs=False, rowwise=False) -> nx.DiGraph:
    adj_mat = nx.to_numpy_array(graph)
    axis = 1 if rowwise else 0
    if not log_probs:
        adj_mat = adj_mat / adj_mat.sum(axis=axis, keepdims=True)
    else:
        adj_mat = adj_mat - np.logaddexp.reduce(adj_mat, axis=axis, keepdims=True)
    norm_graph = reset_adj_matrix(graph, adj_mat)
    return norm_graph

def gumbel_max_trick_sample(log_probs: np.ndarray) -> int:
    # check that input log probs are normalized
    # assert np.isclose(sp.logsumexp(log_probs), 0.0), (f"sum of log probs should be 0.0, but is "
    #                                                   f": {sp.logsumexp(log_probs)}")
    gumbels = np.random.gumbel(size=len(log_probs))
    sample = np.argmax(log_probs + gumbels)
    return sample

def tree_to_list(tree_nx: nx.DiGraph) -> list[int]:
    # return the list of parents for each node, root node has -1
    tree_list = [-1] * tree_nx.number_of_nodes()
    for i, j in tree_nx.edges():
        tree_list[j] = i

    return tree_list

def test():
    N = 10000
    n_seeds = 100
    k = 4
    acc = 0
    for seed in range(n_seeds):
        X = np.random.uniform(0, 1, size=(k, k))
        # setup matrix
        np.fill_diagonal(X, 0)
        X[:, 0] = 0.
        X[:, 1:] = X[:, 1:] / np.sum(X[:, 1:], axis=0)
        # compute total trees weight
        Z = tuttes_determinant(X)
        # print(f"total weight: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        for i in range(N):
            tree_nx = _castaway_rst_plain(X, root=0, trick=False)
            tree = tuple(tree_to_list(tree_nx))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(np.array(tree, dtype=int), X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, empirical: {dist[tree]}")

    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")


def test_log():
    print("Testing castaway_legacy with log probabilities...")
    n_seeds = 100
    N = 10000
    n_seeds = 100
    k = 4
    acc = 0
    for seed in range(n_seeds):
        X = np.random.uniform(0, 1, size=(k, k))
        # setup matrix
        np.fill_diagonal(X, 0)
        X[:, 0] = 0.
        X[:, 1:] = X[:, 1:] / np.sum(X[:, 1:], axis=0)
        # compute total trees weight
        Z = tuttes_determinant(X)
        # print(f"total weight: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        for i in range(N):
            tree_nx = _castaway_rst_log(np.log(X), root=0, trick=False)
            tree = tuple(tree_to_list(tree_nx))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(np.array(tree, dtype=int), X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, empirical: {dist[tree]}")

    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")


if __name__=='__main__':
    test_log()

    # create a directed graph
    # # graph, g_tree = random_tree_skewed_graph(5, 20, root=0, log_probs=True)
    # graph = random_block_matrix_graph(5, 2, True, p=0.05)
    #
    # # set seed
    # # random.seed(42)
    # # np.random.seed(42)
    # # # sample a tree with legacy version
    # # tree1 = random_spanning_tree_log(graph, root=0)
    #
    # random.seed(42)
    # np.random.seed(42)
    # # sample with OO version
    # sampler = CastawayRST(graph, root=0, log_probs=True)
    # for _ in range(100):
    #     tree2 = None
    #     miss = 0
    #     while tree2 is None and miss < 20:
    #         try:
    #             tree2 = random_spanning_tree_log(graph, root=0)
    #         except ValueError as ve:
    #             miss += 1
    #             tree2 = None
    #     if miss > 20:
    #         print(f"FAIL")
    #         continue
    #
    # # print(tree_to_newick(tree1))
    # print(tree_to_newick(tree2))
