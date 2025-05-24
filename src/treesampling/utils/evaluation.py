import heapq
import itertools
import logging
import time

import networkx as nx
import numpy as np

from treesampling import StableOp
from treesampling.utils.graphs import prufer_to_rooted_parent, tree_weight, cayleys_formula, tuttes_tot_weight, \
    mat_minor, tree_to_newick, parlist_to_newick


def get_true_pmf(matrix, cutoff_p=0.99999, cutoff_k=1000, verbose=False):
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # keep first K trees in a dictionary
    K = cutoff_k
    p = cutoff_p
    n = matrix.shape[0]
    # priority queue for K trees
    it, top_k_trees, tot_weight = enumerate_trees(K, matrix, n)

    logger.debug(f"Total weight: e^{tot_weight} (it: {it}, cayley tot #trees: {cayleys_formula(n)})")
    # walrus op
    if (tuttes_det:=tuttes_tot_weight(np.exp(matrix), 0)) > 0:
        logger.debug(f"Tutte's det: e^{np.log(tuttes_det)} ({tuttes_det})")
    else:
        logger.debug(f"Tutte's det is non-positive: {tuttes_det}")
    # normalize weights
    logger.debug(f"First {p * 100}% of trees:")
    acc = 0.
    top_trees = {}  # either limited by K or by the p-th percentile
    for i, (w, t, nwk) in enumerate(heapq.nlargest(K, top_k_trees)):
        norm_tree_weight = np.exp(w - tot_weight)
        top_trees[nwk] = (w, norm_tree_weight, t)
        acc += norm_tree_weight
        if acc < p:
            logger.debug(f"Tree {i}: {w} {norm_tree_weight} {t} - newick: {nwk}")
        else:
            break
    logger.debug(f"Total norm weight (should be close to 1): {acc}")
    return top_trees


def enumerate_trees(K, matrix, n):
    top_k_trees = []
    # compute total weight of trees
    tot_weight = -np.inf
    # print total weight increase in
    it = 0
    for prufer_tree in itertools.product(range(n), repeat=n - 2):
        tree, newick = prufer_to_rooted_parent(prufer_tree)
        rooted_tree = tree
        assert rooted_tree[0] == -1
        weight = tree_weight(rooted_tree, matrix, log_probs=True)
        tot_weight = np.logaddexp(tot_weight, weight)

        if len(top_k_trees) < K:
            # largest weight first
            heapq.heappush(top_k_trees, (weight, rooted_tree, newick))
        else:
            if weight > top_k_trees[0][0]:
                heapq.heapreplace(top_k_trees, (weight, rooted_tree, newick))
        it += 1
    return it, top_k_trees, tot_weight


def get_sampler_pmf(matrix: np.ndarray, sampler: callable, n: int, time_limit=None) -> dict:
    trees = {}
    for i in range(n):
        start = time.time()
        t = tuple(sampler(matrix))
        exec_time = time.time() - start
        if time_limit is not None and exec_time > time_limit:
            logging.debug(f"Time limit exceeded: {exec_time} > {time_limit}")
            return {}
        if t not in trees:
            trees[t] = 0
        trees[t] += 1 / n
    pmf = {}
    for tree, freq in sorted(trees.items(), key=lambda x: x[1], reverse=True):
        nwk = parlist_to_newick(tree)
        w = tree_weight(tree, matrix, log_probs=True)
        pmf[nwk] = (w, freq, tree)
    return pmf

def get_sampler_pmf_times(matrix: np.ndarray, sampler: callable, n: int, time_limit=None) -> (dict, list):
    trees = {}
    times = []
    for i in range(n):
        start = time.time()
        t = tuple(sampler(matrix))
        exec_time = time.time() - start
        times.append(exec_time)
        if time_limit is not None and exec_time > time_limit:
            logging.debug(f"Time limit exceeded: {exec_time} > {time_limit}")
            return {}, times
        if t not in trees:
            trees[t] = 0
        trees[t] += 1 / n
    pmf = {}
    for tree, freq in sorted(trees.items(), key=lambda x: x[1], reverse=True):
        nwk = parlist_to_newick(tree)
        w = tree_weight(tree, matrix, log_probs=True)
        pmf[nwk] = (w, freq, tree)
    return pmf, times


def cheeger_constant(matrix, root=None, log_probs=False):
    """
    Compute the Cheeger constant of a directed graph
    """
    op = StableOp(log_probs=log_probs)
    if root is not None:
        matrix = mat_minor(matrix, root, root)
    n = matrix.shape[0]
    component = None
    cheeger = np.inf
    max_out, min_in = 0, np.inf
    for s_size in range(2, n-1):
        for s in itertools.combinations(range(n), s_size):
            # print(s)
            s = set(s)
            s_bar = set(range(n)) - s
            num = op.add([matrix[v, u] for v in s_bar for u in s])
            denom = min(op.add([matrix[v, u] for v in s for u in s if v != u]),
                        op.add([matrix[v, u] for v in s_bar for u in s_bar if v != u]))
            cheeger_ = op.div(num, denom)
            if cheeger_ < cheeger:
                max_out, min_in = num, denom
                cheeger = cheeger_
                component = s
    return cheeger, [c + (0 if c < root else 1) for c in component], max_out, min_in


def jens_conductance(matrix, root=None, log_probs=False):
    """
    Compute the Jens conductance of a directed graph
    """
    op = StableOp(log_probs=log_probs)
    if root is not None:
        matrix = mat_minor(matrix, root, root)
    n = matrix.shape[0]
    component = None
    conductance = np.inf
    max_out, min_in = 0, np.inf
    for s_size in range(2, n-1):
        for s in itertools.combinations(range(n), s_size):
            # print(s)
            s = set(s)
            s_bar = set(range(n)) - s
            num = max([op.add([matrix[v, u] for v in s_bar]) for u in s])
            denom = min([op.add([matrix[v, u] for v in s if v != u]) for u in s])
            conductance_ = op.div(num, denom)
            if conductance_ < conductance:
                max_out, min_in = num, denom
                conductance = conductance_
                component = s
    return conductance, [c + (0 if c < root else 1) for c in component], max_out, min_in


def get_victree_demo_matrix():
    matrix = np.array([
        [-np.inf, -174.769, -174.769, -244.775, -249.29, -276.033, -240.241],
        [-np.inf, -np.inf, -56.238, -124.203, -143.693, -174.44, -145.639],
        [-np.inf, -59.999, -np.inf, -18.449, -56.7, -107.202, -51.118],
        [-np.inf, -133.159, 0., -np.inf, -126.356, -155.468, -123.161],
        [-np.inf, -141.243, -39.487, -117.625, -np.inf, -105.354, -60.438],
        [-np.inf, -177.913, -78.955, -150.078, -84.43, -np.inf, -102.489],
        [-np.inf, -131.7, -25.589, -115.073, -59.796, -109.966, -np.inf]
    ])
    return matrix


def analyse_true_dist(matrix) -> tuple:
    """
    Compute the true distribution of trees (via enumeration of all trees). Print the MST and the frequency of each edge
    in the trees distribution.
    :param matrix: log-weights matrix
    :return: dict of trees with their frequencies: {tree: (weight, norm_weight, parent_list)}, MST_newick, edge_freq
    """
    true_pmf = get_true_pmf(matrix, cutoff_p=0.99999, cutoff_k=16000)
    graph = nx.DiGraph()
    for i, j in itertools.product(range(matrix.shape[0]), repeat=2):
        if i != j and j != 0:
            graph.add_edge(i, j, weight=matrix[i, j])
    mst_nwk = tree_to_newick(nx.maximum_spanning_arborescence(graph))
    logging.debug(f"MST: {mst_nwk}")
    edge_freq = np.zeros_like(matrix)
    for _, norm_weight, tree in true_pmf.values():
        for i in range(1, len(tree)):
            edge_freq[tree[i], i] += norm_weight
    logging.debug(np.array_str(edge_freq, max_line_width=100, precision=3, suppress_small=True))
    return true_pmf, mst_nwk, edge_freq
