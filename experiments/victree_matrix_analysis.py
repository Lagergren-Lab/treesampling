"""
Take an interesting matrix from VICTree execution, compute the total weight via brute force and use it for
assessing sampling correctness.
"""
import heapq
import itertools
import logging

import networkx as nx
import numpy as np

from treesampling.algorithms import CastawayRST
from treesampling.utils.graphs import tree_weight, cayleys_formula, tree_to_newick


def prufer_to_rooted_parent(prufer):
    nx_tree = nx.from_prufer_sequence(prufer)
    rooted_tree = nx.dfs_tree(nx_tree, 0)
    parent = [-1] * nx_tree.number_of_nodes()
    for u, v in rooted_tree.edges:
        parent[v] = u

    return tuple(parent), tree_to_newick(rooted_tree)

def get_true_pmf(matrix, cutoff_p=0.99999, cutoff_k=1000):
    # keep first K trees in a dictionary
    K = cutoff_k
    p = cutoff_p
    n = matrix.shape[0]
    # priority queue for K trees
    top_k_trees = []
    # compute total weight of trees
    tot_weight = -np.inf
    # print total weight increase in
    it = 0
    for prufer_tree in itertools.product(range(n), repeat=n-2):
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
        it +=1

    print(f"Total weight: {tot_weight} (it: {it}, cayley tot #trees: {cayleys_formula(n)})")
    # normalize weights
    print(f"First {p * 100}% of trees:")
    acc = 0.
    top_trees = []  # either limited by K or by the p-th percentile
    for i, (w, t, nwk) in enumerate(heapq.nlargest(K, top_k_trees)):
        norm_tree_weight = np.exp(w - tot_weight)
        top_trees.append((w, norm_tree_weight, t, nwk))
        acc += norm_tree_weight
        if acc < p:
            print(f"Tree {i}: {w} {norm_tree_weight} {t} - newick: {nwk}")
        else:
            break
    print(f"Total norm weight (should be close to 1): {acc}")
    return top_trees


def main():
    matrix = np.array([
        [-np.inf, -174.769, -174.769, -244.775, -249.29 , -276.033, -240.241],
        [-np.inf, -np.inf, -56.238, -124.203, -143.693, -174.44 , -145.639],
        [-np.inf, -59.999, -np.inf, -18.449, -56.7  , -107.202, -51.118],
        [-np.inf, -133.159, 0.   , -np.inf, -126.356, -155.468, -123.161],
        [-np.inf, -141.243, -39.487, -117.625, -np.inf, -105.354, -60.438],
        [-np.inf, -177.913, -78.955, -150.078, -84.43 , -np.inf, -102.489],
        [-np.inf, -131.7  , -25.589, -115.073, -59.796, -109.966, -np.inf]
    ])
    true_pmf = get_true_pmf(matrix, cutoff_p=0.99999, cutoff_k=1000)
    edge_occurrency = np.zeros_like(matrix)
    for _, norm_weight, tree, _ in true_pmf:
        for i in range(1, len(tree)):
            edge_occurrency[tree[i], i] += norm_weight
    print(np.array_str(edge_occurrency, max_line_width=100, precision=3, suppress_small=True))

    # sample with debug on
    sampler = CastawayRST(matrix, root=0, log_probs=True, debug=True)
    sampler.sample_tree_as_list()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
