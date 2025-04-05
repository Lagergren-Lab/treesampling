import itertools
import logging

import numpy as np
from tqdm import tqdm

import treesampling.utils.graphs as tg
from treesampling.utils.math import StableOp

def kulkarni_rst(X: np.ndarray, root=0, log_probs: bool = False, debug: bool = False) -> list[int]:
    """
    Implementation of Kulkarni A8 algorithm for directed graphs.
    """
    logger = logging.getLogger("kulkarni")
    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    # set root column to ones (it only matters it's not zero)
    op = StableOp(log_probs=log_probs)
    matrix = np.copy(X)
    matrix[:, root] = op.one()
    # normalize graph weights
    non_root_col = [i for i in range(matrix.shape[1]) if i != root]
    matrix[:, non_root_col] = op.normalize(matrix[:, non_root_col], axis=0)
    logger.debug("matrix after normalization: ")
    logger.debug(matrix)
    if log_probs:
        log_matrix = np.copy(matrix)
        matrix = np.exp(log_matrix)

    # initialize empty digraph
    n = matrix.shape[0]
    tree = [-1] * n
    rooted_tree_arcs = [(u, v) for u, v in itertools.product(range(n), repeat=2) if u != v and v != root]
    arc_candidates = sorted([e for e in rooted_tree_arcs if matrix[e] > 0], key=lambda x: matrix[x], reverse=True)
    # shuffle(arc_candidates)
    aa = tg.tuttes_tot_weight(matrix, root)
    deleted_arcs = []
    while tree.count(-1) > 1:
        # pick edge (sorted by weight so to increase acceptance ratio)
        tree_arcs = [(u, v) for v, u in enumerate(tree) if u != -1]
        logger.debug(f"candidates: {arc_candidates}")
        logger.debug(f"current tree: {tree_arcs}")
        arc = arc_candidates.pop(0)
        # sample uniform (0,1) and check if edge is added
        # sum of weights of trees including tree edges
        a = aa
        # sum of weights of trees including tree edges + e (a' in Kulkarni A8)
        # Leverage score of arc
        aa = 1 if tree.count(-1) == n else tg.tree_weight(tree, matrix)
        aa *= matrix[arc] * tg.tuttes_tot_weight(matrix, root,
                                                 contracted_arcs=tree_arcs + [arc], deleted_arcs=deleted_arcs)
        acceptance_ratio = aa / a
        logger.debug(f"a: {a}, aa: {aa}")
        logger.debug(f"acceptance ratio: {acceptance_ratio}")
        if np.random.random() < acceptance_ratio:# or not arc_candidates:
            logger.debug(f"adding edge {arc}")
            tree[arc[1]] = arc[0]
            # remove edges going in edge[1] from candidates or opposite
            for e in arc_candidates.copy():
                if e[1] == arc[1] or e[::-1] == arc:
                    arc_candidates.remove(e)
        else:
            logger.debug(f"excluding edge {arc}")
            # exclude edge from future consideration
            deleted_arcs.append(arc)
            # recompute aa
            aa = 1 if tree.count(-1) == n else tg.tree_weight(tree, matrix)
            aa *= tg.tuttes_tot_weight(matrix, root, contracted_arcs=tree_arcs, deleted_arcs=deleted_arcs)

    return tree

def check_log():
    n_seeds = 100
    N = 10000
    k = 4
    acc = 0
    bar = tqdm(total=n_seeds * N)
    seed_acc = {}
    for i in range(n_seeds):
        old_acc = acc
        X = np.random.uniform(0, 1, size=(k, k))
        # setup matrix
        np.fill_diagonal(X, 0)
        X[:, 0] = 0.
        X[:, 1:] = X[:, 1:] / np.sum(X[:, 1:], axis=0)
        log_X = -np.inf * np.ones_like(X)
        log_X[X > 0] = np.log(X[X > 0])
        # compute total trees weight
        Z = tg.tuttes_tot_weight(X, 0)
        L1 = tg.kirchoff_matrix(X)
        # L1 is also known as Kirchoff matrix (as in Colbourn 1996)
        L1r = tg.mat_minor(L1, row=0, col=0)
        cond_number = np.linalg.cond(L1r)
        # L = _koo_laplacian(X[1:, 1:], X[0, 1:])
        # Z = np.linalg.det(L)
        # print(f"total weight: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        for i in range(N):
            tree = tuple(kulkarni_rst(log_X, root=0, log_probs=True))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N
            bar.update(1)

        for tree in dist:
            prob = tg.tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
        seed_acc[i] = acc - old_acc, cond_number
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")
    # check accuracy and condition numbers
    print(sorted(seed_acc.items(), key=lambda x: x[1][0], reverse=True))

def check():
    n_seeds = 100
    N = 10000
    k = 4
    acc = 0
    bar = tqdm(total=n_seeds * N)
    for i in range(n_seeds):
        X = np.random.uniform(0, 1, size=(k, k))
        # setup matrix
        np.fill_diagonal(X, 0)
        X[:, 0] = 0.
        X[:, 1:] = X[:, 1:] / np.sum(X[:, 1:], axis=0)
        # compute total trees weight
        Z = tg.tuttes_determinant(X)
        # L = _koo_laplacian(X[1:, 1:], X[0, 1:])
        # Z = np.linalg.det(L)
        # print(f"total weight: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        for i in range(N):
            tree = tuple(kulkarni_rst(X))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N
            bar.update(1)

        for tree in dist:
            prob = tg.tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

if __name__ == "__main__":
    check()
    # check_log()