import networkx as nx
import numpy as np
from tqdm import tqdm
from treesample.wilson import WilsonSample
import treesample

from treesampling.utils.graphs import tuttes_tot_weight, tuttes_determinant, tree_weight, brute_force_tot_weight
from treesampling.utils.math import StableOp


def wilson_rst(graph: nx.DiGraph, root: int, log_probs: bool = False) -> nx.DiGraph:
    # get matrix from graph
    W = nx.to_numpy_array(graph)
    # setup matrix
    zero = -np.inf if log_probs else 0.
    np.fill_diagonal(W, zero)
    W[:, root] = zero

    # swap root to 0
    Wr = np.roll(W, -root, axis=0)
    Wr = np.roll(Wr, -root, axis=1)
    # sample tree
    tree = wilson_rst_from_matrix(Wr, log_probs)
    # swap back to original root relabeling nodes
    # FIXME: this is not correct, not only root should be swapped, but also the other nodes
    tree = np.roll(tree, root)

    # build tree
    tree_nx = nx.DiGraph()
    for j, i in enumerate(tree):
        if i != -1:
            tree_nx.add_edge(i, j, weight=W[i, j])
    return tree_nx


def wilson_rst_from_matrix(X: np.ndarray, log_probs: bool = False) -> list[int]:
    """
    Takes a weight matrix and return the tree as array of shape (n_nodes-1,)
    :param weights: np.ndarray of shape (n_nodes, n_nodes)
    :param root: int, root node
    :param log_probs: bool, if True, weights are in log scale
    :return: np.ndarray of shape (n_nodes-1,)
    """
    op = StableOp(log_probs=log_probs)
    # std matrix setup
    weights = np.copy(X)
    weights[:, 0] = op.zero() # set root weights to zero
    weights[np.diag_indices(weights.shape[0])] = op.zero() # set diagonal to zero
    weights[:, 1:] = op.normalize(weights[:, 1:], axis=0) # normalize weights
    # ---------------------

    n = weights.shape[0]
    tree = [-1] * n
    t_set = {0}
    x_set = {i for i in range(1, n)}
    # pick random node from x
    prev = [-1] * n
    while x_set:
        i = int(np.random.choice(list(x_set)))
        u = i
        while u not in t_set:
            # loop-erased random walk
            prev[u] = op.random_choice(weights[:, u])
            u = prev[u]
        u = i
        while u not in t_set:
            # add to tree
            tree[u] = prev[u]
            x_set.remove(u)
            t_set.add(u)
            u = prev[u]
    return tree

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
        Z = tuttes_determinant(X)
        # print(f"tutte's Z: {Z}")
        Z = brute_force_tot_weight(X)
        # print(f"brute force Z: {Z}")
        # L = _koo_laplacian(X[1:, 1:], X[0, 1:])
        L = treesample.util.laplacian(X[1:, 1:], X[0, 1:])
        Z = np.linalg.det(L)
        # print(f"koo's Z: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        sampler = WilsonSample(X)
        for i in range(N):
            tree = tuple(next(sampler.sample()).tolist())
            # tree = tuple(wilson_rst_from_matrix(X, log_probs=False))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N
            bar.update(1)

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

if __name__ == '__main__':
    check()
