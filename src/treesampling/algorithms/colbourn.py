import numpy as np
import networkx as nx
from tqdm import tqdm

import treesampling.utils.graphs as tg
from treesampling.utils.graphs import laplacian, nxtree_from_list


def colbourn_rst(graph: nx.DiGraph | np.ndarray, root=0, log_probs: bool = False):
    """
    Re-adapted from rycolab/treesample implementation
    :param graph:
    :param root:
    :param log_probs:
    :return:
    """
    # normalize graph weights
    if isinstance(graph, nx.DiGraph):
        graph = tg.normalize_graph_weights(graph, log_probs=log_probs)
        W = nx.to_numpy_array(graph)
    elif isinstance(graph, np.ndarray):
        W = graph.copy()
        graph = nx.from_numpy_array(graph)
    W0 = W.copy()

    if log_probs:
        logW = W.copy()
        W = tg.stable_matrix_exp(logW)
        graph = tg.reset_adj_matrix(graph, W)
        W0 = logW

    nodes_perm = [i for i in range(W.shape[1])]
    if root != 0:
        # this makes the root 0 which is required by the subroutine below
        nodes_perm = [root] + [i for i in range(W.shape[1]) if i != root]
        W = W[:, nodes_perm]
    tree_list = _colbourn_tree_from_matrix(W)

    tree = tg.nxtree_from_list(tree_list)
    tree = nx.relabel_nodes(tree, {i: nodes_perm[i] for i in range(W.shape[1])})
    for e in tree.edges():
        # set the weight of the edge to the original weight
        tree.edges()[e]['weight'] = W0[e[0], e[1]]
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


def _colbourn_tree_from_matrix(W: np.ndarray) -> list[int]:
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
    L = laplacian(W)[1:, 1:]
    # L = _koo_laplacian(A, r)
    B = np.linalg.inv(L).transpose()
    if not np.allclose(L @ B.T, np.eye(n)):
        raise ValueError("Inverse is not correct")

    tree = [-1] * (n+1)

    for j in range(n):
        # sample parent i of j
        i, p_i = _sample_edge(j=j, B=B, A=A, r=r)
        if i == j:
            # i is root
            tree[j + 1] = 0
        else:
            tree[j + 1] = i + 1
        B, L = _update_BL(i, j, B, L, A, r)

    assert nx.is_arborescence(nxtree_from_list(tree)), f"Tree is not arborescence: {tree}"
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
        Z = tg.tuttes_determinant(X)
        # L = _koo_laplacian(X[1:, 1:], X[0, 1:])
        # Z = np.linalg.det(L)
        # print(f"total weight: {Z}")

        # save frequencies and weight of each new tree
        dist = {}
        for i in range(N):
            tree = tuple(_colbourn_tree_from_matrix(X))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N
            bar.update(1)

        for tree in dist:
            prob = tg.tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

if __name__ == '__main__':
    check()
