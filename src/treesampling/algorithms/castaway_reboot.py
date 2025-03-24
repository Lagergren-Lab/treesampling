import numpy as np

import itertools

from treesampling.utils.graphs import tuttes_determinant, tree_weight
from treesampling.utils.math import StableOp


def castaway_rst_from_matrix(W: np.ndarray, log_probs: bool = False, trick: bool = False) -> np.ndarray:
    """
    Takes a weight matrix and return the tree as array of shape (n_nodes-1,)
    :param W: matrix of shape (n_nodes, n_nodes)
    :param log_probs: bool, if True, weights are in log scale
    :return: np.ndarray of shape (n_nodes-1,) with the tree, entries are the parent node
    """
    global op
    op = StableOp(log_probs=log_probs)

    def build_table(x_list, W):
        # base step: x = [v] (first node)
        v = x_list[0]
        wx = {(v, v): op.one()}
        for i in range(1, len(x_list)):
            x = x_list[:i]
            u = x_list[i]
            # Y = X U { u }
            wy = {}
            # compute Ry(u) where u is Y \ X (u)
            ry_1 = op.zero() # log: -np.inf
            # marginalize over all paths from v to w
            for v, w in itertools.product(x, repeat=2):
                ry_1 = op.add([ry_1, op.mul([W[u, v], wx[v, w], W[w, u]])])
            ry = op.div(op.one(), op.sub(op.one(), ry_1))

            # compute Wy
            # partial computations
            # - Wxy, all paths from any v to new vertex u = Y \ X
            # - Wyx, all paths from the new vertex u to any v in X
            wxy = {}
            wyx = {}
            for v in x:
                wxy[v] = op.zero() # log: -np.inf
                wyx[v] = op.zero() # log: -np.inf
                for vv in x:
                    wxy[v] = op.add([wxy[v], op.mul([W[vv, u], wx[v, vv]])]) # log: logaddexp(wxy[v], gw[vv, u] + wx[v, vv])
                    wyx[v] = op.add([wyx[v], op.mul([wx[vv, v], W[u, vv]])]) # log: logaddexp(wyx[v], wx[vv, v] + gw[u, vv])

            # write new W table
            for v in x:
                wy[u, v] = op.mul([ry, wyx[v]]) # log: wy[u, v] = ry + wyx[v]
                wy[v, u] = op.mul([wxy[v], ry]) # log: wy[v, u] = wxy[v] * ry
                for w in x:
                    # general update
                    wy[v, w] = op.add([wx[v, w], op.mul([wxy[v], ry, wyx[w]])]) # log: wy[v, w] = logaddexp(wx[v, w], wxy[v] + ry + wyx[w])
            # new self returning random path
            wy[u, u] = ry
            wx = wy
        return wx

    def pick_u_prob(wx, W, i, t_set):
        # compute probabilities of direct arc u -> i
        q = np.zeros(W.shape[0])
        x_set = set(k for k, _ in wx.keys())
        p_attach = {j: 0 for j in x_set} # sum_{j' in t_set} wj'j, for all j in X-{u}
        for j in x_set:
            p_attach[j] = op.add([W[j_prime, j] for j_prime in t_set])

        for k in x_set:
            q[k] = op.mul([W[k, i], op.add([ op.mul([wx[j, k], p_attach[j]]) for j in x_set - {k} ])])
        for k in t_set:
            q[k] = W[k, i]

        return op.normalize(q)


    # std matrix setup
    weights = np.copy(W)
    weights[:, 0] = op.zero() # set root weights to zero
    weights[np.diag_indices(weights.shape[0])] = op.zero() # set diagonal to zero
    weights[:, 1:] = op.normalize(weights[:, 1:], axis=0) # normalize weights
    # ---------------------

    dangling_path: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
    tree = [-1] * weights.shape[0]
    x_set = set(range(1, weights.shape[0]))  # set of nodes not in tree
    # iterate for each node
    while len(x_set) > 1:
        # choose new i from x if no dangling nodes
        if not dangling_path:
            # NOTE: stochastic vs sorted choice (should not matter)
            i_vertex = np.random.choice(np.array(list(x_set), dtype=int))
        # or set i to be last node
        else:
            latest_edge = dangling_path[-1]
            i_vertex = latest_edge[0]
        x_set.remove(i_vertex)

        # update Wx table
        wx = build_table(list(x_set), weights)

        # for each node in either S or X - u, compute the probability of direct arc u -> i
        # nodes_lab, w_choice = self._compute_lexit_table(i_vertex, x_set, self.wx.get_wx(), tree)
        # O(n^2)
        tree_nodes = [j for j, i in enumerate(tree) if i != -1] + [0]
        q = pick_u_prob(wx, weights, i_vertex, tree_nodes)
        # pick u from q
        u_vertex = np.random.choice(np.array(range(weights.shape[0]), dtype=int), p=q)
        dangling_path.append((u_vertex, i_vertex))
        if u_vertex in tree_nodes:
            # if u picked from tree, attach dangling path and reset
            # add dangling path edges to tree
            for u, i in dangling_path:
                tree[i] = u

            dangling_path = []
    return tree


def test():
    n_seeds = 100
    N = 10000
    k = 4
    acc = 0
    for i in range(n_seeds):
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
            tree = tuple(castaway_rst_from_matrix(X, log_probs=False))
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

def test_uniform():
    X = np.random.uniform(0, 1, size=(4, 4))
    tree = castaway_rst_from_matrix(X, log_probs=False)
    print(tree)


if __name__ == "__main__":
    test()
