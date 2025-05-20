import itertools
import logging

import networkx as nx
import numpy as np

from treesampling import TreeSampler
from treesampling.utils.graphs import normalize_graph_weights, tree_to_newick, random_uniform_graph, tuttes_determinant, \
    tree_weight, graph_weight
from treesampling.utils.math import StableOp


class WxTable:
    def __init__(self, x: list, weights: np.ndarray = None, root: int = 0, log_probs: bool = False, cache_size: int = 1, **kwargs):
        self.op = StableOp(log_probs=log_probs)  # dispatch stable operations
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if weights is None:
            self.logger.warning("graph parameter is deprecated, just use weight matrix")
            self.weights = kwargs.get("graph")
        self.root = root
        self.x = x
        self.weights = weights.copy()
        self.debug = kwargs.get('debug', False)
        self.cache_size = cache_size
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # cache which stores the computed tables and the amount of calls to the cache (hits)
        # this is useful to avoid recomputing the same table multiple times when graph weights are not updated and
        # there are recurrent x sets
        self._cache_hits = {}
        self._cached_tables = {}
        self.wx_dict = self._build()
        self._complete_x = self.x.copy()
        self._complete_table = self.wx_dict.copy()

    @property
    def complete_table(self):
        return self._complete_table

    @property
    def complete_x(self):
        return self._complete_x

    def _build(self) -> dict:
        # graph weights
        gw = self.weights

        x_list = self.x
        # TEST: order x set by node out-degree among nodes in x
        x_order = np.argsort(self.op.add(gw[:, x_list], axis=1)).tolist()
        x_list = [i for i in x_order if i in x_list]
        self.logger.debug(f"\t- Order of x set: {x_list}")

        # base step: x = (first node)
        v0 = x_list[0]
        wx = {(v0, v0): self.op.one()}
        self.logger.debug(f"\t- Initializing Wx table with node {v0}: wx = {wx}")
        # for i in range(1, len(x_list)):
        i = 1
        while len(wx) < len(x_list) ** 2:
            x = x_list[:i]
            u = x_list[i]
            # Y = X U { u }
            wy = {}
            # compute Ry(u) where u is Y \ X (u)
            ry_1 = self.op.zero() # log: -np.inf
            # marginalize over all paths from v to w
            for v, w in itertools.product(x, repeat=2):
                ry_1 = self.op.add([ry_1, self.op.mul([gw[u, v], wx[v, w], gw[w, u]])])
                # log: logaddexp(ry_1, gw[u, v] + wx[v, w] + gw[w, u])
            self.logger.debug(f"\t- Ry(u, 1) for node {u} in Y = {x}: {ry_1}")
            ry = self.op.div(self.op.one(), self.op.sub(self.op.one(), ry_1))
            # log: -np.log(1 - np.exp(ry_1))
            self.logger.debug(f"\t- Ry(u) for node {u} in Y = {x}: {ry}")

            # compute Wy
            # partial computations
            # - Wxy, all paths from any v to new vertex u = Y \ X
            # - Wyx, all paths from the new vertex u to any v in X
            wxy = {}
            wyx = {}
            for v in x:
                wxy[v] = self.op.zero() # log: -np.inf
                wyx[v] = self.op.zero() # log: -np.inf
                for vv in x:
                    wxy[v] = self.op.add([wxy[v], self.op.mul([gw[vv, u], wx[v, vv]])]) # log: logaddexp(wxy[v], gw[vv, u] + wx[v, vv])
                    wyx[v] = self.op.add([wyx[v], self.op.mul([wx[vv, v], gw[u, vv]])]) # log: logaddexp(wyx[v], wx[vv, v] + gw[u, vv])

            # write new W table
            for v in x:
                wy[u, v] = self.op.mul([ry, wyx[v]]) # log: wy[u, v] = ry + wyx[v]
                wy[v, u] = self.op.mul([wxy[v], ry]) # log: wy[v, u] = wxy[v] * ry
                self.logger.debug(f"\t- Wy({u}, {v}) = ry({ry}) * wyx({v}) = {wy[u, v]}")
                self.logger.debug(f"\t- Wy({v}, {u}) = wxy({v}) * ry({ry}) = {wy[v, u]}")
                for w in x:
                    # general update
                    wy[v, w] = self.op.add([wx[v, w], self.op.mul([wxy[v], ry, wyx[w]])]) # log: wy[v, w] = logaddexp(wx[v, w], wxy[v] + ry + wyx[w])
                    self.logger.debug(f"\t- Wy({v}, {w}) = wx({v}, {w}) + wxy({v}) * ry({ry}) * wyx({w}) = {wy[v, w]}")
            # new self returning random path
            wy[u, u] = ry
            self.logger.debug(f"\t- Wy({u}, {u}) = ry({ry}) = {wy[u, u]}")

            # update wx
            wx = wy
            self.logger.debug(f"\t- Updated Wx table with node {u}")

            self._update_cache(wx, x)
            i += 1
        return wx

    def update(self, u, trick=False):
        """
        Update the Wx table by removing the node u from the current x set (Y -> X).
        :param u: node to remove from the x set
        :param trick: boolean, if True, the algorithm runs in O(n^3) by efficiently updating the W table. Otherwise W
            is computed from scratch every time a new arc is added to the tree
        """
        assert u in self.x, f"Node {u} not in x set prior to removal and update"
        self.x = [v for v in self.x if v != u]

        x_tuple = tuple(sorted(self.x))
        if not self.x:
            # if x is empty, table is empty
            self.wx_dict = {}
        elif self.cache_size > 1 and x_tuple in self._cached_tables:
            # if the requested table has already been computed, avoid recomputing
            self.wx_dict = self._cached_tables[x_tuple]
            self._cache_hits[x_tuple] += 1
        elif trick:
            # remove node in O(n^2)
            self.wx_dict = self._update_trick(u)
        else:
            # recompute Wx from scratch in O(n^3)
            self.wx_dict = self._build()
        # no nans allowed
        assert not np.any([np.isnan(self.wx_dict[k]) for k in self.wx_dict]), f"NaN values in wx table: {self.wx_dict} at node {u}"

    def _update_trick(self, u) -> dict:
        self.logger.debug(f"\t- Updating Wx table removing node {u} using trick...")
        wx_table = {}
        for (v, w) in self.wx_dict.keys():
            if v != u and w != u:
                try:
                    wx_table[v, w] = self.op.sub(self.wx_dict[v, w], self.op.div(self.op.mul([self.wx_dict[v, u], self.wx_dict[u, w]]), self.wx_dict[u, u]))
                except ValueError as ve:
                    self.logger.debug(f"Error updating Wx({v}, {w}) = Wx({v}, {w}) - Wx({v}, {u}) * Wx({u}, {w}) / Wx({u}, {u})")
                    self.logger.debug(f"wx({v}, {w}) = {self.wx_dict[v, w]}")
                    self.logger.debug(f"wx({v}, {u}) = {self.wx_dict[v, u]}")
                    self.logger.debug(f"wx({u}, {w}) = {self.wx_dict[u, w]}")
                    self.logger.debug(f"wx({u}, {u}) = {self.wx_dict[u, u]}")
                    raise ve
                # log: logsubexp(self.wx[v, w], self.wx[v, u] + self.wx[u, w] - self.wx[u, u])
                self.logger.debug(f"\t- Updated Wx({v}, {w}) = Wx({v}, {w})({self.wx_dict[v, w]})"
                              f" - Wx({v}, {u})({self.wx_dict[v, u]}) * Wx({u}, {w})({self.wx_dict[u, w]}) / Wx({u}, {u})({self.wx_dict[u, u]}) = {wx_table[v, w]}")

        if self.cache_size > 1:
            self._update_cache(wx_table, self.x)
        return wx_table

    def get_wx(self):
        return self.wx_dict

    def to_array(self):
        wx_arr = wx_dict_to_array(self.wx_dict, self.weights.shape[0])
        return wx_arr

    def reset(self):
        self.x = self.complete_x.copy()
        self.wx_dict = self._complete_table.copy()

    def to_log_array(self):
        """
        Return the Wx table as a numpy array in log scale, leaving 0 values as 0 instead of - inf
        """
        wxarr = self.to_array()
        wxarr[wxarr == 0] = 1
        return np.log(wxarr)

    def _update_cache(self, wx, x):
        """
        Update the cache with the new Wx table and the current x set.
        :param wx: dict, new Wx table
        :param x: list, current x set
        """
        x = tuple(sorted(x))
        if x in self._cached_tables:
            self._cache_hits[x] += 1
        elif len(self._cached_tables) < self.cache_size:
            assert x not in self._cached_tables, f"Cache already contains x set {x}"
            self._cached_tables[x] = wx
            self._cache_hits[x] = 1
        else:
            # replace least used cache if cache is full and new x set occurred more times than the least used cache
            # check least used cached tables and check x hits
            min_cached = min(self._cached_tables, key=lambda k: self._cache_hits[k])
            x_hits = self._cache_hits.get(x, 0) + 1
            min_hits = self._cache_hits[min_cached]
            # if x has more hits than the least used cache, replace the least used cache
            if x_hits > min_hits:
                del self._cached_tables[min_cached]
                self._cached_tables[x] = wx

            # update hits regardless of the cache update so that eventually the least used cache is replaced
            self._cache_hits[x] = x_hits


def wx_dict_to_array(wx_dict: dict, n_nodes: int) -> np.ndarray:
    """
    Convert a Wx dictionary to a numpy array. Nodes that are not in the table are set to 0.
    :param wx_dict: dict, current Wx table with tuple (u, v) keys
    :param n_nodes: int, total number of nodes in the graph
    :return: np.ndarray, Wx table as a matrix
    """
    wx_arr = np.zeros((n_nodes, n_nodes))
    for (i, j), v in wx_dict.items():
        wx_arr[i, j] = v
    return wx_arr

class CastawayRST(TreeSampler):

    def __init__(self, graph: nx.DiGraph | np.ndarray, root: int, log_probs: bool = False, trick: bool = True, **kwargs):
        """
        Initialize the Castaway Random Spanning Tree sampler.
        :param graph: nx.DiGraph with weights on arcs, or np.ndarray matrix of graph weights
        :param root: label of the root in the graph
        :param log_probs: boolean, if the graph has log-weights
        :param trick: boolean, if True, the algorithm runs in O(n^3) by efficiently updating the W table. Otherwise W
            is computed from scratch every time a new arc is added to the tree
        :param kwargs: additional arguments
        """
        super().__init__(graph, root, log_probs, **kwargs)
        self.trick = trick
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.debug = kwargs.get('debug', False)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        #self._adjust_graph()
        # initialize x set to all nodes except the root
        self.x_list = list(x for x in range(self.weights.shape[0]) if x != self.root)
        self.wx = WxTable(self.x_list, self.weights, root=self.root, log_probs=self.log_probs, debug=self.debug, cache_size=kwargs.get('cache_size', 1))

    def sample(self, n_samples: int = 1) -> dict:
        """
        Sample n_samples trees from the graph and return the tree occurrences.
        :param n_samples: number of trees to sample
        :return: dictionary with tree occurrences
        """
        trees = {}
        for _ in range(n_samples):
            tree = tuple(self.castaway_rst())
            if tree in trees:
                trees[tree] += 1
            else:
                trees[tree] = 1
        return trees

    def sample_tree(self) -> nx.DiGraph:
        tree = self.castaway_rst()

        tree_nx = nx.DiGraph()
        for j, i in enumerate(tree):
            if i != -1:
                tree_nx.add_weighted_edges_from([(i, j, {'weight': self.weights[i, j]})])
        assert nx.is_arborescence(tree_nx)
        return  tree_nx

    def sample_tree_as_list(self) -> list[int]:
        tree = self.castaway_rst()
        return tree

    def castaway_rst(self) -> list[int]:
        """
        Wrapper for the original random spanning tree sampler inspired by Wilson algorithm.
        """
        # reset wx table
        self.x_list = list(set(self.graph.nodes()).difference([self.root]))
        self.wx.reset()

        tree = self._castaway()
        return tree

    def _castaway(self) -> list[int]:
        """
        Sample one tree from a given graph with fast arborescence sampling algorithm.
        :return: list of length num_nodes with parent idx for each node. root node has -1
        """
        tree = [-1] * self.weights.shape[0]

        dangling_path: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
        # iterate for each node
        self.logger.debug(f"Starting CastawayRST with x set: {self.wx.x} ({len(self.wx.x)} nodes)")
        while len(self.wx.x) > 0:
            self.logger.debug(f"*** Remaining: {len(self.wx.x)}, x: {self.wx.x} ***")
            # choose new i from x if no dangling nodes
            self.logger.debug("Picking new node i...")
            if not dangling_path:
                # NOTE: stochastic vs sorted choice (should not matter)
                i_vertex = np.random.choice(self.wx.x)
                # i_vertex = self.wx.x[0]
                self.logger.debug(f"- No dangling path, picking node {i_vertex} from x set")
            # or set i to be last node
            else:
                latest_edge = dangling_path[-1]
                i_vertex = latest_edge[0]
                self.logger.debug(f"- Attaching dangling path {dangling_path} at node {i_vertex}")

            # update Wx table and remove x from x set
            self.logger.debug(f"Updating Wx table and removing node {i_vertex} from x set...")
            self.wx.update(i_vertex, trick=self.trick)

            # for each node in either S or X - u, compute the probability of direct arc u -> i
            # nodes_lab, w_choice = self._compute_lexit_table(i_vertex, self.wx.x, self.wx.get_wx(), tree)
            # O(n^2)
            tree_nodes = [j for j, i in enumerate(tree) if i != -1] + [self.root]
            u_vertex = self._pick_u(i_vertex, tree_nodes)
            dangling_path.append((u_vertex, i_vertex))
            if u_vertex in tree_nodes:
                # if u picked from tree, attach dangling path and reset
                # add dangling path edges to tree
                for i, j in dangling_path:
                    tree[j] = i
                dangling_path = []

        # no more nodes in x set, the tree is complete
        return tree

    def _pick_u(self, i: int, tree_nodes: list) -> int:
        """
        Compute the probability of a direct connection in the tree
        from any node (u) in the tree or in the x set to the new node i.
        Then sample the next node u proportionally to the computed probabilities.
        :param i: exit node
        :param tree_nodes: list, nodes in the tree
        :return: int, str - u node label and origin ('t' or 'x')
        """
        assert i not in self.wx.x

        w_choice = self._compute_lexit_probs(i, tree_nodes)

        # pick u proportionally to lexit_i(u)
        u_vertex = self.op.random_choice(w_choice) # random choice (if log_probs, uses gumbel trick)
        self.logger.debug(f"- Picking node u = {u_vertex} from w_choice: {w_choice} (origin: {'t' if u_vertex in tree_nodes else 'x'})")

        return u_vertex

    def _compute_lexit_probs(self, i, tree_nodes) -> np.ndarray:
        """
        Compute the probability of a direct connection from any node (u) in the tree or in the x set to the new node i.
        :param i: int, exit node
        :param tree_nodes: list, nodes already in the tree
        :return: weights for random choice at each node
        """
        gw = self.weights
        w_choice = [self.op.zero()] * self.weights.shape[0]

        # probability of any u in V(T) U X to be the next connection to i
        # attachment from tree to any node in X
        pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
        # O(n^2)
        for v in self.wx.x:
            pattach[v] = self.op.add([gw[w, v] for w in tree_nodes])
        # for tree nodes as origin, probability is barely the weight of the arc
        for u in tree_nodes:
            w_choice[u] = gw[u, i]
        # for x nodes as origin, probability is the sum of all paths from any w in T to u (through any v in X)
        # O(n^2)
        for u in self.wx.x:
            # lexit_i(u) = (sum_{v in X} (sum_{w in T} Ww,v) * Wv,u ) * p(u, i)
            # any w, v (from tree to X) * any v, u (from X to i) * p(u, i)
            p_tree_u_i = self.op.mul(
                [self.op.add([self.op.mul([pattach[v], self.wx.wx_dict[v, u]]) for v in self.wx.x]), gw[u, i]])
            w_choice[u] = p_tree_u_i
        # print(f"unnormalized choices:{w_choice} {nodes}, i: {i}")
        return self.op.normalize(w_choice)

def importance_sample(matrix, n, temp=1., log_probs: bool = False, trick: bool = False):
    """
    Importance sampling for random spanning trees.
    :param matrix: np.ndarray, matrix of weights
    :param n: int, number of samples
    :param temp: float, temperature parameter, will apply w^(1 / temp) to the weights
    :param log_probs: bool, whether weights matrix values are in log scale, if True, then w' = w / temp
    :return: dict, (tree_tuple, iw) where tree_tuple is the parent-list representation of the tree
        (i.e. t_i is parent of node i, if t_i = -1, node i is the root)
    """
    op = StableOp(log_probs)
    orig_w = matrix.copy()
    matrix = op.temper(matrix, temp)
    sampler = CastawayRST(matrix, root=0, log_probs=log_probs, trick=trick)
    trees = {}
    for _ in range(n):
        tree = sampler.castaway_rst()
        iw = op.div(tree_weight(tree, orig_w, log_probs=log_probs), tree_weight(tree, matrix, log_probs=log_probs))
        t_id = tuple(tree)
        trees[t_id] = op.add([trees.get(t_id, op.zero()), iw])

    return trees


def check():
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
        sampler = CastawayRST(X, root=0, trick=False)
        dist = {}
        for i in range(N):
            tree = tuple(sampler.sample_tree_as_list())
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

def check_log():
    print("Testing CastawayRST with log probabilities...")
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
        sampler = CastawayRST(np.log(X), root=0, trick=False, log_probs=True)
        dist = {}
        for i in range(N):
            tree = tuple(sampler.sample_tree_as_list())
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")

def check_trick():
    print("Testing CastawayRST trick -> O(n^3)...")
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
        sampler = CastawayRST(X, root=0, trick=True)
        dist = {}
        for i in range(N):
            tree = tuple(sampler.sample_tree_as_list())
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")


def check_log_trick():
    print("Testing CastawayRST trick -> O(n^3) with log probabilities...")
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
        sampler = CastawayRST(np.log(X), root=0, trick=True, log_probs=True)
        dist = {}
        for i in range(N):
            tree = tuple(sampler.sample_tree_as_list())
            if tree not in dist:
                dist[tree] = 0
            dist[tree] += 1 / N

        for tree in dist:
            prob = tree_weight(tree, X) / Z
            acc += 1 if np.isclose(dist[tree], prob, rtol=.1) else 0
            # print(f"tree: {tree}, prob: {prob}, freq: {dist[tree]}")
    print(acc / (len(dist) * n_seeds) * 100, "% of trees have been sampled correctly")


if __name__ == '__main__':
    check_log_trick()
    # # debug algorithm steps
    # np.random.seed(42)
    # # random graph
    # g = random_uniform_graph(5, log_probs=False, normalize=True)
    # sampler = CastawayRST(graph=g, root=0, log_probs=False, trick=True, debug=True)
    # logging.debug("Running CastawayRST on random graph (non log) with trick")
    # print("Wx matrix")
    # print(sampler.wx.to_array())
    # tree = sampler.sample_tree()
    # logging.debug(f"tree_to_newick(tree): {tree_to_newick(tree)}")





