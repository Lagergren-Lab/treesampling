"""
Implementation which consider components and a crasher node
"""
import itertools
import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

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
        self.crashers = []  # nodes that cause numerical issues

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
        self._init_x = self.x.copy()  # initial x
        self._init_crashers = self.crashers.copy()  # initial crashers
        self._init_table = self.wx_dict.copy()

    @property
    def init_table(self):
        return self._init_table

    @property
    def init_x(self):
        return self._init_x

    @property
    def init_crashers(self):
        return self._init_crashers

    def _build(self) -> (dict, list):
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
        while i < len(x_list):
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
            if np.isinf(ry):
                self.logger.debug(f"\t- Crasher node found: Ry_1({u}) = {ry_1} => Ry({u}) = {ry}")
                self.x.remove(u)
                self.crashers.append(u)
            else:
                # update table with u node
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
                    self.logger.error(f"Error updating Wx({v}, {w}) = Wx({v}, {w}) - Wx({v}, {u}) * Wx({u}, {w}) / Wx({u}, {u})")
                    self.logger.error(f"wx({v}, {w}) = {self.wx_dict[v, w]}")
                    self.logger.error(f"wx({v}, {u}) = {self.wx_dict[v, u]}")
                    self.logger.error(f"wx({u}, {w}) = {self.wx_dict[u, w]}")
                    self.logger.error(f"wx({u}, {u}) = {self.wx_dict[u, u]}")
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
        self.x = self.init_x.copy()
        self.crashers = self.init_crashers.copy()
        self.wx_dict = self._init_table.copy()

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

    # UNUSED
    # TODO: remove
    def _contract_matrix(self, x: list, u: int):
        """
        Once a transient node is found, the matrix is contracted by finding the arc that must be contracted to avoid
        numerical issues.
        """
        # total time complexity: O(n^3)
        # find mst
        graph = nx.DiGraph()
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if i != j and j != self.root:
                    graph.add_edge(i, j, weight=self.weights[i, j])
        mst = nx.maximum_spanning_arborescence(graph)
        mst_weight = self.op.mul([self.weights[i, j] for i, j in mst.edges()])
        self.logger.debug(f"MST found: {tree_to_newick(mst)} with weight {mst_weight}")
        alt_msts = {}

        # find arc that must be contracted
        contracted_arc = None
        mst_weight_ratio = self.op.one()  # gain in including i, j arc in mst
        for i, j in mst.edges():
            if i == u or (i != self.root and j == u):
                self.logger.debug(f"Attempting to remove arc {i} -> {j}...")
                # remove arc i, j and compute mst weight
                alt_graph = graph.copy()
                alt_graph.remove_edge(i, j)
                alt_mst_weights = nx.maximum_spanning_arborescence(alt_graph)  # O(n^2)
                alt_msts[(i, j)] = self.op.mul([self.weights[ii, jj] for ii, jj in alt_mst_weights.edges()])
                if self.op.div(mst_weight, alt_msts[(i, j)]) > mst_weight_ratio:
                    mst_weight_ratio = self.op.div(mst_weight, alt_msts[(i, j)])
                    contracted_arc = (i, j)

        if contracted_arc is None:
            self.logger.error(f"Could not find arc to contract for transient node {u}")
            raise ValueError(f"Could not find arc to contract for transient node {u}")

        self.logger.warning(f"Contracting arc {contracted_arc} to avoid transient node {u}")
        # contract arc
        i, j = contracted_arc
        self.weights[:, j] = self.op.zero()
        self.weights[i, j] = self.op.one()
        self.weights[j, i] = self.op.zero()
        # re-normalize column i
        self.weights[:, i] = self.op.normalize(self.weights[:, i], axis=0)


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
            tree = self.sample_tree()
            tree_str = tree_to_newick(tree)
            if tree_str in trees:
                trees[tree_str] += 1
            else:
                trees[tree_str] = 1
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
        self.logger.debug(f"Starting CastawayRST with x set: {self.wx.x} ({len(self.wx.x)} nodes) and crasher(s): {self.wx.crashers}")
        candidates = self.wx.x.copy() + self.wx.crashers
        while len(self.wx.x) > 0:
            tree_nodes = [j for j, i in enumerate(tree) if i != -1] + [self.root]
            self.logger.debug(f"*** Remaining: {len(candidates)}, x + crasher(s): {candidates} ***")
            # choose new i from x if no dangling nodes
            self.logger.debug(f"Picking new node i... [x set: {self.wx.x}, crashers: {self.wx.crashers}, dangling path: {dangling_path}, tree: {tree_nodes}]")
            if not dangling_path:
                # NOTE: stochastic vs sorted choice (should not matter)
                i_vertex = np.random.choice(candidates)
                candidates.remove(i_vertex)
                # i_vertex = self.wx.x[0]
                self.logger.debug(f"- No dangling path, picking node {i_vertex} from x set")
            # or set i to be last node
            else:
                latest_edge = dangling_path[-1]
                i_vertex = latest_edge[0]
                self.logger.debug(f"- Proceed from dangling path {dangling_path} at node i={i_vertex}")

            if i_vertex not in self.wx.crashers:
                # update Wx table and remove x from x set
                self.logger.debug(f"Updating Wx table and removing node {i_vertex} from x set...")
                self.wx.update(i_vertex, trick=self.trick)
            else:
                # remove crasher node from crasher set
                self.wx.crashers.remove(i_vertex)
                self.logger.debug(f"RW starting from crasher node {i_vertex}, removing node {i_vertex} from crasher set... (remaining: {self.wx.crashers})")

            # for each node in X, compute the probability of a direct connection in the tree
            u_vertex = self._pick_u(i_vertex, tree_nodes)
            dangling_path.append((u_vertex, i_vertex))
            if u_vertex in candidates:
                candidates.remove(u_vertex)

            self.logger.debug(f"- Dangling path: {dangling_path}, tree nodes: {tree_nodes}, x set: {self.wx.x}, crashers: {self.wx.crashers}")

            if u_vertex in tree_nodes:
                # if u picked from tree, attach dangling path and reset
                # add dangling path edges to tree
                self.logger.debug(f"- Attaching dangling path {dangling_path} to tree with arc {u_vertex} -> {i_vertex}")
                for i, j in dangling_path:
                    tree[j] = i
                dangling_path = []

        # no more nodes in x set, the tree is complete
        # check if tree is valid
        if self.debug:
            assert len(tree) == self.weights.shape[0], f"Tree has {len(tree)} nodes, expected {self.weights.shape[0]}"
            assert tree.count(-1) == 1, f"Tree has {tree.count(-1)} roots, expected 1"
            for i, p in enumerate(tree):
                if p != -1:
                    assert p < self.weights.shape[0], f"Parent node {p} of node {i} is out of bounds (max {self.weights.shape[0] - 1})"
        self.logger.debug(f"- Tree sampled: {tree}\n***\n")
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
        self.logger.debug(f"- Picking node u = {u_vertex} from w_choice: {w_choice} (origin: {'t' if u_vertex in tree_nodes else 'x' if u_vertex in self.wx.x else 'crasher'})")

        return u_vertex

    def _compute_lexit_probs(self, i, tree_nodes) -> np.ndarray:
        """
        Compute the probability of a direct connection from any node (u) in the tree or in the x set to the new node i.
        :param i: int, exit node
        :param tree_nodes: list, nodes already in the tree
        :return: weights for random choice at each node
        """
        gw = self.weights
        PX = self.wx.wx_dict
        w_choice = [self.op.zero()] * self.weights.shape[0]
        self.logger.debug(f"Computing lexit_i(u) for node {i}... {'[CRASHER]' if i in self.wx.crashers else ''}")

        # probability of any u in V(T) U X to be the next connection to i
        # attachment from tree to any node in X
        OT = {}  # for each v in X U {c}, probability of path Out of T to v (walk from v to T directly)
        # O(n^2)
        for v in self.wx.x + self.wx.crashers:
            OT[v] = self.op.add([gw[w, v] for w in tree_nodes])

        # compute the probability of walking from a node v in X to T (no crasher)
        RT = {}
        for v in self.wx.x:
            RT[v] = self.op.add([self.op.mul([OT[w], PX[w, v]]) for w in self.wx.x])

        # compute the probability of walking from a node v in X to T, visiting the crasher (ScT(v))
        ST = {}
        RcXT = {}
        for c in self.wx.crashers:
            # from crasher to X, T or from crasher straight to T
            RcXT[c] = self.op.add([self.op.mul([gw[w, c], RT[w]]) for w in self.wx.x])
            RTc = self.op.add([OT[c], RcXT[c]])
            for v in self.wx.x:
                ST[c,v] = self.op.add([self.op.mul([PX[w, v], gw[c, w], RTc]) for w in self.wx.x])

        # for tree nodes as origin, probability is barely the weight of the arc
        for k in tree_nodes:
            # k in T
            w_choice[k] = gw[k, i]
        for k in self.wx.x:
            ScT = self.op.zero()
            for c in self.wx.crashers:
                ScT = self.op.add([ScT, ST[c, k]])
            # k in X
            w_choice[k] = self.op.mul([gw[k, i], self.op.add([RT[k], ScT])])
        for k in self.wx.crashers:
            # k = c
            w_choice[k] = self.op.mul([gw[k, i], self.op.add([RcXT[k], OT[k]])])
        self.logger.debug(f"Unnormalized choices:{w_choice} {self.wx.x}, i: {i}, crashers: {self.wx.crashers}")
        return self.op.normalize(w_choice)

            dangling_path = []
    return tree


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
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO)
    check_log()
    # check_log_trick()
    # # debug algorithm steps
    # # random graph
    # g = random_uniform_graph(5, log_probs=False, normalize=True)
    # sampler = CastawayRST(graph=g, root=0, log_probs=False, trick=True, debug=True)
    # logging.debug("Running CastawayRST on random graph (non log) with trick")
    # print("Wx matrix")
    # print(sampler.wx.to_array())
    # tree = sampler.sample_tree()
    # logging.debug(f"tree_to_newick(tree): {tree_to_newick(tree)}")





