import itertools
import logging

import networkx as nx
import numpy as np

from treesampling import TreeSampler
from treesampling.utils.graphs import normalize_graph_weights, tree_to_newick, random_uniform_graph
from treesampling.utils.math import StableOp


class WxTable:
    def __init__(self, x: list, graph: nx.DiGraph, log_probs: bool = False, **kwargs):
        self.op = StableOp(log_probs=log_probs)  # dispatch stable operations
        self.x = x
        self.graph = graph
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.debug = kwargs.get('debug', False)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

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
        gw = nx.to_numpy_array(self.graph)

        # base step: x = [v] (first node)
        v = self.x[0]
        wx = {(v, v): self.op.one()}
        self.logger.debug(f"\t- Initializing Wx table with node {v}: wx = {wx}")
        for i in range(1, len(self.x)):
            x = self.x[:i]
            u = self.x[i]
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
        if not self.x:
            self.wx_dict = {}
        elif trick:
            self.wx_dict = self._update_trick(u)
        else:
            self.wx_dict = self._build()
        # no nans allowed
        assert not np.any([np.isnan(self.wx_dict[k]) for k in self.wx_dict]), f"NaN values in wx table: {self.wx_dict} at node {u}"

    def _update_trick(self, u) -> dict:
        self.logger.debug(f"\t- Updating Wx table removing node {u} using trick...")
        wx_table = {}
        for (v, w) in self.wx_dict.keys():
            if v != u and w != u:
                wx_table[v, w] = self.op.sub(self.wx_dict[v, w], self.op.div(self.op.mul([self.wx_dict[v, u], self.wx_dict[u, w]]), self.wx_dict[u, u]))
                # log: logsubexp(self.wx[v, w], self.wx[v, u] + self.wx[u, w] - self.wx[u, u])
                self.logger.debug(f"\t- Updated Wx({v}, {w}) = Wx({v}, {w})({self.wx_dict[v, w]})"
                              f" - Wx({v}, {u})({self.wx_dict[v, u]}) * Wx({u}, {w})({self.wx_dict[u, w]}) / Wx({u}, {u})({self.wx_dict[u, u]}) = {wx_table[v, w]}")
        return wx_table

    def get_wx(self):
        return self.wx_dict

    def to_array(self):
        wx_arr = wx_dict_to_array(self.wx_dict, self.graph.number_of_nodes())
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

    def __init__(self, graph: nx.DiGraph, root: int, log_probs: bool = False, trick: bool = True, **kwargs):
        """
        Initialize the Castaway Random Spanning Tree sampler.
        :param graph: nx.DiGraph with weights on arcs
        :param root: label of the root in the graph
        :param log_probs: boolean, if the graph has log-weights
        :param trick: boolean, if True, the algorithm runs in O(n^3) by efficiently updating the W table. Otherwise W
            is computed from scratch every time a new arc is added to the tree
        :param kwargs: additional arguments
        """
        super().__init__(graph, root, log_probs, **kwargs)
        self.op = StableOp(log_probs=self.log_probs)
        self.trick = trick
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.debug = kwargs.get('debug', False)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        self._adjust_graph()
        # initialize x set to all nodes except the root
        self.x_list = list(set(self.graph.nodes()).difference([self.root]))
        self.wx = WxTable(self.x_list, self.graph, log_probs=self.log_probs, debug=self.debug)

    def _adjust_graph(self):
        # add missing edges with null weight and normalize
        missing_edges = nx.difference(nx.complete_graph(self.graph.number_of_nodes()), self.graph)
        self.graph.add_edges_from([(u, v, {'weight': self.op.zero()}) for u, v in missing_edges.edges()])
        not_normalized = False
        # if weights are normalized, check for forced edges with weight 1
        for i in range(self.graph.number_of_nodes()):
            # when sum is close to one, and one single edge has weight 1, then
            # all other edges should never be sampled (set their weight to 0)
            if np.isclose(self.op.add(nx.to_numpy_array(self.graph)[:, i].tolist()), self.op.one()):
                # check whether there is a node with weight 1.
                if np.any(np.isclose(nx.to_numpy_array(self.graph)[:, i], self.op.one())):
                    j = np.argwhere(np.isclose(nx.to_numpy_array(self.graph)[:, i], self.op.one()))[0]
                    self.logger.warning(f"edge {j} -> {i} has weight 1.0, all other edges in node {i} will be set to 0.")
                    for k in range(self.graph.number_of_nodes()):
                        if k != j:
                            self.graph.edges()[k, i]['weight'] = self.op.zero()
            else:
                not_normalized = True
        # normalize graph weights
        if not_normalized:
            self.graph = normalize_graph_weights(self.graph, log_probs=self.log_probs)

    def sample(self, n_samples: int = 1) -> dict:
        """
        Sample n_samples trees from the graph and return the tree occurrences.
        :param n_samples: number of trees to sample
        :return: dictionary with tree occurrences
        """
        trees = {}
        for _ in range(n_samples):
            tree = self.castaway_rst()
            tree_str = tree_to_newick(tree)
            if tree_str in trees:
                trees[tree_str] += 1
            else:
                trees[tree_str] = 1
        return trees

    def sample_tree(self) -> nx.DiGraph:
        return self.castaway_rst()

    def castaway_rst(self) -> nx.DiGraph:
        """
        Wrapper for the original random spanning tree sampler inspired by Wilson algorithm.
        :param graph: nx.DiGraph, with weights on arcs
        :param root: label of the root in the graph
        :param log_probs: if the graph has log-weights
        :param trick: if True, the algorithm runs in O(n^3) by efficiently updating the W table. Otherwise W
            is computed from scratch every time a new arc is added to the tree
        :return:
        """
        # reset wx table
        self.x_list = list(set(self.graph.nodes()).difference([self.root]))
        self.wx.reset()

        tree = self._castaway()
        assert nx.is_arborescence(tree), "Tree is not an arborescence"
        return tree

    def _castaway(self) -> nx.DiGraph:
        """
        Sample one tree from a given graph with fast arborescence sampling algorithm.
        :return: nx.DiGraph, arborescence T with P(T) \propto w(T)
        """
        # initialize tree with root
        tree = nx.DiGraph()
        tree.add_node(self.root)

        dangling_path: list[tuple] = []  # store dangling path branch (not yet attached to tree, not in X)
        # iterate for each node
        self.logger.debug(f"Starting CastawayRST with x set: {self.wx.x} ({len(self.wx.x)} nodes)")
        while len(self.wx.x) > 0:
            self.logger.debug(f"*** Remaining: {len(self.wx.x)}, x: {self.wx.x} ***")
            # choose new i from x if no dangling nodes
            self.logger.debug("Picking new node i...")
            if not dangling_path:
                # NOTE: stochastic vs sorted choice (should not matter)
                # i_vertex = random.choice(self.wx.x)
                i_vertex = self.wx.x[0]
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
            u_vertex, origin_lab = self._pick_u(i_vertex, [n for n in tree.nodes()])
            dangling_path.append((u_vertex, i_vertex, self.graph.edges()[u_vertex, i_vertex]['weight']))
            if origin_lab == 't':
                # if u picked from tree, attach dangling path and reset
                # add dangling path edges to tree
                tree.add_weighted_edges_from(dangling_path)
                dangling_path = []

        # no more nodes in x set, the tree is complete
        return tree

    def _pick_u(self, i: int, tree_nodes: list) -> tuple[list, np.ndarray]:
        """
        Compute the probability of a direct connection in the tree
        from any node (u) in the tree or in the x set to the new node i.
        Then sample the next node u proportionally to the computed probabilities.
        :param i: exit node
        :param tree_nodes: list, nodes in the tree
        :return: int, str - u node label and origin ('t' or 'x')
        """
        assert i not in self.wx.x

        w_choice, nodes = self._compute_lexit_probs(i, tree_nodes)

        # pick u proportionally to lexit_i(u)
        u_idx = self.op.random_choice(w_choice) # random choice (if log_probs, uses gumbel trick)
        u_vertex, origin_lab = nodes[u_idx]
        self.logger.debug(f"- Picking node u = {u_vertex} from w_choice: {w_choice} (origin: {origin_lab})")

        return u_vertex, origin_lab  # (u, origin) node label and origin ('t' or 'x')

    def _compute_lexit_probs(self, i, tree_nodes) -> tuple[np.ndarray, list]:
        """
        Compute the probability of a direct connection from any node (u) in the tree or in the x set to the new node i.
        :param i: int, exit node
        :param tree_nodes: list, nodes already in the tree
        :return: tuple, weights for random choice and list of nodes label and source str ('t' or 'x')
        """
        gw = nx.to_numpy_array(self.graph)
        nodes = []  # tuples (node, source) - source can be 'x' or 't'
        w_choice = []  # weights for random choice at each node

        # probability of any u in V(T) U X to be the next connection to i
        # attachment from tree to any node in X
        pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
        # O(n^2)
        for v in self.wx.x:
            pattach[v] = self.op.add([gw[w, v] for w in tree_nodes])
        # for tree nodes as origin, probability is barely the weight of the arc
        for u in tree_nodes:
            nodes.append((u, 't'))
            w_choice.append(gw[u, i])
        # for x nodes as origin, probability is the sum of all paths from any w in T to u (through any v in X)
        # O(n^2)
        for u in self.wx.x:
            # lexit_i(u) = (sum_{v in X} (sum_{w in T} Ww,v) * Wv,u ) * p(u, i)
            # any w, v (from tree to X) * any v, u (from X to i) * p(u, i)
            p_tree_u_i = self.op.mul(
                [self.op.add([self.op.mul([pattach[v], self.wx.wx_dict[v, u]]) for v in self.wx.x]), gw[u, i]])
            nodes.append((u, 'x'))
            w_choice.append(p_tree_u_i)
        # print(f"unnormalized choices:{w_choice} {nodes}, i: {i}")
        w_choice = self.op.normalize(w_choice)
        return w_choice, nodes


if __name__ == '__main__':
    # debug algorithm steps
    np.random.seed(42)
    # random graph
    g = random_uniform_graph(5, log_probs=False, normalize=True)
    sampler = CastawayRST(graph=g, root=0, log_probs=False, trick=True, debug=True)
    logging.debug("Running CastawayRST on random graph (non log) with trick")
    print("Wx matrix")
    print(sampler.wx.to_array())
    tree = sampler.sample_tree()
    logging.debug(f"tree_to_newick(tree): {tree_to_newick(tree)}")





