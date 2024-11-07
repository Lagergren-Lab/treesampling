import itertools

import networkx as nx
import numpy as np

from treesampling import TreeSampler
from treesampling.utils.graphs import normalize_graph_weights, tree_to_newick
from treesampling.utils.math import gumbel_max_trick_sample, logdiffexp, StableOp


class WxTable:
    def __init__(self, x: list, graph: nx.DiGraph, log_probs: bool = False):
        self.log_probs = log_probs
        self.x = x
        self.graph = graph
        self.wx = self._build()

    def _build(self) -> dict:
        op = StableOp(log_probs=self.log_probs)  # dispatch stable operations
        # graph weights
        gw = nx.to_numpy_array(self.graph)

        # base step: x = [v] (first node)
        v = self.x[0]
        wx = {(v, v): op.one()}
        for i in range(1, len(self.x)):
            x = self.x[:i]
            u = self.x[i]
            # Y = X U { u }
            wy = {}
            # compute Ry(u) where u is Y \ X (u)
            ry_1 = op.zero() # log: -np.inf
            # marginalize over all paths from v to w
            for v, w in itertools.product(x, repeat=2):
                ry_1 = op.add([ry_1, op.mul([gw[u, v], wx[v, w], gw[w, u]])])
                # log: logaddexp(ry_1, gw[u, v] + wx[v, w] + gw[w, u])
            ry = op.div(op.one(), op.sub(op.one(), ry_1))
            # log: -np.log(1 - np.exp(ry_1))

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
                    wxy[v] = op.add([wxy[v], op.mul([gw[vv, u], wx[v, vv]])]) # log: logaddexp(wxy[v], gw[vv, u] + wx[v, vv])
                    wyx[v] = op.add([wyx[v], op.mul([wx[vv, v], gw[u, vv]])]) # log: logaddexp(wyx[v], wx[vv, v] + gw[u, vv])

            # write new W table
            for v in x:
                wy[u, v] = op.mul([ry, wyx[v]]) # log: wy[u, v] = ry + wyx[v]
                wy[v, u] = op.mul([wxy[v], ry]) # log: wy[v, u] = wxy[v] * ry
                for w in x:
                    # general update
                    wy[v, w] = op.add([wx[v, w], op.mul([wxy[v], ry, wyx[w]])]) # log: wy[v, w] = logaddexp(wx[v, w], wxy[v] + ry + wyx[w])
            # new self returning random path
            wy[u, u] = ry

            # update wx
            wx = wy
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
            self.wx = {}
        elif trick:
            self.wx = self._update_trick(u)
        else:
            self.wx = self._build()
        # no nans allowed
        assert not np.any([np.isnan(self.wx[k]) for k in self.wx]), f"NaN values in wx table: {self.wx} at node {u}"

    def _update_trick(self, u) -> dict:
        op = StableOp(log_probs=self.log_probs)  # dispatch stable operations
        wx_table = {}
        for (v, w) in self.wx.keys():
            if v != u and w != u:
                wx_table[v, w] = op.sub(self.wx[v, w], op.div(op.mul([self.wx[v, u], self.wx[u, w]]), self.wx[u, u]))
                # log: logsubexp(self.wx[v, w], self.wx[v, u] + self.wx[u, w] - self.wx[u, u])
        return wx_table

    def get_wx(self):
        return self.wx


class CastawayRST(TreeSampler):

    def __init__(self, graph: nx.DiGraph, root: int, log_probs: bool = False, trick: bool = False, **kwargs):
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
        self.trick = trick
        self._adjust_graph()

        # initialize x set to all nodes except the root
        self.x_list = list(set(self.graph.nodes()).difference([self.root]))
        self.wx = WxTable(self.x_list, self.graph, log_probs=self.log_probs)

    def _adjust_graph(self):
        # add missing edges with null weight and normalize
        missing_edges = nx.difference(nx.complete_graph(self.graph.number_of_nodes()), self.graph)
        zero_weight = 0. if not self.log_probs else -np.inf
        self.graph.add_edges_from([(u, v, {'weight': zero_weight}) for u, v in missing_edges.edges()])
        # normalize graph weights
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
        self.wx = WxTable(self.x_list, self.graph, log_probs=self.log_probs)

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
        while len(self.wx.x) > 0:
            # choose new i from x if no dangling nodes
            if not dangling_path:
                # NOTE: stochastic vs sorted choice (should not matter)
                # i_vertex = random.choice(self.wx.x)
                i_vertex = self.wx.x[0]
            # or set i to be last node
            else:
                latest_edge = dangling_path[-1]
                i_vertex = latest_edge[0]

            # update Wx table and remove x from x set
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
        :param tree: nx.DiGraph, current tree
        :return: int, str - u node label and origin ('t' or 'x')
        """
        assert i not in self.wx.x
        op = StableOp(log_probs=self.log_probs)  # dispatch stable operations
        nodes = []  # tuples (node, source) - source can be 'x' or 't'
        w_choice = []  # weights for random choice at each node
        gw = nx.to_numpy_array(self.graph)

        # probability of any u in V(T) U X to be the next connection to i
        # attachment from tree to any node in X
        pattach = {}  # for each v in X, gives sum_{w in T} p(w, v)
        # O(n^2)
        for v in self.wx.x:
            pattach[v] = op.add([gw[w, v] for w in tree_nodes])

        # for tree nodes as origin, probability is barely the weight of the arc
        for u in tree_nodes:
            nodes.append((u, 't'))
            w_choice.append(gw[u, i])
        # for x nodes as origin, probability is the sum of all paths from any w in T to u (through any v in X)
        # O(n^2)
        for u in self.wx.x:
            # lexit_i(u) = (sum_{v in X} (sum_{w in T} Ww,v) * Wv,u ) * p(u, i)
            # any w, v (from tree to X) * any v, u (from X to i) * p(u, i)
            p_tree_u_i = op.mul([op.add([op.mul([pattach[v], self.wx.wx[v, u]]) for v in self.wx.x]), gw[u, i]])
            nodes.append((u, 'x'))
            w_choice.append(p_tree_u_i)
        w_choice = op.normalize(w_choice)

        # pick u proportionally to lexit_i(u)
        u_idx = op.random_choice(w_choice) # random choice (if log_probs, uses gumbel trick)
        u_vertex, origin_lab = nodes[u_idx]

        return u_vertex, origin_lab  # (u, origin) node label and origin ('t' or 'x')
