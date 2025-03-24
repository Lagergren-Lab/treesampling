"""
This module contains the abstract class TreeSampler, which is the base class for all tree sampling algorithms.
"""

import networkx as nx
import numpy as np

from treesampling.utils.math import StableOp


class TreeSampler:
    def __init__(self, graph: nx.DiGraph | np.ndarray, root: int, log_probs: bool = False, **kwargs):
        if isinstance(graph, nx.DiGraph):
            graph = graph.copy()
            weights = nx.to_numpy_array(graph)
        elif isinstance(graph, np.ndarray):
            weights = graph.copy()
            graph = nx.from_numpy_array(graph)
        else:
            raise TypeError("graph is not a DiGraph or np.ndarray")
        self.op = StableOp(log_probs=log_probs)
        # remove root incoming arcs and self-loops
        weights[:, root] = self.op.one() # init root weights to one to avoid division by zero in normalization
        weights[np.diag_indices(weights.shape[0])] = self.op.zero()
        # normalize weights
        weights = self.op.normalize(weights, axis=0)
        # set root to zero
        weights[:, root] = self.op.zero()

        # TODO: graph input should be removed, not handled well for logprob matrix due to nx.from_numpy_array not reading zeros

        self.graph = graph
        self.weights = weights
        self.root = root
        self.log_probs = log_probs
        self.kwargs = kwargs

    def sample(self, n_samples: int) -> list:
        raise NotImplementedError("This method should be implemented in the subclass")

    def sample_tree(self) -> nx.DiGraph:
        raise NotImplementedError("This method should be implemented in the subclass")
