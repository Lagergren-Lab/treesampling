"""
This module contains the abstract class TreeSampler, which is the base class for all tree sampling algorithms.
"""

import networkx as nx

class TreeSampler:
    def __init__(self, graph: nx.DiGraph, root: int, log_probs: bool = False, **kwargs):
        self.graph = graph
        self.root = root
        self.log_probs = log_probs
        self.kwargs = kwargs

    def sample(self, n_samples: int) -> list:
        raise NotImplementedError("This method should be implemented in the subclass")

    def sample_tree(self) -> nx.DiGraph:
        raise NotImplementedError("This method should be implemented in the subclass")
